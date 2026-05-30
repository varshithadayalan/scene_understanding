import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

class ResearchEngine:
    def __init__(self, nusc, model, config):
        self.nusc = nusc
        self.model = model
        self.cfg = config
        self.track_history = {} # track_id -> {node, age, active}

    def extract_nodes(self, curr_sample, prev_sample=None):
        prev_positions = {}
        if prev_sample:
            for ann_token in prev_sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                prev_positions[ann['instance_token']] = ann['translation']
        
        nodes = []
        for ann_token in curr_sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            pos = ann['translation']
            size = ann['size'] # [w, l, h]
            instance = ann['instance_token']
            
            if instance in prev_positions:
                prev_pos = prev_positions[instance]
                vx, vy = pos[0] - prev_pos[0], pos[1] - prev_pos[1]
            else:
                vx, vy = 0, 0
            
            visibility = int(ann['visibility_token'])
            confidence = visibility / 4.0
            uncertainty = 1 - confidence
            
            nodes.append({
                "instance": instance, "label": ann['category_name'],
                "position": pos, "vx": vx, "vy": vy,
                "size": size,
                "confidence": confidence, "uncertainty": uncertainty,
                "token": ann_token
            })
        return nodes

    def match(self, nodes_t1):
        """
        Trajectory-to-Frame Matching with Configurable Identity Persistence.
        """
        if not self.track_history:
            for node in nodes_t1:
                self.track_history[node["instance"]] = {"node": node, "age": 0, "active": True}
            return [], [], None

        track_ids = list(self.track_history.keys())
        active_tracks = [self.track_history[tid] for tid in track_ids]
        
        if not nodes_t1 or not active_tracks:
            return [], [], None
            
        cost_matrix = np.zeros((len(active_tracks), len(nodes_t1)))
        
        for i, track in enumerate(active_tracks):
            tid = track_ids[i]
            n_t = track["node"]
            
            # Position Extrapolation
            if track["age"] > 0:
                extrapolated_pos = [
                    n_t["position"][0] + (n_t["vx"] * track["age"]),
                    n_t["position"][1] + (n_t["vy"] * track["age"])
                ]
                n_t_eff = n_t.copy()
                n_t_eff["position"] = extrapolated_pos
            else:
                n_t_eff = n_t

            for j, n_t1 in enumerate(nodes_t1):
                with torch.no_grad():
                    # Adaptive Weighting using config
                    avg_unc = (n_t_eff["uncertainty"] + n_t1["uncertainty"]) / 2.0
                    alpha = self.cfg.alpha_base * (1.0 - avg_unc)
                    beta = self.cfg.beta_base * (1.0 + avg_unc)
                    gamma = self.cfg.gamma_base
                    
                    # 1. Embedding Cost
                    f1 = torch.tensor([n_t_eff["position"][0], n_t_eff["position"][1], n_t_eff["vx"], n_t_eff["vy"], n_t_eff["confidence"], n_t_eff["uncertainty"]], dtype=torch.float32)
                    f2 = torch.tensor([n_t1["position"][0], n_t1["position"][1], n_t1["vx"], n_t1["vy"], n_t1["confidence"], n_t1["uncertainty"]], dtype=torch.float32)
                    emb_dist = torch.norm(self.model.forward_embedding(f1) - self.model.forward_embedding(f2)).item()
                    
                    # 2. Motion Cost
                    dx, dy = n_t1["position"][0] - n_t_eff["position"][0], n_t1["position"][1] - n_t_eff["position"][1]
                    dist_norm = self.model.compute_distance(n_t_eff["position"], n_t1["position"]) / 20.0
                    same_label = 1 if n_t_eff["label"] == n_t1["label"] else 0
                    v_diff = self.model.compute_distance([n_t_eff["vx"], n_t_eff["vy"]], [n_t1["vx"], n_t1["vy"]])
                    
                    m_feat = torch.tensor([dx, dy, dist_norm, same_label, n_t_eff["confidence"], n_t1["confidence"], n_t_eff["uncertainty"], n_t1["uncertainty"], n_t_eff["vx"], n_t_eff["vy"], v_diff], dtype=torch.float32)
                    motion_score = self.model.forward_motion(m_feat).item()
                    motion_cost = 1.0 - motion_score
                    
                    # Hybrid Cost with Persistence Bias from config
                    cost = (alpha * emb_dist) + (beta * motion_cost) + (gamma * avg_unc)
                    
                    if n_t1["instance"] != tid:
                        cost += self.cfg.persistence_weight
                        
                    if n_t_eff["label"] != n_t1["label"]: 
                        cost += 10.0
                        
                    cost_matrix[i][j] = cost
                    
        r_ind, c_ind = linear_sum_assignment(cost_matrix)
        
        matched_tracks = set()
        matched_nodes = set()
        
        for r, c in zip(r_ind, c_ind):
            # Thresholding from config
            threshold = self.cfg.re_id_threshold + (active_tracks[r]["age"] * self.cfg.age_threshold_step)
            
            if cost_matrix[r][c] < threshold: 
                tid = track_ids[r]
                self.track_history[tid] = {"node": nodes_t1[c], "age": 0, "active": True}
                matched_tracks.add(tid)
                matched_nodes.add(c)

        for i, tid in enumerate(track_ids):
            if tid not in matched_tracks:
                self.track_history[tid]["age"] += 1
                if self.track_history[tid]["age"] > self.cfg.max_age:
                    del self.track_history[tid]
        
        for i, node in enumerate(nodes_t1):
            if i not in matched_nodes:
                self.track_history[node["instance"]] = {"node": node, "age": 0, "active": True}

        return r_ind, c_ind, cost_matrix

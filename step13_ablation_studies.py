from nuscenes import NuScenes
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

"""
Step 13: Ablation Studies
Author: Varshitha Dayalan
Research Goal: Quantify the contribution of each component (Motion, Embedding, Uncertainty).

Ablation Configurations:
1. FULL:    Hybrid (Motion + Embedding + Uncertainty)
2. NO_EMB:  Motion Only
3. NO_MOT:  Embedding Only
4. NO_UNC:  Motion + Embedding (No Uncertainty weighting)
"""

# -------------------------------
# CONFIG
# -------------------------------
DATAROOT = 'C:/Users/varsh/nuscenes_project/data/sets/nuscenes'
VERSION = 'v1.0-mini'

# -------------------------------
# Load dataset
# -------------------------------
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

# -------------------------------
# Utility & Extraction
# -------------------------------
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def extract_nodes_with_velocity(curr_sample, prev_sample):
    prev_positions = {}
    if prev_sample:
        for ann_token in prev_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            prev_positions[ann['instance_token']] = ann['translation']
    
    nodes = []
    for ann_token in curr_sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        pos = ann['translation']
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
            "confidence": confidence, "uncertainty": uncertainty
        })
    return nodes

# -------------------------------
# HYBRID MODEL (Same as Step 11/12)
# -------------------------------
class HybridMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16) 
        )
        self.motion_net = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def get_embedding(self, node):
        feat = torch.tensor([
            node["position"][0], node["position"][1],
            node["vx"], node["vy"],
            node["confidence"], node["uncertainty"]
        ], dtype=torch.float32)
        return self.embedding_net(feat)

    def get_motion_score(self, n1, n2):
        dx, dy = n2["position"][0] - n1["position"][0], n2["position"][1] - n1["position"][1]
        dist_norm = distance(n1["position"], n2["position"]) / 20.0
        same_label = 1 if n1["label"] == n2["label"] else 0
        motion_diff = distance([n1["vx"], n1["vy"]], [n2["vx"], n2["vy"]])
        feat = torch.tensor([dx, dy, dist_norm, same_label, n1["confidence"], n2["confidence"], n1["uncertainty"], n2["uncertainty"], n1["vx"], n1["vy"], motion_diff], dtype=torch.float32)
        return torch.sigmoid(self.motion_net(feat))

# -------------------------------
# CORE EVALUATION ENGINE
# -------------------------------
def evaluate_hybrid_config(model, scenes, alpha, beta, gamma):
    total_correct, total_count = 0, 0
    
    for s_info in scenes:
        samples = []
        current = s_info['first_sample_token']
        while current != "":
            s = nusc.get('sample', current)
            samples.append(s)
            current = s['next']
            
        for i in range(1, len(samples)-1):
            nodes_t = extract_nodes_with_velocity(samples[i], samples[i-1])
            nodes_t1 = extract_nodes_with_velocity(samples[i+1], samples[i])
            if not nodes_t or not nodes_t1: continue
            
            cost_matrix = np.zeros((len(nodes_t), len(nodes_t1)))
            for r, n1 in enumerate(nodes_t):
                for c, n2 in enumerate(nodes_t1):
                    with torch.no_grad():
                        # ABLATION COST CALCULATION
                        emb_dist = torch.norm(model.get_embedding(n1) - model.get_embedding(n2)).item() if alpha > 0 else 0
                        motion_cost = (1.0 - model.get_motion_score(n1, n2).item()) if beta > 0 else 0
                        unc_factor = (n1["uncertainty"] + n2["uncertainty"]) / 2.0 if gamma > 0 else 0
                        
                        cost = (alpha * emb_dist) + (beta * motion_cost) + (gamma * unc_factor)
                        if n1["label"] != n2["label"]: cost += 10.0 # Label penalty
                        cost_matrix[r][c] = cost
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if nodes_t[r]["instance"] == nodes_t1[c]["instance"]:
                    total_correct += 1
            total_count += len(nodes_t)
            
    return (total_correct / total_count) * 100 if total_count > 0 else 0

# -------------------------------
# MAIN ABLATION RUN
# -------------------------------
print("--- Step 13: Starting Ablation Studies ---")
model = HybridMatcher() # Using fresh model or pre-trained if available. 
# For demonstration, we use weights initialized similarly.

train_scenes = nusc.scene[:7]
test_scenes  = nusc.scene[7:]

configs = [
    {"name": "FULL HYBRID (M+E+U)", "alpha": 0.4, "beta": 0.4, "gamma": 0.2},
    {"name": "MOTION ONLY (M)",     "alpha": 0.0, "beta": 1.0, "gamma": 0.0},
    {"name": "EMBEDDING ONLY (E)",  "alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    {"name": "NO UNCERTAINTY (M+E)","alpha": 0.5, "beta": 0.5, "gamma": 0.0},
]

ablation_results = []

for cfg in configs:
    print(f"\nEvaluating: {cfg['name']}...")
    train_acc = evaluate_hybrid_config(model, train_scenes, cfg['alpha'], cfg['beta'], cfg['gamma'])
    test_acc  = evaluate_hybrid_config(model, test_scenes,  cfg['alpha'], cfg['beta'], cfg['gamma'])
    
    ablation_results.append({
        "Method": cfg['name'],
        "Train Acc": train_acc,
        "Test Acc": test_acc
    })

# -------------------------------
# RESULTS TABLE
# -------------------------------
print("\n" + "="*60)
print(f"{'Ablation Study: Component Impact':^60}")
print("="*60)
print(f"{'Configuration':<25} | {'Train Acc':<12} | {'Test Acc':<12}")
print("-" * 60)

for res in ablation_results:
    print(f"{res['Method']:<25} | {res['Train Acc']:>10.2f}% | {res['Test Acc']:>10.2f}%")

print("="*60)

# Scientific Interpretation
print("\n--- Research Interpretation ---")
full = ablation_results[0]['Test Acc']
no_u = ablation_results[3]['Test Acc']
u_impact = full - no_u

print(f"1. Uncertainty Impact: The addition of uncertainty weighting changed accuracy by {u_impact:+.2f}%.")
print("2. Generalization Check: If 'Embedding Only' has a smaller gap than 'Motion Only', it proves embeddings are better for cross-scene work.")
print("3. Hybrid Value: If 'Full Hybrid' > others, our architecture is scientifically justified.")
print("\n--- NEXT: ROBUSTNESS ANALYSIS ---")

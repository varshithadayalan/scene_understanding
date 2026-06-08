import os
from nuscenes import NuScenes
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

"""
Step 12: Cross-Scene Generalization Analysis
Author: Varshitha Dayalan
Research Goal: Prove the model generalizes to unseen environments.

Experimental Design:
1. Training: Scenes 0-6 (7 scenes)
2. Testing:  Scenes 7-9 (3 scenes - entirely unseen)
3. Metric: Temporal Identity Accuracy (TIA)
"""

# -------------------------------
# CONFIG
# -------------------------------
DATAROOT = os.environ.get("NUSCENES_DATAROOT", os.path.expanduser("~/data"))
VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-mini")

# Weights for Hybrid Cost (same as Step 11 for consistency)
ALPHA = 0.4  
BETA  = 0.4  
GAMMA = 0.2  

# -------------------------------
# Load dataset
# -------------------------------
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

# -------------------------------
# Utility
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
            vx = pos[0] - prev_pos[0]
            vy = pos[1] - prev_pos[1]
        else:
            vx, vy = 0, 0
        
        visibility = int(ann['visibility_token'])
        confidence = visibility / 4.0
        uncertainty = 1 - confidence
        
        nodes.append({
            "instance": instance,
            "label": ann['category_name'],
            "position": pos,
            "vx": vx,
            "vy": vy,
            "confidence": confidence,
            "uncertainty": uncertainty
        })
    return nodes

# -------------------------------
# HYBRID MODEL
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
        dx = n2["position"][0] - n1["position"][0]
        dy = n2["position"][1] - n1["position"][1]
        dist = distance(n1["position"], n2["position"])
        dist_norm = dist / 20.0
        same_label = 1 if n1["label"] == n2["label"] else 0
        motion_diff = distance([n1["vx"], n1["vy"]], [n2["vx"], n2["vy"]])
        
        feat = torch.tensor([
            dx, dy, dist_norm, same_label,
            n1["confidence"], n2["confidence"],
            n1["uncertainty"], n2["uncertainty"],
            n1["vx"], n1["vy"], motion_diff
        ], dtype=torch.float32)
        return torch.sigmoid(self.motion_net(feat))

# -------------------------------
# DATASET SPLITTING
# -------------------------------
train_scenes = nusc.scene[:7]
test_scenes  = nusc.scene[7:]

def get_pairs_from_scene(scene):
    samples = []
    current = scene['first_sample_token']
    while current != "":
        s = nusc.get('sample', current)
        samples.append(s)
        current = s['next']
    
    pairs = []
    for i in range(1, len(samples)-1):
        nodes_t = extract_nodes_with_velocity(samples[i], samples[i-1])
        nodes_t1 = extract_nodes_with_velocity(samples[i+1], samples[i])
        for n1 in nodes_t:
            for n2 in nodes_t1:
                label = 1 if n1["instance"] == n2["instance"] else 0
                pairs.append((n1, n2, label))
    return pairs

# -------------------------------
# TRAINING
# -------------------------------
print("--- PHASE 2: TRAINING ON 7 SCENES ---")
model = HybridMatcher()
criterion_bce = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Collect all training data
train_data = []
for s_info in train_scenes:
    train_data += get_pairs_from_scene(s_info)

print(f"Total training pairs: {len(train_data)}")

# Simple training loop for demonstration
for epoch in range(10):
    # In real research, we'd use a proper DataLoader
    # Here we sample for speed
    batch = random.sample(train_data, 2000)
    
    # Update motion net
    optimizer.zero_grad()
    loss = 0
    for n1, n2, label in batch:
        # Simplified feature building in the loop for this script
        dx = n2["position"][0] - n1["position"][0]
        dy = n2["position"][1] - n1["position"][1]
        dist_norm = distance(n1["position"], n2["position"]) / 20.0
        same_label = 1 if n1["label"] == n2["label"] else 0
        motion_diff = distance([n1["vx"], n1["vy"]], [n2["vx"], n2["vy"]])
        feat = torch.tensor([dx, dy, dist_norm, same_label, n1["confidence"], n2["confidence"], n1["uncertainty"], n2["uncertainty"], n1["vx"], n1["vy"], motion_diff], dtype=torch.float32)
        
        score = model.motion_net(feat)
        loss += criterion_bce(score, torch.tensor([label], dtype=torch.float32))
    
    loss = loss / 2000
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# EVALUATION FUNCTION
# -------------------------------
def evaluate_on_scenes(model, scenes, label_name):
    print(f"\n--- EVALUATING: {label_name} ---")
    results = []
    
    for s_info in scenes:
        samples = []
        current = s_info['first_sample_token']
        while current != "":
            s = nusc.get('sample', current)
            samples.append(s)
            current = s['next']
        
        correct, total = 0, 0
        for i in range(1, len(samples)-1):
            nodes_t = extract_nodes_with_velocity(samples[i], samples[i-1])
            nodes_t1 = extract_nodes_with_velocity(samples[i+1], samples[i])
            
            if not nodes_t or not nodes_t1: continue
            
            cost_matrix = np.zeros((len(nodes_t), len(nodes_t1)))
            for row, n1 in enumerate(nodes_t):
                for col, n2 in enumerate(nodes_t1):
                    with torch.no_grad():
                        emb1, emb2 = model.get_embedding(n1), model.get_embedding(n2)
                        emb_dist = torch.norm(emb1 - emb2).item()
                        motion_cost = 1.0 - model.get_motion_score(n1, n2).item()
                        uncertainty = (n1["uncertainty"] + n2["uncertainty"]) / 2.0
                        cost = (ALPHA * emb_dist) + (BETA * motion_cost) + (GAMMA * uncertainty)
                        if n1["label"] != n2["label"]: cost += 5.0
                        cost_matrix[row][col] = cost
            
            r_ind, c_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(r_ind, c_ind):
                if nodes_t[r]["instance"] == nodes_t1[c]["instance"]:
                    correct += 1
            total += len(nodes_t)
        
        acc = (correct / total) * 100 if total > 0 else 0
        print(f" Scene: {s_info['name']:<12} | Accuracy: {acc:.2f}%")
        results.append(acc)
    
    avg_acc = sum(results)/len(results)
    print(f" AVERAGE {label_name} ACCURACY: {avg_acc:.2f}%")
    return avg_acc

# -------------------------------
# EXECUTE EVALUATION
# -------------------------------
train_acc = evaluate_on_scenes(model, train_scenes, "TRAIN (Seen Scenes)")
test_acc  = evaluate_on_scenes(model, test_scenes, "TEST (Unseen Scenes)")

print("\n" + "="*40)
print(" FINAL CROSS-SCENE REPORT ")
print("="*40)
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy:  {test_acc:.2f}%")
print(f"Generalization Gap: {abs(train_acc - test_acc):.2f}%")
print("="*40)

if abs(train_acc - test_acc) < 5:
    print("STATUS: SUCCESS. High generalization confirmed.")
else:
    print("STATUS: OVERFITTING DETECTED. Adjust weights or complexity.")

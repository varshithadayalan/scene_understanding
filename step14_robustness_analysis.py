from nuscenes import NuScenes
import math
import torch
import torch.nn as nn
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

"""
Step 14: Robustness Analysis
Author: Varshitha Dayalan
Research Goal: Stress-test the hybrid model by introducing artificial sensor noise.

Types of Noise:
1. Positional Jitter: Gaussian noise on (x, y).
2. Random Occlusions: Dropping nodes with a certain probability.
3. Uncertainty Injection: Artificially increasing uncertainty values.
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

def extract_nodes_with_noise(curr_sample, prev_sample, jitter_std=0.0, drop_prob=0.0):
    prev_positions = {}
    if prev_sample:
        for ann_token in prev_sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            prev_positions[ann['instance_token']] = ann['translation']
    
    nodes = []
    for ann_token in curr_sample['anns']:
        # Random Drop (Simulate occlusion/missed detection)
        if random.random() < drop_prob:
            continue
            
        ann = nusc.get('sample_annotation', ann_token)
        pos = list(ann['translation'])
        
        # Add Jitter
        pos[0] += random.gauss(0, jitter_std)
        pos[1] += random.gauss(0, jitter_std)
        
        instance = ann['instance_token']
        if instance in prev_positions:
            prev_pos = prev_positions[instance]
            vx, vy = pos[0] - prev_pos[0], pos[1] - prev_pos[1]
        else:
            vx, vy = 0, 0
        
        visibility = int(ann['visibility_token'])
        confidence = visibility / 4.0
        
        # If we added jitter, we should technically increase uncertainty
        # but for this test, let's keep it as is to see how the model breaks.
        uncertainty = 1 - confidence
        
        nodes.append({
            "instance": instance, "label": ann['category_name'],
            "position": pos, "vx": vx, "vy": vy,
            "confidence": confidence, "uncertainty": uncertainty
        })
    return nodes

# -------------------------------
# HYBRID MODEL (Same as Step 11/12/13)
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
# EVALUATION ENGINE
# -------------------------------
def evaluate_robustness(model, scenes, jitter_std, drop_prob):
    total_correct, total_count = 0, 0
    alpha, beta, gamma = 0.4, 0.4, 0.2 # Full Hybrid weights
    
    for s_info in scenes:
        samples = []
        current = s_info['first_sample_token']
        while current != "":
            s = nusc.get('sample', current)
            samples.append(s)
            current = s['next']
            
        for i in range(1, len(samples)-1):
            nodes_t = extract_nodes_with_noise(samples[i], samples[i-1], jitter_std, drop_prob)
            nodes_t1 = extract_nodes_with_noise(samples[i+1], samples[i], jitter_std, drop_prob)
            if not nodes_t or not nodes_t1: continue
            
            cost_matrix = np.zeros((len(nodes_t), len(nodes_t1)))
            for r, n1 in enumerate(nodes_t):
                for c, n2 in enumerate(nodes_t1):
                    with torch.no_grad():
                        emb_dist = torch.norm(model.get_embedding(n1) - model.get_embedding(n2)).item()
                        motion_cost = (1.0 - model.get_motion_score(n1, n2).item())
                        unc_factor = (n1["uncertainty"] + n2["uncertainty"]) / 2.0
                        
                        cost = (alpha * emb_dist) + (beta * motion_cost) + (gamma * unc_factor)
                        if n1["label"] != n2["label"]: cost += 10.0
                        cost_matrix[r][c] = cost
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if nodes_t[r]["instance"] == nodes_t1[c]["instance"]:
                    total_correct += 1
            # Ground truth count should be based on how many instances exist in both noisy sets
            # For simplicity, we use len(nodes_t) as baseline potential matches
            total_count += len(nodes_t)
            
    return (total_correct / total_count) * 100 if total_count > 0 else 0

# -------------------------------
# MAIN ROBUSTNESS RUN
# -------------------------------
print("--- Step 14: Starting Robustness Analysis ---")
model = HybridMatcher()

test_scenes = nusc.scene[7:] # Evaluate on test scenes

noise_scenarios = [
    {"name": "Clean Baseline", "jitter": 0.0, "drop": 0.0},
    {"name": "Low Jitter (0.2m)", "jitter": 0.2, "drop": 0.0},
    {"name": "High Jitter (1.0m)", "jitter": 1.0, "drop": 0.0},
    {"name": "Low Occlusion (10%)", "jitter": 0.0, "drop": 0.1},
    {"name": "High Occlusion (30%)", "jitter": 0.0, "drop": 0.3},
    {"name": "Extreme (Jitter+Drop)", "jitter": 0.5, "drop": 0.2},
]

results = []

for sc in noise_scenarios:
    print(f"Testing Scenario: {sc['name']}...")
    acc = evaluate_robustness(model, test_scenes, sc['jitter'], sc['drop'])
    results.append({
        "Scenario": sc['name'],
        "Accuracy": acc
    })

# -------------------------------
# RESULTS TABLE
# -------------------------------
print("\n" + "="*50)
print(f"{'Robustness Analysis: Stress-Test Results':^50}")
print("="*50)
print(f"{'Scenario':<25} | {'Accuracy':<15}")
print("-" * 50)

for res in results:
    print(f"{res['Scenario']:<25} | {res['Accuracy']:>12.2f}%")

print("="*50)

# Research Insights
print("\n--- Research Insights ---")
baseline = results[0]['Accuracy']
extreme = results[-1]['Accuracy']
drop = baseline - extreme

print(f"1. Overall Degradation: Performance dropped by {drop:.2f}% under extreme conditions.")
print("2. Jitter vs Occlusion: Check which scenario caused more failure. Usually, High Jitter breaks Motion branch more than Embedding.")
print("3. Resilience: If Accuracy stays > 70% under 'Extreme', the Hybrid model is considered robust for mini-scale.")

print("\n--- PHASE 3 COMPLETE: READY FOR PHASE 4 (NARRATIVE FORMALIZATION) ---")

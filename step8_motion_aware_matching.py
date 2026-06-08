import os
from nuscenes import NuScenes
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from scipy.optimize import linear_sum_assignment

# -------------------------------
# Load dataset
# -------------------------------
nusc = NuScenes(
    version=os.environ.get("NUSCENES_VERSION", "v1.0-mini"),
    dataroot=os.environ.get("NUSCENES_DATAROOT", os.path.expanduser("~/data")),
    verbose=True
)

# -------------------------------
# Utility
# -------------------------------
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# -------------------------------
# Extract nodes with velocity
# -------------------------------
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
        
        # velocity
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
# Feature builder (motion-aware)
# -------------------------------
def build_feature(n1, n2):
    dx = n2["position"][0] - n1["position"][0]
    dy = n2["position"][1] - n1["position"][1]
    
    dist = distance(n1["position"], n2["position"])
    dist_norm = dist / 20.0
    
    same_label = 1 if n1["label"] == n2["label"] else 0
    
    # motion consistency
    motion_diff = distance([n1["vx"], n1["vy"]], [n2["vx"], n2["vy"]])
    
    return [
        dx, dy,
        dist_norm,
        same_label,
        n1["confidence"],
        n2["confidence"],
        n1["uncertainty"],
        n2["uncertainty"],
        n1["vx"], n1["vy"],
        motion_diff
    ]

# -------------------------------
# Model
# -------------------------------
class MatcherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Build dataset (multi-frame)
# -------------------------------
scene = nusc.scene[0]

samples = []
current = scene['first_sample_token']

while current != "":
    sample = nusc.get('sample', current)
    samples.append(sample)
    current = sample['next']

positive_pairs, negative_pairs = [], []

for i in range(1, len(samples)-1):
    
    prev_s = samples[i-1]
    curr_s = samples[i]
    next_s = samples[i+1]
    
    nodes_t = extract_nodes_with_velocity(curr_s, prev_s)
    nodes_t1 = extract_nodes_with_velocity(next_s, curr_s)
    
    for n1 in nodes_t:
        for n2 in nodes_t1:
            
            feat = build_feature(n1, n2)
            
            if n1["instance"] == n2["instance"]:
                positive_pairs.append((feat, 1))
            else:
                negative_pairs.append((feat, 0))

# balance
neg_sample = random.sample(negative_pairs, len(positive_pairs))
dataset = positive_pairs + neg_sample
random.shuffle(dataset)

X = torch.tensor([d[0] for d in dataset], dtype=torch.float32)
y = torch.tensor([d[1] for d in dataset], dtype=torch.float32).view(-1,1)

print("Dataset size:", len(X))

# -------------------------------
# Train
# -------------------------------
model = MatcherNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1)%5==0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Test with Hungarian
# -------------------------------
test_prev = samples[0]
test_curr = samples[1]
test_next = samples[2]

nodes_t = extract_nodes_with_velocity(test_curr, test_prev)
nodes_t1 = extract_nodes_with_velocity(test_next, test_curr)

cost_matrix = np.zeros((len(nodes_t), len(nodes_t1)))

for i,n1 in enumerate(nodes_t):
    for j,n2 in enumerate(nodes_t1):
        feat = torch.tensor(build_feature(n1,n2), dtype=torch.float32)
        score = torch.sigmoid(model(feat)).item()
        cost_matrix[i][j] = -score

row_ind, col_ind = linear_sum_assignment(cost_matrix)

correct = 0

print("\n===== MOTION-AWARE MATCHING =====\n")

for i in range(min(10, len(row_ind))):
    r,c = row_ind[i], col_ind[i]
    
    n1, n2 = nodes_t[r], nodes_t1[c]
    
    is_correct = n1["instance"] == n2["instance"]
    if is_correct: correct+=1
    
    print("Object:", n1["label"])
    print("→ Matched:", n2["label"])
    print("Correct?", is_correct)
    print("-"*40)


print("\nAccuracy:", correct / min(10,len(row_ind)))

full_correct = 0

for r, c in zip(row_ind, col_ind):
    if nodes_t[r]["instance"] == nodes_t1[c]["instance"]:
        full_correct += 1

full_acc = full_correct / len(row_ind)
print("FULL Accuracy:", full_acc)
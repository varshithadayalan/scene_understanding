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
# Feature for embedding
# -------------------------------
def build_node_feature(n):
    return [
        n["position"][0],
        n["position"][1],
        n["vx"],
        n["vy"],
        n["confidence"],
        n["uncertainty"]
    ]

# -------------------------------
# Embedding Network
# -------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # embedding vector
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Contrastive Loss
# -------------------------------
def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = torch.norm(emb1 - emb2, dim=1)
    
    loss = label * dist**2 + (1 - label) * torch.clamp(margin - dist, min=0)**2
    return loss.mean()

# -------------------------------
# Build dataset
# -------------------------------
scene = nusc.scene[0]

samples = []
current = scene['first_sample_token']

while current != "":
    sample = nusc.get('sample', current)
    samples.append(sample)
    current = sample['next']

pairs = []

for i in range(1, len(samples)-1):
    
    prev_s = samples[i-1]
    curr_s = samples[i]
    next_s = samples[i+1]
    
    nodes_t = extract_nodes_with_velocity(curr_s, prev_s)
    nodes_t1 = extract_nodes_with_velocity(next_s, curr_s)
    
    for n1 in nodes_t:
        for n2 in nodes_t1:
            
            f1 = build_node_feature(n1)
            f2 = build_node_feature(n2)
            
            label = 1 if n1["instance"] == n2["instance"] else 0
            
            pairs.append((f1, f2, label))

# balance dataset
pos = [p for p in pairs if p[2] == 1]
neg = [p for p in pairs if p[2] == 0]

neg_sample = random.sample(neg, len(pos))
dataset = pos + neg_sample
random.shuffle(dataset)

print("Dataset size:", len(dataset))

# tensors
X1 = torch.tensor([d[0] for d in dataset], dtype=torch.float32)
X2 = torch.tensor([d[1] for d in dataset], dtype=torch.float32)
y = torch.tensor([d[2] for d in dataset], dtype=torch.float32)

# -------------------------------
# Train model
# -------------------------------
model = EmbeddingNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    
    emb1 = model(X1)
    emb2 = model(X2)
    
    loss = contrastive_loss(emb1, emb2, y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Matching using embeddings + Hungarian
# -------------------------------
test_prev = samples[0]
test_curr = samples[1]
test_next = samples[2]

nodes_t = extract_nodes_with_velocity(test_curr, test_prev)
nodes_t1 = extract_nodes_with_velocity(test_next, test_curr)

cost_matrix = np.zeros((len(nodes_t), len(nodes_t1)))

for i, n1 in enumerate(nodes_t):
    for j, n2 in enumerate(nodes_t1):
        
        f1 = torch.tensor(build_node_feature(n1), dtype=torch.float32)
        f2 = torch.tensor(build_node_feature(n2), dtype=torch.float32)
        
        with torch.no_grad():
            e1 = model(f1)
            e2 = model(f2)
            dist = torch.norm(e1 - e2).item()
        
        cost_matrix[i][j] = dist  # minimize distance

# Hungarian
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Evaluate
correct = 0

print("\n===== EMBEDDING MATCHING =====\n")

for i in range(min(10, len(row_ind))):
    r, c = row_ind[i], col_ind[i]
    
    n1 = nodes_t[r]
    n2 = nodes_t1[c]
    
    is_correct = n1["instance"] == n2["instance"]
    
    if is_correct:
        correct += 1
    
    print("Object:", n1["label"])
    print("→ Matched:", n2["label"])
    print("Correct?", is_correct)
    print("-"*40)

# full accuracy
full_correct = 0

for r, c in zip(row_ind, col_ind):
    if nodes_t[r]["instance"] == nodes_t1[c]["instance"]:
        full_correct += 1

full_acc = full_correct / len(row_ind)

print("\nFULL Accuracy:", full_acc)
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
# Utility functions
# -------------------------------
def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2
    )

def extract_nodes(sample):
    nodes = []
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        visibility = int(ann['visibility_token'])
        confidence = visibility / 4.0
        uncertainty = 1 - confidence
        
        node = {
            "instance": ann['instance_token'],
            "label": ann['category_name'],
            "position": ann['translation'],
            "confidence": confidence,
            "uncertainty": uncertainty
        }
        
        nodes.append(node)
    
    return nodes

# -------------------------------
# Feature builder
# -------------------------------
def build_feature(node1, node2):
    dx = node2["position"][0] - node1["position"][0]
    dy = node2["position"][1] - node1["position"][1]
    
    dist = distance(node1["position"], node2["position"])
    dist_norm = dist / 20.0
    
    same_label = 1 if node1["label"] == node2["label"] else 0
    
    return [
        dx, dy,
        dist_norm,
        same_label,
        node1["confidence"],
        node2["confidence"],
        node1["uncertainty"],
        node2["uncertainty"]
    ]

# -------------------------------
# Neural Network
# -------------------------------
class MatcherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Build multi-frame training data
# -------------------------------
scene = nusc.scene[0]

samples = []
current = scene['first_sample_token']

while current != "":
    sample = nusc.get('sample', current)
    samples.append(sample)
    current = sample['next']

positive_pairs = []
negative_pairs = []

for i in range(len(samples) - 1):
    nodes_t = extract_nodes(samples[i])
    nodes_t1 = extract_nodes(samples[i+1])
    
    for node1 in nodes_t:
        for node2 in nodes_t1:
            
            feat = build_feature(node1, node2)
            
            if node1["instance"] == node2["instance"]:
                positive_pairs.append((feat, 1))
            else:
                negative_pairs.append((feat, 0))

# balance
neg_sample = random.sample(negative_pairs, len(positive_pairs))
dataset = positive_pairs + neg_sample
random.shuffle(dataset)

X = torch.tensor([d[0] for d in dataset], dtype=torch.float32)
y = torch.tensor([d[1] for d in dataset], dtype=torch.float32).view(-1, 1)

print("Training samples:", len(X))

# -------------------------------
# Train model
# -------------------------------
model = MatcherNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    
    outputs = model(X)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# TEST: Learned + Hungarian
# -------------------------------
test_sample = samples[0]
test_next = samples[1]

nodes_t = extract_nodes(test_sample)
nodes_t1 = extract_nodes(test_next)

# -------------------------------
# Build COST MATRIX from model
# -------------------------------
cost_matrix = np.zeros((len(nodes_t), len(nodes_t1)))

for i, node1 in enumerate(nodes_t):
    for j, node2 in enumerate(nodes_t1):
        
        feat = torch.tensor(build_feature(node1, node2), dtype=torch.float32)
        
        with torch.no_grad():
            logit = model(feat)
            score = torch.sigmoid(logit).item()
        
        cost_matrix[i][j] = -score  # maximize score → minimize cost

# -------------------------------
# Hungarian Matching
# -------------------------------
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# -------------------------------
# Evaluate
# -------------------------------
correct = 0

print("\n===== LEARNED + HUNGARIAN MATCHING =====\n")

for i in range(min(10, len(row_ind))):
    
    r = row_ind[i]
    c = col_ind[i]
    
    node1 = nodes_t[r]
    node2 = nodes_t1[c]
    
    score = -cost_matrix[r][c]
    
    is_correct = node1["instance"] == node2["instance"]
    
    if is_correct:
        correct += 1
    
    print("Object:", node1["label"])
    print("→ Matched:", node2["label"])
    print("Score:", round(score, 3))
    print("Correct?", is_correct)
    print("-" * 40)

# accuracy (sample)
accuracy = correct / min(10, len(row_ind))

print("\n===== SAMPLE ACCURACY =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

# full accuracy
full_correct = 0

for r, c in zip(row_ind, col_ind):
    if nodes_t[r]["instance"] == nodes_t1[c]["instance"]:
        full_correct += 1

full_accuracy = full_correct / len(row_ind)

print("\n===== FULL ACCURACY =====")
print("Full Accuracy:", round(full_accuracy * 100, 2), "%")
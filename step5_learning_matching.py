import os
from nuscenes import NuScenes
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random

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
# IMPROVED FEATURE BUILDER
# -------------------------------
def build_feature(node1, node2):
    dx = node2["position"][0] - node1["position"][0]
    dy = node2["position"][1] - node1["position"][1]
    
    dist = distance(node1["position"], node2["position"])
    dist_norm = dist / 20.0  # normalization
    
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
# Neural Network (updated input size)
# -------------------------------
class MatcherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 32),   # changed from 9 → 8
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)    # logits
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Load frames
# -------------------------------
scene = nusc.scene[0]

sample = nusc.get('sample', scene['first_sample_token'])
next_sample = nusc.get('sample', sample['next'])

nodes_t = extract_nodes(sample)
nodes_t1 = extract_nodes(next_sample)

# -------------------------------
# Build BALANCED dataset
# -------------------------------
positive_pairs = []
negative_pairs = []

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

# convert to tensors
X = torch.tensor([d[0] for d in dataset], dtype=torch.float32)
y = torch.tensor([d[1] for d in dataset], dtype=torch.float32).view(-1, 1)

print("Balanced dataset size:", len(X))

# -------------------------------
# Train model
# -------------------------------
model = MatcherNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):   # increased epochs
    optimizer.zero_grad()
    
    outputs = model(X)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------------
# Evaluate matching
# -------------------------------
correct = 0
total = 0

print("\n===== LEARNED MATCHING (IMPROVED) =====\n")

for node1 in nodes_t[:10]:
    
    best_score = -1
    best_match = None
    
    for node2 in nodes_t1:
        feat = torch.tensor(build_feature(node1, node2), dtype=torch.float32)
        
        with torch.no_grad():
            logit = model(feat)
            score = torch.sigmoid(logit).item()
        
        if score > best_score:
            best_score = score
            best_match = node2
    
    is_correct = node1["instance"] == best_match["instance"]
    
    if is_correct:
        correct += 1
    
    total += 1
    
    print("Object:", node1["label"])
    print("→ Predicted:", best_match["label"])
    print("Score:", round(best_score, 3))
    print("Correct?", is_correct)
    print("-" * 40)

accuracy = correct / total

print("\n===== FINAL ACCURACY =====")
print("Accuracy:", round(accuracy * 100, 2), "%")
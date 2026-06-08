import os
from nuscenes import NuScenes
import math
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
            "id": ann_token,
            "instance": ann['instance_token'],  # for evaluation only
            "label": ann['category_name'],
            "position": ann['translation'],
            "confidence": round(confidence, 2),
            "uncertainty": round(uncertainty, 2)
        }
        
        nodes.append(node)
    
    return nodes

# -------------------------------
# IMPROVED COST FUNCTION
# -------------------------------
def compute_cost(node1, node2):
    
    dist = distance(node1["position"], node2["position"])
    
    # 🚨 Spatial gating (reject impossible matches)
    if dist > 20:
        return 1000  # very high cost
    
    # Semantic cost
    label_cost = 0 if node1["label"] == node2["label"] else 5
    
    # Uncertainty cost
    uncertainty_cost = node1["uncertainty"] + node2["uncertainty"]
    
    # Motion / velocity proxy (encourage smooth movement)
    velocity_cost = dist
    
    # Weighted combination (important)
    total_cost = (
        1.0 * dist +
        2.0 * label_cost +
        0.5 * uncertainty_cost +
        1.0 * velocity_cost
    )
    
    return total_cost

# -------------------------------
# Load frames
# -------------------------------
scene = nusc.scene[0]

sample_token = scene['first_sample_token']
sample = nusc.get('sample', sample_token)

next_sample_token = sample['next']
next_sample = nusc.get('sample', next_sample_token)

nodes_t = extract_nodes(sample)
nodes_t1 = extract_nodes(next_sample)

print("\nFrame t objects:", len(nodes_t))
print("Frame t+1 objects:", len(nodes_t1))

# -------------------------------
# Build COST MATRIX
# -------------------------------
cost_matrix = np.zeros((len(nodes_t), len(nodes_t1)))

for i, node1 in enumerate(nodes_t):
    for j, node2 in enumerate(nodes_t1):
        cost_matrix[i][j] = compute_cost(node1, node2)

# -------------------------------
# Hungarian Matching
# -------------------------------
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# -------------------------------
# Evaluate matches
# -------------------------------
correct = 0

print("\n===== IMPROVED MATCHING RESULTS =====\n")

for i in range(min(10, len(row_ind))):
    
    r = row_ind[i]
    c = col_ind[i]
    
    node1 = nodes_t[r]
    node2 = nodes_t1[c]
    
    cost = cost_matrix[r][c]
    
    is_correct = node1["instance"] == node2["instance"]
    
    if is_correct:
        correct += 1
    
    print("Object:", node1["label"])
    print("→ Matched with:", node2["label"])
    print("Distance:", round(distance(node1["position"], node2["position"]), 2))
    print("Cost:", round(cost, 2))
    print("Correct?", is_correct)
    print("-" * 40)

# -------------------------------
# Accuracy
# -------------------------------
total = len(row_ind)
accuracy = correct / min(10, total)

print("\n===== SAMPLE ACCURACY =====")
print("Correct matches:", correct)
print("Accuracy:", round(accuracy * 100, 2), "%")

# -------------------------------
# Full accuracy
# -------------------------------
full_correct = 0

for r, c in zip(row_ind, col_ind):
    if nodes_t[r]["instance"] == nodes_t1[c]["instance"]:
        full_correct += 1

full_accuracy = full_correct / len(row_ind)

print("\n===== FULL MATCHING ACCURACY =====")
print("Total matches:", len(row_ind))
print("Correct matches:", full_correct)
print("Full Accuracy:", round(full_accuracy * 100, 2), "%")
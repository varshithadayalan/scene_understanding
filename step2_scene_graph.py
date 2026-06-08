import os
from nuscenes import NuScenes
import math

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
            "instance": ann['instance_token'],  # only for evaluation
            "label": ann['category_name'],
            "position": ann['translation'],
            "confidence": round(confidence, 2),
            "uncertainty": round(uncertainty, 2)
        }
        
        nodes.append(node)
    
    return nodes

# -------------------------------
# COST FUNCTION (YOUR CORE LOGIC)
# -------------------------------
def compute_cost(node1, node2):
    # spatial distance
    dist = distance(node1["position"], node2["position"])
    
    # semantic mismatch penalty
    label_cost = 0 if node1["label"] == node2["label"] else 5
    
    # uncertainty penalty
    uncertainty_cost = node1["uncertainty"] + node2["uncertainty"]
    
    total_cost = dist + label_cost + uncertainty_cost
    
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
# MATCHING WITHOUT GROUND TRUTH
# -------------------------------
predicted_matches = []
correct_matches = 0

print("\n===== MATCHING RESULTS =====\n")

for node1 in nodes_t[:10]:  # limit to first 10 for readability
    
    
    
    best_match = None
    best_cost = float("inf")
    
    for node2 in nodes_t1:
        cost = compute_cost(node1, node2)
        
        if cost < best_cost:
            best_cost = cost
            best_match = node2
    
    predicted_matches.append((node1, best_match, best_cost))
    
    # check correctness (only for evaluation)
    is_correct = node1["instance"] == best_match["instance"]
    
    if is_correct:
        correct_matches += 1
    
    print("Object:", node1["label"])
    print("→ Predicted match:", best_match["label"])
    print("Cost:", round(best_cost, 2))
    print("Correct match?", is_correct)
    print("-" * 40)

# -------------------------------
# ACCURACY
# -------------------------------
accuracy = correct_matches / len(predicted_matches)

print("\n===== MATCHING ACCURACY =====")
print("Correct matches:", correct_matches)
print("Total tested:", len(predicted_matches))
print("Accuracy:", round(accuracy * 100, 2), "%")
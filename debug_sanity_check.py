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
        
        # simple heuristic confidence
        visibility = ann['visibility_token']  # 1 to 4
        
        confidence = int(visibility) / 4.0
        uncertainty = 1 - confidence
        
        node = {
            "id": ann_token,
            "instance": ann['instance_token'],
            "label": ann['category_name'],
            "position": ann['translation'],
            "size": ann['size'],
            "confidence": round(confidence, 2),
            "uncertainty": round(uncertainty, 2)
        }
        
        nodes.append(node)
    
    return nodes

# -------------------------------
# Load first frame
# -------------------------------
scene = nusc.scene[0]
sample_token = scene['first_sample_token']
sample = nusc.get('sample', sample_token)

print("\n===== FRAME t =====")

print("Number of objects:", len(sample['anns']))

nodes_t = extract_nodes(sample)

print("Sample node:", nodes_t[0])

# -------------------------------
# Build edges (scene graph)
# -------------------------------
edges = []
THRESHOLD = 10  # meters

for i in range(len(nodes_t)):
    for j in range(i + 1, len(nodes_t)):
        
        node1 = nodes_t[i]
        node2 = nodes_t[j]
        
        dist = distance(node1["position"], node2["position"])
        
        relation = "near" if dist < THRESHOLD else "far"
        
        direction = "left_of" if node1["position"][0] < node2["position"][0] else "right_of"
        
        edge = {
            "from": node1["id"],
            "to": node2["id"],
            "relation": relation,
            "direction": direction,
            "distance": round(dist, 2)
        }
        
        edges.append(edge)

print("\nTotal edges created:", len(edges))
print("Sample edge:", edges[0])

# -------------------------------
# Load next frame
# -------------------------------
next_sample_token = sample['next']

if next_sample_token == "":
    print("No next frame available.")
    exit()

next_sample = nusc.get('sample', next_sample_token)

print("\n===== FRAME t+1 =====")
print("Number of objects:", len(next_sample['anns']))

nodes_t1 = extract_nodes(next_sample)

# -------------------------------
# Match objects across frames
# -------------------------------
matches = []

for node1 in nodes_t:
    for node2 in nodes_t1:
        
        if node1["instance"] == node2["instance"]:
            matches.append((node1, node2))

print("\n===== UNCERTAINTY SAMPLE =====")
for node in nodes_t[:5]:
    print(node["label"], "| confidence:", node["confidence"], "| uncertainty:", node["uncertainty"])

print("\nNumber of matched objects:", len(matches))

# -------------------------------
# Movement analysis
# -------------------------------
print("\n===== MOVEMENT ANALYSIS =====")

for i, (node1, node2) in enumerate(matches[:5]):  # show first 5
    
    dist = distance(node1["position"], node2["position"])
    
    print(f"Object {i+1}:")
    print("Label:", node1["label"])
    print("Moved:", round(dist, 2), "meters\n")
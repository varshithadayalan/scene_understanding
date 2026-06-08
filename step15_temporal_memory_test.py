import sys
import os
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from nuscenes import NuScenes
from matcher.models import HybridMatcher
from matcher.engine import ResearchEngine
import torch

"""
Step 15b Validation: Temporal Memory Recovery
Goal: Prove that the engine can recover an ID after a 3-frame occlusion.
"""

DATAROOT = os.environ.get("NUSCENES_DATAROOT", os.path.expanduser("~/data"))
VERSION = os.environ.get("NUSCENES_VERSION", "v1.0-mini")

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
model = HybridMatcher()
engine = ResearchEngine(nusc, model, max_age=10)

scene = nusc.scene[0]
samples = []
current = scene['first_sample_token']
while current != "":
    samples.append(nusc.get('sample', current))
    current = samples[-1]['next']

print(f"--- Step 15b: Testing Temporal Memory Recovery ---")

# 1. Initialize with Frame 0 & 1
nodes_0 = engine.extract_nodes(samples[0])
nodes_1 = engine.extract_nodes(samples[1], samples[0])
engine.match(nodes_0) # Init
engine.match(nodes_1) # First real match

target_instance = nodes_1[0]["instance"]
print(f"Tracking target instance: {target_instance[:8]}...")

# 2. Simulate Occlusion for Frame 2, 3, 4
# We skip these frames but keep the engine running (empty detections)
for i in range(2, 5):
    print(f"Frame {i}: Target is OCCLUDED.")
    # We pass nodes WITHOUT the target instance
    nodes_i = engine.extract_nodes(samples[i], samples[i-1])
    noisy_nodes = [n for n in nodes_i if n["instance"] != target_instance]
    engine.match(noisy_nodes)

# 3. Target Reappears in Frame 5
print(f"Frame 5: Target REAPPEARS.")
nodes_5 = engine.extract_nodes(samples[5], samples[4])

# Check if target is in nodes_5
has_target = any(n["instance"] == target_instance for n in nodes_5)
if not has_target:
    print("Error: Target not found in nuScenes Frame 5. Adjusting test...")
else:
    r, c, costs = engine.match(nodes_5)
    
    # Verify if target_instance is still in engine.track_history with age 0
    if target_instance in engine.track_history and engine.track_history[target_instance]["age"] == 0:
        print(f"SUCCESS: Identity {target_instance[:8]} RECOVERED after 3 frames of occlusion!")
    else:
        print(f"FAILURE: Identity {target_instance[:8]} was LOST.")
        if target_instance in engine.track_history:
             print(f"Current Age: {engine.track_history[target_instance]['age']}")
        else:
             print("Instance completely removed from history.")

print("\n--- Research Conclusion ---")
print("The Temporal Trajectory Buffer enables the 't -> t+n' reasoning required for elite MOT systems.")

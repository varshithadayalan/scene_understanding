import sys
import os
import numpy as np
import torch
from nuscenes import NuScenes

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from matcher.models import HybridMatcher
from matcher.engine import ResearchEngine
from matcher.config import ResearchConfig

"""
Step 16: Research Evaluation Suite
Goal: Quantify MOT metrics (IDRR, IDSW) using the ResearchConfig.
"""

cfg = ResearchConfig()
nusc = NuScenes(version=cfg.version, dataroot=cfg.dataroot, verbose=False)
model = HybridMatcher()
engine = ResearchEngine(nusc, model, cfg)

test_scenes = nusc.scene[7:] # Research on test set

metrics = {
    "total_occlusions": 0,
    "recovered_ids": 0,
    "id_switches": 0,
    "total_matches": 0
}

print(f"--- Step 16: Running Metrics Evaluation Suite ---")

for scene_info in test_scenes:
    samples = []
    current = scene_info['first_sample_token']
    while current != "":
        s = nusc.get('sample', current)
        samples.append(s)
        current = s['next']
    
    # Reset engine for each scene
    engine.track_history = {}
    
    # Ground truth mapping: track_id -> instance_token
    # In nuScenes, instance_token is the ground truth.
    
    for i in range(len(samples)):
        nodes = engine.extract_nodes(samples[i], samples[i-1] if i > 0 else None)
        
        # We need to track which instances were 'lost' and then 'recovered'
        # Before matching, check which previously active tracks are missing in 'nodes'
        previous_active = {tid for tid, t in engine.track_history.items() if t["age"] == 0}
        current_instances = {n["instance"] for n in nodes}
        
        lost_this_frame = previous_active - current_instances
        metrics["total_occlusions"] += len(lost_this_frame)
        
        # Run matching
        r_ind, c_ind, costs = engine.match(nodes)
        
        # After matching, check if any 'lost' tracks (age > 0) were restored to age 0
        for tid, track in engine.track_history.items():
            if track["age"] == 0:
                # If it was lost before (we'd need to track state transition)
                # For simplicity, we check if it matches its own instance token
                # In our engine, track_id IS instance_token
                metrics["total_matches"] += 1
                
                # Check for ID switches: If the engine assigned a node with instance B 
                # to a track history for instance A.
                # Note: Our current engine uses instance_token as the key, so an ID switch 
                # manifests as a node with a different instance_token being matched to that key.
                # However, our engine currently initializes NEW tracks with their own instance_token.
                # So we check if node["instance"] == tid.
                if track["node"]["instance"] != tid:
                    metrics["id_switches"] += 1
                
        # To calculate Recovery Rate, we need to know if an instance that was lost 
        # (in engine.track_history with age > 0) is now back to age 0.
        for tid, track in engine.track_history.items():
             # This is a heuristic for "Recovery"
             if track["age"] == 0 and tid in current_instances and tid in previous_active:
                 # Standard match
                 pass
             elif track["age"] == 0 and tid in current_instances and tid not in previous_active:
                 # This was a RECOVERY (it was missing in a previous frame but is now matched)
                 metrics["recovered_ids"] += 1

print("\n" + "="*40)
print(f"{'UA-HTIM Metrics Results':^40}")
print("="*40)

idrr = (metrics["recovered_ids"] / metrics["total_occlusions"] * 100) if metrics["total_occlusions"] > 0 else 0
idsw = metrics["id_switches"]

print(f"Total Occlusions Detected : {metrics['total_occlusions']}")
print(f"Identities Recovered      : {metrics['recovered_ids']}")
print(f"ID Recovery Rate (IDRR)   : {idrr:.2f}%")
print(f"Identity Switches (IDSW)  : {idsw}")
print("="*40)

print("\n--- Research Inference ---")
if idrr > 50:
    print("The Temporal Trajectory Buffer effectively captures latent object continuity.")
else:
    print("IDRR suggests further tuning of the alpha/beta weights for better Re-ID.")

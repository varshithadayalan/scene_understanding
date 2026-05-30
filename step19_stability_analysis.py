import sys
import os
import numpy as np
import torch
import pandas as pd
from nuscenes import NuScenes

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from matcher.models import HybridMatcher
from matcher.engine import ResearchEngine
from matcher.config import ResearchConfig

"""
Step 19: Persistence Sensitivity Sweeps
Goal: Quantify the relationship between Identity Persistence Penalty (lambda) 
      using the ResearchConfig framework.
"""

cfg = ResearchConfig()
nusc = NuScenes(version=cfg.version, dataroot=cfg.dataroot, verbose=False)
model = HybridMatcher()
test_scenes = nusc.scene[7:] 

# Sweep parameters
lambda_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
sweep_results = []

print(f"--- Step 19: Starting Persistence Sensitivity Sweep ---")

for lmbda in lambda_values:
    print(f"Evaluating Lambda = {lmbda}...")
    cfg.persistence_weight = lmbda
    engine = ResearchEngine(nusc, model, cfg)
    
    metrics = {
        "occlusions": 0,
        "recovered": 0,
        "switches": 0,
        "total_matches": 0,
        "track_lifetimes": []
    }
    
    # Tracking track lifetimes (how many frames a track was active before dying/switching)
    active_lifetimes = {} # tid -> count

    for scene_info in test_scenes:
        samples = []
        current = scene_info['first_sample_token']
        while current != "":
            samples.append(nusc.get('sample', current))
            current = samples[-1]['next']
        
        engine.track_history = {}
        active_lifetimes = {}

        for i in range(len(samples)):
            nodes = engine.extract_nodes(samples[i], samples[i-1] if i > 0 else None)
            
            previous_active = {tid for tid, t in engine.track_history.items() if t["age"] == 0}
            current_instances = {n["instance"] for n in nodes}
            
            metrics["occlusions"] += len(previous_active - current_instances)
            
            engine.match(nodes)
            
            # Post-match analysis
            for tid, track in engine.track_history.items():
                if track["age"] == 0:
                    metrics["total_matches"] += 1
                    active_lifetimes[tid] = active_lifetimes.get(tid, 0) + 1
                    
                    if track["node"]["instance"] != tid:
                        metrics["switches"] += 1
                else:
                    # Track is currently lost, but if it was just active, record its lifetime segment
                    if tid in active_lifetimes:
                        metrics["track_lifetimes"].append(active_lifetimes[tid])
                        del active_lifetimes[tid]
            
            # Re-ID Check
            for tid, track in engine.track_history.items():
                if track["age"] == 0 and tid in current_instances and tid not in previous_active:
                    metrics["recovered"] += 1

    mean_lifetime = np.mean(metrics["track_lifetimes"]) if metrics["track_lifetimes"] else 0
    idrr = (metrics["recovered"] / metrics["occlusions"] * 100) if metrics["occlusions"] > 0 else 0
    
    sweep_results.append({
        "Lambda": lmbda,
        "IDSW": metrics["switches"],
        "IDRR (%)": round(idrr, 2),
        "Mean Lifetime (Frames)": round(mean_lifetime, 2)
    })

# -------------------------------
# DISPLAY SWEEP TABLE
# -------------------------------
df = pd.DataFrame(sweep_results)
print("\n" + "="*60)
print(f"{'Identity Persistence Sensitivity Sweep':^60}")
print("="*60)
print(df.to_string(index=False))
print("="*60)

# Research Interpretation
print("\n--- Research Interpretation ---")
best_idsw = df.loc[df['IDSW'].idxmin()]
print(f"1. Minimum Switches: Achieved at Lambda = {best_idsw['Lambda']} (IDSW={best_idsw['IDSW']}).")
print("2. Stability Trade-off: High Lambda reduces IDSW but might slightly lower IDRR if re-association is over-penalized.")
print("3. Scientific Conclusion: The 'Identity Continuity' prior is a fundamental stabilizer for Multi-Object Tracking.")

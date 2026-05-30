import sys
import os
import json
import numpy as np
import pandas as pd
from nuscenes import NuScenes

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from matcher.models import HybridMatcher
from matcher.engine import ResearchEngine
from matcher.config import ResearchConfig

"""
Step 23c: Full-Mini Benchmark Suite (Optimized)
Goal: Execute a lambda sweep across multiple scenes to validate statistical trends.
"""

def run_on_scenes(lmbda, scene_list, nusc):
    cfg = ResearchConfig()
    cfg.persistence_weight = lmbda
    model = HybridMatcher()
    engine = ResearchEngine(nusc, model, cfg)
    
    results = []
    for scene_info in scene_list:
        samples = []
        current = scene_info['first_sample_token']
        while current != "":
            samples.append(nusc.get('sample', current))
            current = samples[-1]['next']
        
        engine.track_history = {}
        scene_metrics = {
            "scene_name": scene_info['name'],
            "occlusions": 0, "recovered": 0, "switches": 0, "total_matches": 0, "lambda": lmbda
        }
        
        for i in range(len(samples)):
            nodes = engine.extract_nodes(samples[i], samples[i-1] if i > 0 else None)
            previous_active = {tid for tid, t in engine.track_history.items() if t["age"] == 0}
            current_instances = {n["instance"] for n in nodes}
            scene_metrics["occlusions"] += len(previous_active - current_instances)
            
            engine.match(nodes)
            
            for tid, track in engine.track_history.items():
                if track["age"] == 0:
                    scene_metrics["total_matches"] += 1
                    if track["node"]["instance"] != tid:
                        scene_metrics["switches"] += 1
                
            for tid, track in engine.track_history.items():
                if track["age"] == 0 and tid in current_instances and tid not in previous_active:
                    scene_metrics["recovered"] += 1
        
        scene_metrics["idrr"] = (scene_metrics["recovered"] / scene_metrics["occlusions"] * 100) if scene_metrics["occlusions"] > 0 else 0
        results.append(scene_metrics)
    return results

# -------------------------------
# MAIN SWEEP EXECUTION
# -------------------------------
print("--- Step 23c: Starting Multi-Scene Statistical Validation (Optimized) ---")

cfg = ResearchConfig()
nusc = NuScenes(version=cfg.version, dataroot=cfg.dataroot, verbose=False)
target_scenes = nusc.scene[:5] # Validate on first 5 scenes
lambda_values = [0.0, 2.0]
full_results = []

for lmbda in lambda_values:
    print(f"  Evaluating Lambda = {lmbda} across {len(target_scenes)} scenes...")
    res = run_on_scenes(lmbda, target_scenes, nusc)
    full_results.extend(res)

# -------------------------------
# SAVE & ANALYZE
# -------------------------------
df = pd.DataFrame(full_results)
os.makedirs("experiments", exist_ok=True)
df.to_csv("experiments/full_mini_lambda_sweep.csv", index=False)

print("\n" + "="*50)
print(f"{'Multi-Scene Benchmark Results (Averages)':^50}")
print("="*50)

summary = df.groupby("lambda").agg({
    "switches": "mean",
    "idrr": "mean",
    "occlusions": "sum"
}).reset_index()

print(summary.to_string(index=False))
print("="*50)

print("\nResults saved to: experiments/full_mini_lambda_sweep.csv")
print("--- VALIDATION COMPLETE ---")

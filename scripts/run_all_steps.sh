#!/bin/bash
# Smoke-run the validated core pipeline on the configured nuScenes dataset.
#
# Only the steps verified to run end-to-end on a fresh checkout are included
# here (data exploration -> motion -> embedding -> hybrid). The remaining
# step{12..23} scripts are research/benchmark milestones with additional
# dependencies and outputs; run them individually as needed.
#
# Dataset: NUSCENES_DATAROOT (default ~/data), NUSCENES_VERSION (default v1.0-mini).
set -e
cd "$(dirname "$0")/.."

export MPLBACKEND="${MPLBACKEND:-Agg}"
echo "Dataroot: ${NUSCENES_DATAROOT:-$HOME/data}  Version: ${NUSCENES_VERSION:-v1.0-mini}"

echo "== Step 1: data exploration =="
python step1_data_exploration.py

echo "== Step 8: motion-aware matching =="
python step8_motion_aware_matching.py

echo "== Step 9: embedding matching =="
python step9_embedding_matching.py

echo "== Step 11: hybrid matching =="
python step11_hybrid_matching.py --report-dir results/reports --checkpoint-dir results/checkpoints

echo "All core steps completed."

#!/bin/bash
# Runs the core matching models on the configured nuScenes dataset.
#
# Dataset location is read from environment variables (with sensible defaults):
#   NUSCENES_DATAROOT  (default: ~/data)
#   NUSCENES_VERSION   (default: v1.0-mini)
# MPLBACKEND=Agg keeps plotting headless.
set -e

export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "Dataroot: ${NUSCENES_DATAROOT:-$HOME/data}"
echo "Version:  ${NUSCENES_VERSION:-v1.0-mini}"

echo "Running Motion-aware model..."
python step8_motion_aware_matching.py

echo "Running Embedding model..."
python step9_embedding_matching.py

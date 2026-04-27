#!/bin/bash

echo "Running Motion-aware model..."
python src/step8_motion_aware_matching.py

echo "Running Embedding model..."
python src/step9_embedding_matching.py
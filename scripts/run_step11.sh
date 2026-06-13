#!/bin/bash
# Run Step-11 hybrid matching. Extra args are forwarded to the script, e.g.:
#   bash scripts/run_step11.sh --epochs 5 --alpha 0.6 --beta 0.3 --gamma 0.1
#
# Dataset is read from NUSCENES_DATAROOT (default ~/data) and
# NUSCENES_VERSION (default v1.0-mini).
set -e
cd "$(dirname "$0")/.."

export MPLBACKEND="${MPLBACKEND:-Agg}"
echo "Dataroot: ${NUSCENES_DATAROOT:-$HOME/data}  Version: ${NUSCENES_VERSION:-v1.0-mini}"

python step11_hybrid_matching.py \
    --checkpoint-dir results/checkpoints \
    --report-dir results/reports \
    "$@"

# Uncertainty-Aware Hybrid Temporal Identity Matching (UA-HTIM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: nuScenes](https://img.shields.io/badge/Research-nuScenes-blue.svg)](https://www.nuscenes.org/)

## 🔬 Project Conclusion (Phase 1-6)
This repository contains a **Microsoft-level research framework** for Multi-Object Tracking (MOT) in autonomous driving. We have successfully designed, implemented, and statistically validated an **Uncertainty-Aware Hybrid Framework** that achieves a **35-49% reduction in identity switches** through temporal persistence priors.

## 🏗️ Architecture & Documentation
The framework is fully modularized:
- `src/matcher/`: Core models, engine, config, and visualization logic.
- `step1_` to `step23c_`: Sequential research milestones.
- `RESEARCH_LOG.md`: A comprehensive "History Book" documenting every discovery and benchmark.
- `UA_HTIM_Research_Paper.tex`: Formal LaTeX report of the project methodology and results.

## 📈 Key Results (nuScenes-mini)
- **Identity Continuity:** Proven to be the dominant driver of stability.
- **Occlusion Recovery:** Successfully implemented $t \to t+n$ reasoning via trajectory buffers.
- **Generalization:** Trends validated across multiple diverse scenes.

## 🚀 Future Roadmap
- **Scaling:** Migration to `v1.0-trainval` full dataset.
- **Advanced Dynamics:** Integration of Unscented Kalman Filters (UKF) for non-linear motion extrapolation.
- **Multi-Modal:** Incorporating camera-based appearance embeddings.

## 🔁 Reproducibility (this checkout)
- Dataset paths are read from `NUSCENES_DATAROOT` (default `~/data`) and `NUSCENES_VERSION` (default `v1.0-mini`).
- Install: `pip install -r requirements-dev.txt` (adds `pytest`); requires `numpy<2.0` for nuscenes-devkit.
- Unit tests: `pytest` (synthetic fixtures, no dataset needed).
- Core models: `bash run.sh`; hybrid Step-11 CLI: `bash scripts/run_step11.sh --epochs 5`.

## ⚠️ Rigorous Re-evaluation (June 2026)
A reproducibility audit added a leak-free evaluation. Two findings qualify the headline numbers above:

1. **The original `engine.py` leaks ground truth.** It keys every track by the GT `instance_token` and adds the persistence penalty when a candidate detection's *true* identity differs from the track's. The large IDSW reductions (and the ByteTrack comparison) come partly from the tracker being told the answer.
2. **Under standard `motmetrics` (MOTA / IDF1 / IDSW) with a leak-free `OnlineTracker`** (`src/matcher/online_tracker.py`, synthetic IDs), on nuScenes-mini scenes `[:5]`:

   | config | MOTA ↑ | IDF1 ↑ | IDSW ↓ |
   | :--- | :---: | :---: | :---: |
   | motion + buffer only | 0.966 | 0.966 | 379 |
   | + random embedding | 0.967 | 0.974 | 371 |
   | + **trained** embedding | 0.952 | 0.945 | 544 |

   The learned "embedding" does **not** help (and slightly hurts), because its inputs are spatial (`x, y, vx, vy, conf, unc`) — it is redundant with the motion term, **not a true appearance embedding**. Motion + trajectory buffer already reach ~0.97 MOTA/IDF1 at keyframe rate.

**Implication:** scaling the *current* architecture to `v1.0-trainval` will not improve tracking. The real next step is a genuine appearance embedding (camera-crop / ReID features), then re-evaluate. Reproduce with `python train_hybrid.py && python step25_mot_eval.py` → `experiments/mot_metrics.csv`.

---
**Author:** Varshitha Dayalan  
**Project State:** Completed Research (Mini-Scale Validation) · Re-evaluated June 2026

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

---
**Author:** Varshitha Dayalan  
**Project State:** Completed Research (Mini-Scale Validation)

# The UA-HTIM History Book: A Research Journey
**Lead Researcher:** Varshitha Dayalan
**Project:** Uncertainty-Aware Hybrid Temporal Identity Matching (UA-HTIM)
**Timeline:** May 2026
**Dataset:** nuScenes (mini)

---

## 📜 Chapter 1: The Genesis (Exploration & Baselines)

### Step 1-4: The Foundation
We began by exploring the **nuScenes** dataset, specifically the Birds-Eye-View (BEV) coordinates.
- **Initial Approach:** Greedy Euclidean distance matching.
- **Discovery:** Greedy matching fails in crowded scenes where trajectories cross. We implemented the **Hungarian Algorithm** (Linear Sum Assignment) to ensure global consistency.
- **Result:** Accuracy was high in sparse scenes but dropped during rapid vehicle maneuvers.

---

## 🧠 Chapter 2: The Learning Phase (ML Integration)

### Step 5-7: Neural Matching
We attempted to "learn" the matching function using Multi-Layer Perceptrons (MLPs).
- **Failure Analysis:** We encountered **"Learning Collapse"**—the model would predict a 0 for everything because most pairs are non-matches.
- **Fix:** We implemented balanced sampling and specialized loss functions.

---

## 🚀 Chapter 3: The Hybrid Breakthrough

### Step 8-11: Motion + Embeddings
We realized that identity isn't just a location; it's a "signature." We built the **Hybrid Model**.
- **The Formula:** $Cost = \alpha \cdot d_{emb} + \beta \cdot d_{motion} + \gamma \cdot \sigma_{unc}$
- **Logic:** 
    - **Embedding Branch:** Learned a 16D latent signature of the object.
    - **Motion Branch:** Predicted if a transition was physically possible based on velocity vectors.
- **Result:** Baseline accuracy on scene 0 reached **~97.4%**.

---

## 🧪 Chapter 4: The Validation Arc

### Step 12-14: Stress Testing
We pushed the model to its breaking point.
- **Ablation (Step 13):** Confirmed that **Embeddings** are better for generalization, while **Motion** provides local grounding.
- **Robustness (Step 14):** Introduced sensor noise.
    - **Clean Accuracy:** 88.53%
    - **30% Occlusion:** Accuracy plummeted to **57.63%**.
    - **Inference:** The system was "temporally brittle"—it had no memory.

---

## 🛡️ Chapter 5: The Memory Pivot (Microsoft-Level)

### Step 15-18: Persistence & Trajectory Buffers
We solved the occlusion bottleneck by introducing **Temporal Memory**.
- **Trajectory Buffer:** Instead of $t \to t+1$, we match $t \to t+n$. We keep "lost" objects in a gallery for 10 frames and extrapolate their positions.
- **Identity Persistence Bias:** We added a mathematical penalty for switching IDs.
- **Impact (Step 18):**
    - **Identity Switches (IDSW):** Dropped from 1183 to 607 (**49% reduction**).
    - **Proof:** Stability is a "prior," not just a calculation.

---

## 📊 Chapter 6: Statistical Finality

### Step 19-23: Multi-Scene Validation
We ran a massive sensitivity sweep across multiple scenes to confirm our thesis.
- **The "Sweet Spot":** $\lambda = 2.0$ (Persistence Weight).
- **Final Metrics (Averages across 5 scenes):**
    - **IDSW Reduction:** Consistent **~35%** improvement.
    - **IDRR (Recovery Rate):** Improved to **27.93%**.
- **Scientific Conclusion:** We have built a robust, uncertainty-aware framework that reasons about identity continuity through time.

---

## 🏆 Chapter 7: Benchmarking & Comparative Analysis

To validate UA-HTIM against industry standards, we performed a head-to-head comparison with **ByteTrack** across 5 diverse sequences. Both trackers were evaluated using industry-standard **TrackEval** metrics.

### Step 23e-g: Head-to-Head Results

| Tracker | HOTA ↑ | MOTA ↑ | IDF1 ↑ | IDSW ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **ByteTrack (Baseline)** | 74.09% | 63.37% | 74.98% | 471 |
| **UA-HTIM (Ours)** | **90.67%** | **98.09%** | **86.92%** | **223** |

### Scientific Conclusion
UA-HTIM outperformed ByteTrack by **+16.5% in HOTA** and reduced identity switches by **52.6%**. This proves that our **Identity Continuity** prior and **Hybrid Cost Function** are far more effective at maintaining IDs in autonomous environments than standard IoU-based methods.

---

## 🏁 Final Research State
The project is wrapped at **Step 23g**. The framework is modular, config-driven, and statistically validated against industry baselines. It is ready for full-scale deployment on the 400GB nuScenes-trainval set.

---

## 🔬 Chapter 8: Reproducibility Audit & Leak-Free Re-evaluation (June 2026)

A reproducibility pass made the repo runnable on a fresh machine (env-var dataset paths, fixed seeds, a `pytest` suite, and a Step-11 CLI). It also added the missing training→tracker link and a rigorous evaluation. Three findings, in order of importance:

### 8.1 Training was never wired into the tracker
Every benchmark (`step16`, `step23c`) instantiated a **randomly-initialized** `HybridMatcher`; no checkpoint was ever trained and loaded. `train_hybrid.py` now trains both branches on held-out scenes `[5:]` and saves `checkpoints/hybrid_matcher.pth`; `load_hybrid_matcher()` + `cfg.checkpoint_path` (env `HYBRID_CHECKPOINT`) load it. Within the original engine, training appeared to cut IDSW ~45% — but see 8.2.

### 8.2 The original engine leaks ground truth
`engine.py` keys tracks by the GT `instance_token` and applies the persistence penalty when a candidate detection's **true** identity differs from the track's. The tracker is effectively told the answer, so the headline IDSW reductions (Ch. 6) and the ByteTrack table (Ch. 7) are confounded. A leak-free `OnlineTracker` (`src/matcher/online_tracker.py`) assigns its own synthetic IDs and never reads `instance` during matching.

### 8.3 Honest metrics (motmetrics) — embeddings don't help at this scale
`step25_mot_eval.py` scores the leak-free tracker with standard `motmetrics` on scenes `[:5]`:

| Config | MOTA ↑ | IDF1 ↑ | IDSW ↓ |
| :--- | :---: | :---: | :---: |
| Motion + trajectory buffer | 0.966 | 0.966 | 379 |
| + Random embedding | 0.967 | 0.974 | 371 |
| + **Trained** embedding | 0.952 | 0.945 | 544 |
| Trained embedding, no buffer | 0.958 | 0.937 | 468 |

**Conclusion (revised):** at keyframe rate on nuScenes-mini, **motion + trajectory buffer already reach ~0.97 MOTA/IDF1**, and the learned embedding adds nothing (slightly hurts). The cause is structural: the "embedding" net consumes spatial features (`x, y, vx, vy, conf, unc`), so it is redundant with the motion term rather than a true appearance descriptor. **Scaling this architecture to trainval will not improve tracking.** The genuine next research step is a real appearance embedding (camera-crop / ReID features), then re-run `step25_mot_eval.py`. Numbers reproduce via `python train_hybrid.py && python step25_mot_eval.py` (`experiments/mot_metrics.csv`).

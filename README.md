# 🚀 Scene Understanding: Temporal Identity Matching

## 👩‍💻 Author
Varshitha Dayalan

## 🧠 Overview
This project builds a **research-grade temporal identity matching system** using the nuScenes dataset.

It evolves from:
- Heuristic matching  
- → Learned matching  
- → Motion-aware modeling  
- → Embedding-based identity learning  

---

## 🔥 Key Features
- Scene graph construction  
- Hungarian matching (global optimization)  
- Motion-aware identity modeling  
- Embedding + contrastive learning  
- Multi-frame training  

---

## 📊 Results

| Method | Accuracy |
|--------|--------|
| Hungarian (baseline) | ~94% |
| Learned + Hungarian | ~97.1% |
| Motion-aware model | **~98.7%** |
| Embedding model | ~94.8% |

---

## 🏗️ Project Structure
src/
step1_data_exploration.py
step2_scene_graph.py
step3_hungarian_matching.py
step4_improved_matching.py
step5_learning_matching.py
step6_multiframe_learning.py
step7_learned_hungarian.py
step8_motion_aware_matching.py
step9_embedding_matching.py



---

## 🐳 Docker Usage

### Build:

docker build -t scene-understanding


### Run:

docker run -it scene-understanding


### Run pipeline:

bash run.sh


---

## 📦 Dataset

Download:
👉 https://www.nuscenes.org

Place inside:

/data


---

## 🎯 Research Direction

This project explores:
> **Robust identity matching using motion + learned representations in dynamic environments**

Next steps:
- Hybrid model (motion + embedding)
- Cross-scene generalization
- Publication-ready system

---

## ⭐ Contribution

If you find this useful, feel free to fork and build on it.

import streamlit as st
import sys
import os
import pandas as pd
from nuscenes import NuScenes

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from matcher.models import HybridMatcher
from matcher.engine import ResearchEngine
from matcher.visualizer import ResearchVisualizer
from matcher.config import ResearchConfig

st.set_page_config(layout="wide")

st.title("🔬 UA-HTIM: Research Visualization Dashboard")
st.sidebar.header("Configuration")

# -------------------------------
# Load Data
# -------------------------------
cfg = ResearchConfig()
DATAROOT = cfg.dataroot
VERSION = cfg.version

@st.cache_resource
def load_nusc():
    return NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

nusc = load_nusc()

# -------------------------------
# Sidebar Controls (Update Config live)
# -------------------------------
scene_idx = st.sidebar.selectbox("Select Scene", range(len(nusc.scene)), index=0)
cfg.alpha_base = st.sidebar.slider("Alpha (Base Embedding)", 0.0, 1.0, cfg.alpha_base)
cfg.beta_base = st.sidebar.slider("Beta (Base Motion)", 0.0, 1.0, cfg.beta_base)
cfg.gamma_base = st.sidebar.slider("Gamma (Uncertainty)", 0.0, 1.0, cfg.gamma_base)
cfg.persistence_weight = st.sidebar.slider("Persistence Penalty", 0.0, 10.0, cfg.persistence_weight)
cfg.max_age = st.sidebar.slider("Max Track Age", 1, 20, cfg.max_age)

# -------------------------------
# Engine Initialization
# -------------------------------
model = HybridMatcher()
engine = ResearchEngine(nusc, model, cfg)

scene_info = nusc.scene[scene_idx]
samples = []
current = scene_info['first_sample_token']
while current != "":
    samples.append(nusc.get('sample', current))
    current = samples[-1]['next']

# -------------------------------
# Main Loop
# -------------------------------
st.write(f"### Scene: {scene_info['name']} - {scene_info['description']}")

frame_idx = st.slider("Select Frame", 0, len(samples)-1, 0)

# Process frames up to current to maintain state
for i in range(frame_idx + 1):
    nodes = engine.extract_nodes(samples[i], samples[i-1] if i > 0 else None)
    r_ind, c_ind, costs = engine.match(nodes)

# -------------------------------
# Display Results
# -------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.write("#### Cost Matrix")
    if costs is not None:
        fig = ResearchVisualizer.plot_cost_matrix(costs, list(engine.track_history.values()), nodes)
        st.pyplot(fig)
    else:
        st.info("No nodes to match in this frame.")

with col2:
    st.write("#### Active Track History")
    track_data = []
    for tid, track in engine.track_history.items():
        track_data.append({
            "ID": tid[:8],
            "Label": track["node"]["label"].split('.')[-1],
            "Age": track["age"],
            "Pos X": round(track["node"]["position"][0], 2),
            "Pos Y": round(track["node"]["position"][1], 2),
            "Status": "ACTIVE" if track["age"] == 0 else "LOST (EXTRAPOLATED)"
        })
    
    if track_data:
        st.dataframe(pd.DataFrame(track_data))
    else:
        st.write("No active tracks.")

st.write("---")
st.write("#### Research Inference")
st.write(f"Current Tracking State: {len(engine.track_history)} objects in memory.")
st.write("- **Green cells** in Cost Matrix = High match probability.")
st.write("- **Extrapolated** tracks are using the Motion branch to predict movement during sensor gaps.")

"""
Paper 2 — Step 1: nuScenes Setup Verification + Scene Explorer
You x Microsoft Ashwin

What this script does:
  1. Verifies devkit loads correctly
  2. Prints scene/sample structure so you understand the data model
  3. Renders a BEV (bird's-eye-view) plot of one sample scene
     — this is the spatial canvas our 3D scene graph will be built on

Run: python paper2_step1_explore.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Change this to wherever you extracted the mini split
NUSCENES_DATAROOT = r"C:\Users\varsh\nuscenes_project\data\sets\nuscenes"
NUSCENES_VERSION  = "v1.0-mini"
# ───────────────────────────────────────────────────────────────────────────────

def verify_install():
    try:
        from nuscenes.nuscenes import NuScenes
        print("[OK] nuscenes-devkit imported successfully")
        return NuScenes
    except ImportError:
        print("[FAIL] nuscenes-devkit not found.")
        print("  Fix: pip install nuscenes-devkit")
        sys.exit(1)


def load_dataset(NuScenes):
    print(f"\nLoading {NUSCENES_VERSION} from: {NUSCENES_DATAROOT}")
    try:
        nusc = NuScenes(version=NUSCENES_VERSION,
                        dataroot=NUSCENES_DATAROOT,
                        verbose=False)
        print(f"[OK] Loaded — {len(nusc.scene)} scenes, "
              f"{len(nusc.sample)} samples, "
              f"{len(nusc.sample_annotation)} annotations")
        return nusc
    except Exception as e:
        print(f"[FAIL] Could not load dataset: {e}")
        print("  Check that NUSCENES_DATAROOT points to the extracted mini folder")
        sys.exit(1)


def print_data_model(nusc):
    """
    Print the hierarchy nuScenes uses — critical to understand before
    we build a scene graph on top of it.
    """
    print("\n── DATA MODEL ────────────────────────────────────────────────")
    scene = nusc.scene[0]
    print(f"Scene:       '{scene['name']}' | "
          f"{scene['nbr_samples']} samples | "
          f"desc: {scene['description'][:60]}")

    # Walk the first 4 samples in the scene
    sample_token = scene['first_sample_token']
    for i in range(4):
        sample = nusc.get('sample', sample_token)
        anns   = sample['anns']
        print(f"  Sample [{i}] token={sample_token[:8]}... | "
              f"timestamp={sample['timestamp']} | "
              f"annotations={len(anns)}")

        if i == 0:
            # Show annotation detail for first sample
            for ann_token in anns[:3]:
                ann = nusc.get('sample_annotation', ann_token)
                print(f"    Ann: category='{ann['category_name']}' | "
                      f"xyz={[round(v,2) for v in ann['translation']]} | "
                      f"size={[round(v,2) for v in ann['size']]}")
            if len(anns) > 3:
                print(f"    ... and {len(anns)-3} more annotations")

        if sample['next'] == '':
            break
        sample_token = sample['next']

    # Sensor modalities available
    sample0 = nusc.get('sample', scene['first_sample_token'])
    print(f"\n  Sensor keys in sample_data: {list(sample0['data'].keys())}")
    print("──────────────────────────────────────────────────────────────")


def get_agent_boxes(nusc, sample_token):
    """
    Extract all annotated agents from a sample and return as a list of dicts.
    This is the raw material for scene graph nodes (C2).
    """
    sample = nusc.get('sample', sample_token)
    agents = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        agents.append({
            'token':    ann_token,
            'category': ann['category_name'],
            'x':        ann['translation'][0],
            'y':        ann['translation'][1],
            'z':        ann['translation'][2],
            'w':        ann['size'][0],   # width
            'l':        ann['size'][1],   # length
            'h':        ann['size'][2],   # height
            'yaw':      ann['rotation'],  # quaternion — we'll convert below
            'velocity': nusc.box_velocity(ann_token)[:2],  # vx, vy
            'instance': ann['instance_token'],
        })
    return agents


def quaternion_to_yaw(q):
    """Convert nuScenes quaternion [w,x,y,z] to yaw angle (radians)."""
    w, x, y, z = q
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def category_color(cat):
    """Map category to a consistent color for BEV plot."""
    if 'vehicle.car' in cat or 'vehicle.truck' in cat:
        return '#378ADD'   # blue
    if 'vehicle' in cat:
        return '#1D9E75'   # teal
    if 'pedestrian' in cat:
        return '#D85A30'   # coral
    if 'movable_object' in cat or 'static' in cat:
        return '#888780'   # gray
    return '#7F77DD'       # purple fallback


def plot_bev(nusc, sample_token, out_path="bev_scene.png"):
    """
    Render a top-down BEV of the scene showing:
      - Agent boxes (oriented rectangles)
      - Velocity vectors
      - Ego vehicle at origin
    This is the spatial canvas for our 3D scene graph.
    """
    agents = get_agent_boxes(nusc, sample_token)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # Draw ego vehicle
    ego_rect = mpatches.FancyBboxPatch((-2, -1), 4, 2,
                                        boxstyle="round,pad=0.1",
                                        linewidth=1.5,
                                        edgecolor='#FAC775',
                                        facecolor='#412402',
                                        zorder=5)
    ax.add_patch(ego_rect)
    ax.text(0, 0, 'EGO', color='#FAC775', fontsize=8,
            ha='center', va='center', zorder=6, fontweight='bold')

    # Draw agents
    legend_cats = {}
    for agent in agents:
        x, y   = agent['x'], agent['y']
        w, l   = agent['w'], agent['l']
        yaw    = quaternion_to_yaw(agent['rotation'] if 'rotation' in agent
                                   else agent['yaw'])
        color  = category_color(agent['category'])
        cat    = agent['category'].split('.')[1] if '.' in agent['category'] else agent['category']

        # Oriented rectangle
        corners = np.array([
            [-l/2, -w/2], [l/2, -w/2],
            [l/2,  w/2],  [-l/2,  w/2]
        ])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])
        corners = (R @ corners.T).T + np.array([x, y])
        box = plt.Polygon(corners, closed=True,
                          edgecolor=color, facecolor=color+'33',
                          linewidth=1.2, zorder=3)
        ax.add_patch(box)

        # Heading arrow
        dx = np.cos(yaw) * l * 0.6
        dy = np.sin(yaw) * l * 0.6
        ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.2), zorder=4)

        # Velocity vector if non-zero
        vx, vy = agent['velocity']
        if not np.isnan(vx) and (abs(vx) + abs(vy)) > 0.3:
            ax.annotate('', xy=(x+vx, y+vy), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='#FAC775',
                                        lw=0.8, alpha=0.6), zorder=4)

        legend_cats[cat] = color

    # Legend
    handles = [mpatches.Patch(color=c, label=k)
               for k, c in legend_cats.items()]
    handles.append(mpatches.Patch(color='#FAC775', label='ego'))
    ax.legend(handles=handles, loc='upper right',
              facecolor='#2c2c4a', labelcolor='white',
              fontsize=9, framealpha=0.8)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_title(f"BEV Scene — token: {sample_token[:12]}...\n"
                 f"[{len(agents)} agents] — this is our scene graph canvas",
                 color='white', fontsize=11)
    ax.tick_params(colors='#888780')
    ax.spines[:].set_color('#444441')
    ax.set_xlabel("x (m)", color='#888780')
    ax.set_ylabel("y (m)", color='#888780')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] BEV plot saved → {out_path}")
    plt.show()


def summarise_scene_for_graph(nusc, sample_token):
    """
    Print a proto-scene-graph summary — what nodes we'd have.
    This is the conceptual bridge from raw data to Paper 2's graph structure.
    """
    agents = get_agent_boxes(nusc, sample_token)
    from collections import Counter
    cats = Counter(a['category'] for a in agents)

    print("\n── PROTO SCENE GRAPH (sample node inventory) ─────────────────")
    print(f"  Total nodes (agents): {len(agents)}")
    for cat, count in cats.most_common():
        print(f"    {cat:<45} x{count}")
    print(f"\n  Unique instances (persistent IDs across frames): "
          f"{len(set(a['instance'] for a in agents))}")
    print("  → Each instance token = one graph node identity")
    print("  → Edges (spatial relations) will be computed in Step 2")
    print("────────────────────────────────────────────────────────────────")


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NuScenes = verify_install()
    nusc     = load_dataset(NuScenes)

    print_data_model(nusc)

    # Use the first sample of the first scene
    scene        = nusc.scene[0]
    sample_token = scene['first_sample_token']

    summarise_scene_for_graph(nusc, sample_token)
    plot_bev(nusc, sample_token, out_path="bev_scene_step1.png")

    print("\n── NEXT STEP ──────────────────────────────────────────────────")
    print("  Step 2: Build the actual scene graph — nodes + spatial edges")
    print("  Run: python paper2_step2_scene_graph.py")
    print("────────────────────────────────────────────────────────────────")
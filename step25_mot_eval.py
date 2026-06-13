"""
Step 25: Rigorous MOT evaluation with motmetrics (MOTA / IDF1 / IDSW).

Uses the honest OnlineTracker (synthetic IDs, no ground-truth leakage) and scores
its output against nuScenes ground truth with the standard `motmetrics` library.
Compares configurations so we can see what actually drives tracking quality:

  * random  vs  trained embeddings
  * trajectory buffer on vs off
  * embeddings on vs off (motion-only)

Because nuScenes annotations are used as detections, FP/FN are ~0 and the metric
that matters is identity consistency (IDF1, IDSW) — exactly what a learned
association model should improve.

Usage:
    python train_hybrid.py            # produce checkpoints/hybrid_matcher.pth first
    python step25_mot_eval.py
"""
import os
import sys

import motmetrics as mm
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from matcher.config import ResearchConfig                 # noqa: E402
from matcher.engine import ResearchEngine                 # noqa: E402
from matcher.hybrid_core import set_seed                  # noqa: E402
from matcher.models import HybridMatcher, load_hybrid_matcher  # noqa: E402
from matcher.online_tracker import OnlineTracker          # noqa: E402

CHECKPOINT = os.environ.get("HYBRID_CHECKPOINT", "checkpoints/hybrid_matcher.pth")
SEED = 42
GATE_D2 = 2.0 ** 2          # 2 m matching gate for motmetrics (squared)


def collect_samples(nusc, scene_info):
    samples = []
    current = scene_info["first_sample_token"]
    while current != "":
        s = nusc.get("sample", current)
        samples.append(s)
        current = s["next"]
    return samples


def build_model(kind):
    if kind == "trained":
        return load_hybrid_matcher(CHECKPOINT)
    set_seed(SEED)
    return HybridMatcher().eval()


def eval_config(name, *, model, use_embeddings, use_buffer, eval_scenes, nusc, extractor):
    accs = []
    token2int = {}  # nuScenes instance_token (hex str) -> stable int GT id

    def gt_int(token):
        if token not in token2int:
            token2int[token] = len(token2int)
        return token2int[token]

    for scene_info in eval_scenes:
        samples = collect_samples(nusc, scene_info)
        tracker = OnlineTracker(model, use_embeddings=use_embeddings, use_buffer=use_buffer)
        acc = mm.MOTAccumulator(auto_id=True)
        for i in range(len(samples)):
            nodes = extractor.extract_nodes(samples[i], samples[i - 1] if i > 0 else None)
            hyp_ids = tracker.update(nodes)
            gt_ids = [gt_int(n["instance"]) for n in nodes]
            pts = np.array([[n["position"][0], n["position"][1]] for n in nodes], dtype=float)
            if len(pts) == 0:
                acc.update([], [], [])
                continue
            # GT objects and tracker hypotheses are the same detections; the
            # distance matrix is between GT positions and hypothesis positions.
            dist = mm.distances.norm2squared_matrix(pts, pts, max_d2=GATE_D2)
            acc.update(gt_ids, hyp_ids, dist)
        accs.append(acc)
    return accs


def main():
    from nuscenes import NuScenes

    cfg = ResearchConfig()
    nusc = NuScenes(version=cfg.version, dataroot=cfg.dataroot, verbose=False)
    eval_scenes = nusc.scene[:5]
    extractor = ResearchEngine(nusc, HybridMatcher(), cfg)  # only for extract_nodes

    if not os.path.exists(CHECKPOINT):
        print(f"!! checkpoint {CHECKPOINT} not found; run train_hybrid.py first.")
        return

    configs = [
        ("motion_only",        dict(model=None,                       use_embeddings=False, use_buffer=True)),
        ("random_emb",         dict(model=build_model("random"),      use_embeddings=True,  use_buffer=True)),
        ("trained_emb",        dict(model=build_model("trained"),     use_embeddings=True,  use_buffer=True)),
        ("trained_no_buffer",  dict(model=build_model("trained"),     use_embeddings=True,  use_buffer=False)),
    ]

    mh = mm.metrics.create()
    metric_names = ["mota", "idf1", "num_switches", "num_fragmentations",
                    "mostly_tracked", "mostly_lost", "num_objects"]

    rows = []
    print(f"--- Step 25: motmetrics MOT evaluation on {len(eval_scenes)} scenes ---")
    for name, kw in configs:
        accs = eval_config(name, eval_scenes=eval_scenes, nusc=nusc, extractor=extractor, **kw)
        summary = mh.compute_many(
            accs, metrics=metric_names,
            names=[f"{name}-{i}" for i in range(len(accs))], generate_overall=True)
        overall = summary.loc["OVERALL"]
        rows.append({
            "config": name,
            "MOTA": overall["mota"],
            "IDF1": overall["idf1"],
            "IDSW": int(overall["num_switches"]),
            "Frag": int(overall["num_fragmentations"]),
            "MT": int(overall["mostly_tracked"]),
            "ML": int(overall["mostly_lost"]),
            "GT_objs": int(overall["num_objects"]),
        })
        print(f"  [{name}] MOTA={overall['mota']:.3f} IDF1={overall['idf1']:.3f} "
              f"IDSW={int(overall['num_switches'])}")

    df = pd.DataFrame(rows)
    os.makedirs("experiments", exist_ok=True)
    df.to_csv("experiments/mot_metrics.csv", index=False)

    pd.set_option("display.width", 120)
    print("\n" + "=" * 72)
    print(f"{'Rigorous MOT metrics (motmetrics, no GT leak)':^72}")
    print("=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)
    print("Higher MOTA/IDF1 = better; lower IDSW = better.")
    print("Saved: experiments/mot_metrics.csv")


if __name__ == "__main__":
    main()

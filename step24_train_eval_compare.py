"""
Step 24: Does training the neural branches actually help the tracker?

Runs the step23c lambda-sweep metric on the evaluation scenes (scene[:5]) twice:
once with a RANDOMLY-initialized HybridMatcher (the original pipeline's behavior)
and once with a TRAINED checkpoint produced by train_hybrid.py. This isolates the
contribution of the learned embedding/motion branches from the persistence prior.

Usage:
    python train_hybrid.py                       # produce checkpoints/hybrid_matcher.pth
    python step24_train_eval_compare.py          # compare random vs trained
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from matcher.config import ResearchConfig          # noqa: E402
from matcher.engine import ResearchEngine          # noqa: E402
from matcher.hybrid_core import set_seed           # noqa: E402
from matcher.models import HybridMatcher, load_hybrid_matcher  # noqa: E402

CHECKPOINT = os.environ.get("HYBRID_CHECKPOINT", "checkpoints/hybrid_matcher.pth")
SEED = 42
LAMBDAS = [0.0, 2.0]


def collect_samples(nusc, scene_info):
    samples = []
    current = scene_info["first_sample_token"]
    while current != "":
        s = nusc.get("sample", current)
        samples.append(s)
        current = s["next"]
    return samples


def eval_model(model, lmbda, scene_list, nusc):
    """Identical metric logic to step23c.run_on_scenes."""
    cfg = ResearchConfig()
    cfg.persistence_weight = lmbda
    engine = ResearchEngine(nusc, model, cfg)

    rows = []
    for scene_info in scene_list:
        samples = collect_samples(nusc, scene_info)
        engine.track_history = {}
        m = {"scene_name": scene_info["name"], "occlusions": 0,
             "recovered": 0, "switches": 0, "total_matches": 0, "lambda": lmbda}

        for i in range(len(samples)):
            nodes = engine.extract_nodes(samples[i], samples[i - 1] if i > 0 else None)
            previous_active = {tid for tid, t in engine.track_history.items() if t["age"] == 0}
            current_instances = {n["instance"] for n in nodes}
            m["occlusions"] += len(previous_active - current_instances)

            engine.match(nodes)

            for tid, track in engine.track_history.items():
                if track["age"] == 0:
                    m["total_matches"] += 1
                    if track["node"]["instance"] != tid:
                        m["switches"] += 1
            for tid, track in engine.track_history.items():
                if track["age"] == 0 and tid in current_instances and tid not in previous_active:
                    m["recovered"] += 1

        m["idrr"] = (m["recovered"] / m["occlusions"] * 100) if m["occlusions"] > 0 else 0
        rows.append(m)
    return rows


def make_model(kind):
    if kind == "trained":
        return load_hybrid_matcher(CHECKPOINT)
    set_seed(SEED)            # reproducible random init for the baseline
    return HybridMatcher().eval()


def main():
    from nuscenes import NuScenes

    cfg = ResearchConfig()
    nusc = NuScenes(version=cfg.version, dataroot=cfg.dataroot, verbose=False)
    eval_scenes = nusc.scene[:5]
    print(f"--- Step 24: Trained vs Random on {len(eval_scenes)} eval scenes ---")
    if not os.path.exists(CHECKPOINT):
        print(f"!! checkpoint {CHECKPOINT} not found; run train_hybrid.py first.")
        return

    all_rows = []
    for kind in ["random", "trained"]:
        for lmbda in LAMBDAS:
            model = make_model(kind)
            rows = eval_model(model, lmbda, eval_scenes, nusc)
            for r in rows:
                r["model"] = kind
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    os.makedirs("experiments", exist_ok=True)
    df.to_csv("experiments/trained_vs_random.csv", index=False)

    summary = df.groupby(["model", "lambda"]).agg(
        mean_switches=("switches", "mean"),
        mean_idrr=("idrr", "mean"),
        total_occlusions=("occlusions", "sum"),
    ).reset_index()

    print("\n" + "=" * 60)
    print(f"{'Trained vs Random — Benchmark Summary':^60}")
    print("=" * 60)
    print(summary.to_string(index=False))
    print("=" * 60)

    # Headline: effect of training at the paper's lambda=2.0.
    def cell(model, lmbda, col):
        s = summary[(summary.model == model) & (summary["lambda"] == lmbda)]
        return float(s[col].iloc[0]) if len(s) else float("nan")

    rnd = cell("random", 2.0, "mean_switches")
    trn = cell("trained", 2.0, "mean_switches")
    if rnd and not np.isnan(rnd):
        delta = (rnd - trn) / rnd * 100
        print(f"\nAt lambda=2.0: mean IDSW random={rnd:.1f} -> trained={trn:.1f} "
              f"({delta:+.1f}% from training)")
    print("Saved: experiments/trained_vs_random.csv")


if __name__ == "__main__":
    main()

"""
Step 11: Hybrid Motion + Embedding Model
Author: Varshitha Dayalan
Research Goal: Unified uncertainty-aware temporal identity matching.

This model combines:
1. Representation Learning (Embeddings)
2. Physical Grounding (Motion-aware)
3. Observation Quality (Uncertainty)

Cost = alpha * embedding_dist + beta * motion_cost + gamma * uncertainty

Run:
    python step11_hybrid_matching.py            # defaults: ~/data, v1.0-mini
    python -m step11_hybrid_matching --epochs 5 --alpha 0.6 --beta 0.3 --gamma 0.1

The pure, dependency-light cost/matching primitives live in
``src/matcher/hybrid_core.py`` and are shared with the unit tests.
"""
import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Make the `matcher` package importable (matches the convention used by the
# other step scripts in this repo).
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from matcher.hybrid_core import (  # noqa: E402
    EmbeddingMissingError,
    compute_hybrid_cost,
    embedding_distance,
    set_seed,
    solve_matching,
    validate_weights,
)


# -------------------------------
# Utility
# -------------------------------
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def extract_nodes_with_velocity(nusc, curr_sample, prev_sample):
    prev_positions = {}
    if prev_sample:
        for ann_token in prev_sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            prev_positions[ann["instance_token"]] = ann["translation"]

    nodes = []
    for ann_token in curr_sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        pos = ann["translation"]
        instance = ann["instance_token"]

        if instance in prev_positions:
            prev_pos = prev_positions[instance]
            vx = pos[0] - prev_pos[0]
            vy = pos[1] - prev_pos[1]
        else:
            vx, vy = 0, 0

        visibility = int(ann["visibility_token"])
        confidence = visibility / 4.0
        uncertainty = 1 - confidence

        nodes.append({
            "instance": instance,
            "label": ann["category_name"],
            "position": pos,
            "vx": vx,
            "vy": vy,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "token": ann_token,
        })
    return nodes


# -------------------------------
# HYBRID MODEL ARCHITECTURE
# -------------------------------
class HybridMatcher(nn.Module):
    def __init__(self):
        super().__init__()

        # Branch 1: Embedding (Feature extractor)
        self.embedding_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # Branch 2: Motion Classifier
        self.motion_net = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def get_embedding(self, node):
        feat = torch.tensor([
            node["position"][0], node["position"][1],
            node["vx"], node["vy"],
            node["confidence"], node["uncertainty"],
        ], dtype=torch.float32)
        return self.embedding_net(feat)

    def get_motion_score(self, n1, n2):
        # Motion-aware features from Step 8
        dx = n2["position"][0] - n1["position"][0]
        dy = n2["position"][1] - n1["position"][1]
        dist = distance(n1["position"], n2["position"])
        dist_norm = dist / 20.0
        same_label = 1 if n1["label"] == n2["label"] else 0
        motion_diff = distance([n1["vx"], n1["vy"]], [n2["vx"], n2["vy"]])

        feat = torch.tensor([
            dx, dy, dist_norm, same_label,
            n1["confidence"], n2["confidence"],
            n1["uncertainty"], n2["uncertainty"],
            n1["vx"], n1["vy"], motion_diff,
        ], dtype=torch.float32)

        return torch.sigmoid(self.motion_net(feat))


# -------------------------------
# DATASET PREP
# -------------------------------
def collect_scene_samples(nusc, scene_index=0):
    scene = nusc.scene[scene_index]
    samples = []
    current = scene["first_sample_token"]
    while current != "":
        s = nusc.get("sample", current)
        samples.append(s)
        current = s["next"]
    return samples


def build_training_pairs(nusc, samples):
    training_pairs = []
    for i in range(1, len(samples) - 1):
        nodes_t = extract_nodes_with_velocity(nusc, samples[i], samples[i - 1])
        nodes_t1 = extract_nodes_with_velocity(nusc, samples[i + 1], samples[i])
        for n1 in nodes_t:
            for n2 in nodes_t1:
                label = 1 if n1["instance"] == n2["instance"] else 0
                training_pairs.append((n1, n2, label))
    return training_pairs


# -------------------------------
# TRAINING
# -------------------------------
def train_model(model, training_pairs, epochs, lr, max_pairs_per_epoch=20000):
    """Supervised training of both branches. epochs<=0 means inference-only.

    Motion branch: BCE on same-instance probability.
    Embedding branch: contrastive loss (pull same-instance together, push apart).
    """
    if epochs <= 0:
        return {"trained": False, "epochs": 0, "loss_history": []}

    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    margin = 2.0
    history = []

    for ep in range(epochs):
        random.shuffle(training_pairs)
        batch = training_pairs[:max_pairs_per_epoch]
        if len(training_pairs) > max_pairs_per_epoch:
            print(f"[train] epoch {ep + 1}: using {len(batch)}/{len(training_pairs)} "
                  f"pairs this epoch (capped for speed, not silently skipped)")

        total = 0.0
        for n1, n2, label in batch:
            score = model.get_motion_score(n1, n2).reshape(())
            target = torch.tensor(float(label))
            motion_loss = bce(score, target)

            emb1 = model.get_embedding(n1)
            emb2 = model.get_embedding(n2)
            d = torch.norm(emb1 - emb2)
            if label == 1:
                emb_loss = d ** 2
            else:
                emb_loss = torch.clamp(margin - d, min=0.0) ** 2

            loss = motion_loss + 0.1 * emb_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / max(len(batch), 1)
        history.append(avg)
        print(f"[train] epoch {ep + 1}/{epochs} avg_loss={avg:.4f}")

    return {"trained": True, "epochs": epochs, "lr": lr, "loss_history": history}


# -------------------------------
# HYBRID MATCHING INFERENCE
# -------------------------------
def hybrid_matching(model, nodes_t, nodes_t1, *, alpha, beta, gamma,
                    embedding_metric="l2", motion_clip=10.0,
                    require_embeddings=True, label_penalty=5.0):
    """Build the hybrid cost matrix from the trained NN branches and solve it.

    Returns (row_ind, col_ind, matrices) where matrices holds the full cost
    matrix and its per-component (embedding / motion / uncertainty) matrices.
    """
    n, m = len(nodes_t), len(nodes_t1)
    cost = np.zeros((n, m))
    emb_c = np.zeros((n, m))
    mot_c = np.zeros((n, m))
    unc_c = np.zeros((n, m))

    for i, n1 in enumerate(nodes_t):
        for j, n2 in enumerate(nodes_t1):
            with torch.no_grad():
                emb1 = model.get_embedding(n1)
                emb2 = model.get_embedding(n2)
                if require_embeddings and not (
                    torch.isfinite(emb1).all() and torch.isfinite(emb2).all()
                ):
                    raise EmbeddingMissingError(
                        f"Embedding branch produced non-finite output for pair "
                        f"({i},{j}); cannot proceed with require_embeddings=True."
                    )
                emb_dist = embedding_distance(
                    emb1.numpy(), emb2.numpy(), embedding_metric
                )

                motion_score = model.get_motion_score(n1, n2).item()
                motion_cost = 1.0 - motion_score

                uncertainty = (n1["uncertainty"] + n2["uncertainty"]) / 2.0

            mismatch = n1["label"] != n2["label"]
            emb_c[i][j] = emb_dist
            mot_c[i][j] = motion_cost
            unc_c[i][j] = uncertainty
            cost[i][j] = compute_hybrid_cost(
                emb_dist, motion_cost, uncertainty, alpha, beta, gamma,
                label_mismatch=mismatch, label_penalty=label_penalty,
            )

    row_ind, col_ind = solve_matching(cost)
    matrices = {"cost": cost, "embedding": emb_c, "motion": mot_c, "uncertainty": unc_c}
    return row_ind, col_ind, matrices


def evaluate(nodes_t, nodes_t1, row_ind, col_ind, matrices, threshold):
    """Score the assignment. Matches whose cost exceeds threshold are rejected."""
    cost = matrices["cost"]
    correct = 0
    accepted = 0
    rejected = 0
    for r, c in zip(row_ind, col_ind):
        if threshold is not None and cost[r][c] > threshold:
            rejected += 1
            continue
        accepted += 1
        if nodes_t[r]["instance"] == nodes_t1[c]["instance"]:
            correct += 1
    acc = (correct / len(nodes_t)) * 100 if len(nodes_t) > 0 else 0.0
    return {
        "nodes_t": len(nodes_t),
        "nodes_t1": len(nodes_t1),
        "accepted_matches": accepted,
        "rejected_by_threshold": rejected,
        "correct": correct,
        "accuracy_pct": acc,
    }


# -------------------------------
# CLI / MAIN
# -------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("true", "1", "yes", "y"):
        return True
    if str(v).lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {v!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 11: Hybrid motion + embedding matching")
    p.add_argument("--dataroot", default=os.environ.get("NUSCENES_DATAROOT", os.path.expanduser("~/data")))
    p.add_argument("--version", default=os.environ.get("NUSCENES_VERSION", "v1.0-mini"))
    p.add_argument("--alpha", type=float, default=0.4, help="Embedding weight")
    p.add_argument("--beta", type=float, default=0.4, help="Motion weight")
    p.add_argument("--gamma", type=float, default=0.2, help="Uncertainty weight")
    p.add_argument("--threshold", type=float, default=None, help="Reject matches with cost above this")
    p.add_argument("--motion-clip", type=float, default=10.0)
    p.add_argument("--normalize-method", choices=["minmax", "none"], default="minmax")
    p.add_argument("--embedding-metric", choices=["l2", "cosine"], default="l2")
    p.add_argument("--motion-mode", choices=["clip", "raw"], default="clip")
    p.add_argument("--require-embeddings", type=str2bool, default=True)
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="results/checkpoints")
    p.add_argument("--report-dir", default="results/reports")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=0, help="Training epochs (0 = inference only)")
    p.add_argument("--lr", type=float, default=0.001)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Validate weights early with a clear error.
    validate_weights(args.alpha, args.beta, args.gamma)

    seed = set_seed(args.seed)
    print(f"[seed] all RNGs fixed to seed={seed}")

    from nuscenes import NuScenes  # imported here so module import stays data-free

    print(f"Loading {args.version} from {args.dataroot} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    samples = collect_scene_samples(nusc, args.scene_index)
    print(f"Collected {len(samples)} samples from scene {args.scene_index}.")
    if len(samples) < 3:
        raise RuntimeError(
            f"Scene {args.scene_index} has only {len(samples)} samples; need >= 3 "
            f"for t-1 / t / t+1 matching."
        )

    model = HybridMatcher()

    training_pairs = build_training_pairs(nusc, samples)
    print(f"Built {len(training_pairs)} training pairs.")
    train_info = train_model(model, training_pairs, args.epochs, args.lr)

    # -------------------------------
    # EVALUATE on the first triple (t-1, t, t+1)
    # -------------------------------
    print("\n--- Running Hybrid Matching Validation ---")
    nodes_t = extract_nodes_with_velocity(nusc, samples[1], samples[0])
    nodes_t1 = extract_nodes_with_velocity(nusc, samples[2], samples[1])

    r_ind, c_ind, matrices = hybrid_matching(
        model, nodes_t, nodes_t1,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        embedding_metric=args.embedding_metric, motion_clip=args.motion_clip,
        require_embeddings=args.require_embeddings,
    )
    metrics = evaluate(nodes_t, nodes_t1, r_ind, c_ind, matrices, args.threshold)
    print(f"Hybrid System Accuracy: {metrics['accuracy_pct']:.2f}%")

    # -------------------------------
    # SAVE artifacts (matrices, checkpoint, JSON report)
    # -------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    np.save(os.path.join(args.report_dir, "cost_matrix.npy"), matrices["cost"])
    np.save(os.path.join(args.report_dir, "cost_embedding.npy"), matrices["embedding"])
    np.save(os.path.join(args.report_dir, "cost_motion.npy"), matrices["motion"])
    np.save(os.path.join(args.report_dir, "cost_uncertainty.npy"), matrices["uncertainty"])

    ckpt_path = os.path.join(args.checkpoint_dir, "hybrid_matcher.pth")
    torch.save(model.state_dict(), ckpt_path)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
            "threshold": args.threshold, "motion_clip": args.motion_clip,
            "normalize_method": args.normalize_method,
            "embedding_metric": args.embedding_metric, "motion_mode": args.motion_mode,
            "require_embeddings": args.require_embeddings,
            "epochs": args.epochs, "lr": args.lr, "seed": seed,
        },
        "dataset": {
            "dataroot": args.dataroot, "version": args.version,
            "scene_index": args.scene_index, "num_samples": len(samples),
            "num_training_pairs": len(training_pairs),
        },
        "training": train_info,
        "metrics": metrics,
        "artifacts": {
            "checkpoint": ckpt_path,
            "cost_matrix": os.path.join(args.report_dir, "cost_matrix.npy"),
        },
    }
    report_path = os.path.join(args.report_dir, "step11_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[report] wrote {report_path}")
    print(f"[checkpoint] wrote {ckpt_path}")

    print("\n--- Research Insights ---")
    print("1. Motion-aware branch provides local physical grounding.")
    print("2. Embedding branch provides global identity representation.")
    print("3. Uncertainty weight (gamma) mitigates noise from occluded or distant objects.")
    return report


if __name__ == "__main__":
    main()

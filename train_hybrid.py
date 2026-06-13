"""
Train the tracker's HybridMatcher branches and save a checkpoint that the
ResearchEngine (engine.py / step16 / step23c) can load.

This is the missing link in the original pipeline: the benchmark always ran a
*randomly initialized* HybridMatcher, so the embedding/motion branches added
noise rather than signal. Here we train both branches on held-out scenes using
the SAME feature layout the engine builds at inference time, then save the
weights to a checkpoint.

Usage:
    python train_hybrid.py                       # train on scenes[5:], save checkpoint
    python train_hybrid.py --epochs 20 --lr 1e-3 --out checkpoints/hybrid_matcher.pth
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from matcher.config import ResearchConfig          # noqa: E402
from matcher.engine import ResearchEngine          # noqa: E402
from matcher.hybrid_core import set_seed           # noqa: E402
from matcher.models import HybridMatcher           # noqa: E402


def parse_scene_slice(spec, n):
    """Parse '5:', ':5', '2:8' into a list of scene indices."""
    if ":" in spec:
        a, b = spec.split(":")
        a = int(a) if a else 0
        b = int(b) if b else n
        return list(range(a, b))
    return [int(spec)]


def collect_samples(nusc, scene_info):
    samples = []
    current = scene_info["first_sample_token"]
    while current != "":
        s = nusc.get("sample", current)
        samples.append(s)
        current = s["next"]
    return samples


def embedding_feat(node):
    return [node["position"][0], node["position"][1], node["vx"], node["vy"],
            node["confidence"], node["uncertainty"]]


def motion_feat(n1, n2):
    dx = n2["position"][0] - n1["position"][0]
    dy = n2["position"][1] - n1["position"][1]
    dist_norm = HybridMatcher.compute_distance(n1["position"], n2["position"]) / 20.0
    same_label = 1 if n1["label"] == n2["label"] else 0
    v_diff = HybridMatcher.compute_distance([n1["vx"], n1["vy"]], [n2["vx"], n2["vy"]])
    return [dx, dy, dist_norm, same_label, n1["confidence"], n2["confidence"],
            n1["uncertainty"], n2["uncertainty"], n1["vx"], n1["vy"], v_diff]


def build_dataset(nusc, engine, scene_indices, max_pairs):
    fa, fb, mf, labels = [], [], [], []
    for idx in scene_indices:
        samples = collect_samples(nusc, nusc.scene[idx])
        for i in range(len(samples) - 1):
            nodes_t = engine.extract_nodes(samples[i], samples[i - 1] if i > 0 else None)
            nodes_t1 = engine.extract_nodes(samples[i + 1], samples[i])
            for n1 in nodes_t:
                for n2 in nodes_t1:
                    label = 1.0 if n1["instance"] == n2["instance"] else 0.0
                    fa.append(embedding_feat(n1))
                    fb.append(embedding_feat(n2))
                    mf.append(motion_feat(n1, n2))
                    labels.append(label)
    fa = np.asarray(fa, dtype=np.float32)
    fb = np.asarray(fb, dtype=np.float32)
    mf = np.asarray(mf, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    if max_pairs and len(labels) > max_pairs:
        # Stratified cap: keep all positives (rare) + sample negatives.
        pos = np.where(labels == 1.0)[0]
        neg = np.where(labels == 0.0)[0]
        n_neg = max(max_pairs - len(pos), 0)
        sel_neg = np.random.choice(neg, size=min(n_neg, len(neg)), replace=False)
        sel = np.concatenate([pos, sel_neg])
        np.random.shuffle(sel)
        print(f"[data] capped {len(labels)} -> {len(sel)} pairs "
              f"(kept all {len(pos)} positives + {len(sel_neg)} negatives)")
        fa, fb, mf, labels = fa[sel], fb[sel], mf[sel], labels[sel]
    return fa, fb, mf, labels


def contrastive_loss(emb_a, emb_b, labels, margin):
    d = torch.norm(emb_a - emb_b, dim=1)
    pos = labels * d.pow(2)
    neg = (1 - labels) * torch.clamp(margin - d, min=0.0).pow(2)
    return (pos + neg).mean()


def train(model, fa, fb, mf, labels, epochs, lr, batch_size, margin, emb_weight):
    fa_t = torch.from_numpy(fa)
    fb_t = torch.from_numpy(fb)
    mf_t = torch.from_numpy(mf)
    y_t = torch.from_numpy(labels)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()
    n = len(labels)
    pos_rate = float(labels.mean())
    print(f"[train] {n} pairs, positive rate={pos_rate:.4f}")

    history = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            b = perm[start:start + batch_size]
            score = model.forward_motion(mf_t[b]).squeeze(1)
            motion_loss = bce(score, y_t[b])
            emb_a = model.forward_embedding(fa_t[b])
            emb_b = model.forward_embedding(fb_t[b])
            emb_loss = contrastive_loss(emb_a, emb_b, y_t[b], margin)
            loss = motion_loss + emb_weight * emb_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        avg = total / max(nb, 1)
        history.append(avg)
        print(f"[train] epoch {ep + 1}/{epochs} avg_loss={avg:.4f}")
    model.eval()
    return history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train HybridMatcher and save a tracker checkpoint")
    p.add_argument("--train-scenes", default="5:", help="Scene slice for training (default held-out '5:')")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-pairs", type=int, default=200000)
    p.add_argument("--margin", type=float, default=2.0)
    p.add_argument("--emb-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="checkpoints/hybrid_matcher.pth")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    print(f"[seed] {args.seed}")

    from nuscenes import NuScenes

    cfg = ResearchConfig()
    nusc = NuScenes(version=cfg.version, dataroot=cfg.dataroot, verbose=False)
    scene_indices = parse_scene_slice(args.train_scenes, len(nusc.scene))
    print(f"[data] training on scenes {scene_indices} "
          f"(of {len(nusc.scene)}); evaluation scenes [:5] are held out")

    model = HybridMatcher()
    engine = ResearchEngine(nusc, model, cfg)  # used only for extract_nodes

    fa, fb, mf, labels = build_dataset(nusc, engine, scene_indices, args.max_pairs)
    if len(labels) == 0:
        raise RuntimeError("No training pairs were built; check scene slice.")
    if labels.sum() == 0:
        raise RuntimeError("No positive (same-instance) pairs found; cannot train.")

    train(model, fa, fb, mf, labels, args.epochs, args.lr,
          args.batch_size, args.margin, args.emb_weight)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"[save] wrote checkpoint -> {args.out}")
    return args.out


if __name__ == "__main__":
    main()

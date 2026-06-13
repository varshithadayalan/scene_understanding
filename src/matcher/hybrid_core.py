"""Pure, dependency-light core for Step-11 hybrid temporal identity matching.

This module is importable WITHOUT nuScenes or PyTorch so the matching logic can
be unit-tested against tiny synthetic frames. The heavy NN-based research
pipeline lives in ``step11_hybrid_matching.py`` and reuses ``compute_hybrid_cost``
and ``validate_weights`` from here so the cost formula has a single source of
truth.

Hybrid cost formula::

    cost(i, j) = alpha * embedding_distance(i, j)
               + beta  * motion_cost(i, j)
               + gamma * uncertainty_cost(i, j)
               (+ label_penalty if labels mismatch and label enforcement is on)
"""
import math
import random

import numpy as np
from scipy.optimize import linear_sum_assignment


class HybridConfigError(ValueError):
    """Raised when hyperparameters / weights are invalid."""


class EmbeddingMissingError(ValueError):
    """Raised when a node lacks a required embedding and require_embeddings=True."""


def set_seed(seed):
    """Fix all RNG seeds used by the pipeline and return the seed for reporting."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for the pure path / tests
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def validate_weights(alpha, beta, gamma):
    """Validate the three hybrid weights. Raises HybridConfigError if invalid.

    Weights must be finite, non-negative numbers and not all zero.
    """
    for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HybridConfigError(f"{name} must be a real number, got {value!r}")
        if math.isnan(value) or math.isinf(value):
            raise HybridConfigError(f"{name} must be finite, got {value!r}")
        if value < 0:
            raise HybridConfigError(f"{name} must be >= 0, got {value}")
    if (alpha + beta + gamma) <= 0:
        raise HybridConfigError("alpha + beta + gamma must be > 0 (all weights are zero)")
    return alpha, beta, gamma


def compute_hybrid_cost(emb_dist, motion_cost, uncertainty, alpha, beta, gamma,
                        label_mismatch=False, label_penalty=5.0):
    """The single hybrid cost formula. Pure scalar arithmetic."""
    cost = (alpha * emb_dist) + (beta * motion_cost) + (gamma * uncertainty)
    if label_mismatch:
        cost += label_penalty
    return cost


def embedding_distance(e1, e2, metric="l2"):
    """Distance between two embedding vectors. metric is 'l2' or 'cosine'."""
    a = np.asarray(e1, dtype=float)
    b = np.asarray(e2, dtype=float)
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    if metric == "cosine":
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0.0 or nb == 0.0:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))
    raise HybridConfigError(f"Unknown embedding metric: {metric!r} (use 'l2' or 'cosine')")


def motion_cost(n1, n2, motion_clip=10.0):
    """Physical-grounding cost: how far n2 is from where n1's velocity predicts it.

    Returns a value in [0, 1] (distance clipped at ``motion_clip`` then normalized).
    """
    px = n1["position"][0] + n1.get("vx", 0.0)
    py = n1["position"][1] + n1.get("vy", 0.0)
    d = math.hypot(n2["position"][0] - px, n2["position"][1] - py)
    d = min(d, motion_clip)
    return d / motion_clip if motion_clip > 0 else 0.0


def _require_embedding(node, idx, side):
    emb = node.get("embedding")
    if emb is None:
        raise EmbeddingMissingError(
            f"nodes_{side}[{idx}] (instance={node.get('instance')!r}) has no 'embedding' "
            f"but require_embeddings=True. Provide embeddings or pass --require-embeddings false."
        )
    return emb


def build_cost_matrix(nodes_t, nodes_t1, *, alpha, beta, gamma,
                      embedding_metric="l2", motion_clip=10.0,
                      require_embeddings=True, enforce_label=True,
                      label_penalty=5.0):
    """Build the NxM hybrid cost matrix and its per-component matrices.

    Returns a dict with keys: ``cost``, ``embedding``, ``motion``, ``uncertainty``
    (each an N x M numpy array, where N=len(nodes_t), M=len(nodes_t1)).
    """
    validate_weights(alpha, beta, gamma)
    n, m = len(nodes_t), len(nodes_t1)
    cost = np.zeros((n, m), dtype=float)
    emb_c = np.zeros((n, m), dtype=float)
    mot_c = np.zeros((n, m), dtype=float)
    unc_c = np.zeros((n, m), dtype=float)

    for i, a in enumerate(nodes_t):
        emb_a = _require_embedding(a, i, "t") if require_embeddings else a.get("embedding")
        for j, b in enumerate(nodes_t1):
            emb_b = _require_embedding(b, j, "t1") if require_embeddings else b.get("embedding")

            if emb_a is not None and emb_b is not None:
                ed = embedding_distance(emb_a, emb_b, embedding_metric)
            else:
                ed = 0.0
            mc = motion_cost(a, b, motion_clip)
            uc = (a.get("uncertainty", 0.0) + b.get("uncertainty", 0.0)) / 2.0
            mismatch = enforce_label and (a.get("label") != b.get("label"))

            emb_c[i, j] = ed
            mot_c[i, j] = mc
            unc_c[i, j] = uc
            cost[i, j] = compute_hybrid_cost(
                ed, mc, uc, alpha, beta, gamma,
                label_mismatch=mismatch, label_penalty=label_penalty,
            )

    return {"cost": cost, "embedding": emb_c, "motion": mot_c, "uncertainty": unc_c}


def solve_matching(cost_matrix):
    """Run the Hungarian algorithm. Returns (row_ind, col_ind) arrays.

    By construction linear_sum_assignment yields a one-to-one assignment, so each
    row and each column appears at most once (no duplicate matches).
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

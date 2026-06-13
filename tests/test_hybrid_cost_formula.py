import numpy as np

from matcher.hybrid_core import build_cost_matrix, compute_hybrid_cost


def test_formula_is_weighted_sum():
    alpha, beta, gamma = 0.6, 0.3, 0.1
    emb, motion, unc = 2.0, 0.5, 0.25
    expected = alpha * emb + beta * motion + gamma * unc
    assert compute_hybrid_cost(emb, motion, unc, alpha, beta, gamma) == expected


def test_label_penalty_added_on_mismatch():
    base = compute_hybrid_cost(1.0, 1.0, 1.0, 0.4, 0.4, 0.2, label_mismatch=False)
    penalized = compute_hybrid_cost(1.0, 1.0, 1.0, 0.4, 0.4, 0.2,
                                    label_mismatch=True, label_penalty=5.0)
    assert penalized == base + 5.0


def test_matrix_entries_match_formula(frames):
    alpha, beta, gamma = 0.4, 0.4, 0.2
    nodes_t, nodes_t1 = frames[0], frames[1]
    out = build_cost_matrix(nodes_t, nodes_t1, alpha=alpha, beta=beta, gamma=gamma,
                            enforce_label=False)
    recomputed = alpha * out["embedding"] + beta * out["motion"] + gamma * out["uncertainty"]
    assert np.allclose(out["cost"], recomputed)

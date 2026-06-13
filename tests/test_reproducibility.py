import numpy as np

from matcher.hybrid_core import build_cost_matrix, set_seed, solve_matching


def test_set_seed_makes_rng_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_matching_is_deterministic(frames):
    out1 = build_cost_matrix(frames[1], frames[2], alpha=0.4, beta=0.4, gamma=0.2)
    out2 = build_cost_matrix(frames[1], frames[2], alpha=0.4, beta=0.4, gamma=0.2)
    assert np.array_equal(out1["cost"], out2["cost"])
    r1, c1 = solve_matching(out1["cost"])
    r2, c2 = solve_matching(out2["cost"])
    assert np.array_equal(r1, r2) and np.array_equal(c1, c2)

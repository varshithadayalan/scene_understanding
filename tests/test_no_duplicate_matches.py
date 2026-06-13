import numpy as np

from matcher.hybrid_core import build_cost_matrix, solve_matching


def test_hungarian_unique_assignment(frames):
    out = build_cost_matrix(frames[1], frames[2], alpha=0.4, beta=0.4, gamma=0.2)
    row_ind, col_ind = solve_matching(out["cost"])
    # No row or column is used twice.
    assert len(set(row_ind.tolist())) == len(row_ind)
    assert len(set(col_ind.tolist())) == len(col_ind)


def test_no_duplicates_on_random_matrix():
    rng = np.random.RandomState(0)
    cost = rng.rand(5, 7)
    row_ind, col_ind = solve_matching(cost)
    assert len(set(row_ind.tolist())) == len(row_ind)
    assert len(set(col_ind.tolist())) == len(col_ind)
    # Rectangular: assignment size is min(rows, cols).
    assert len(row_ind) == min(cost.shape)

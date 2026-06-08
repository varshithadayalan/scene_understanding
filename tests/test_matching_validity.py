from matcher.hybrid_core import build_cost_matrix, solve_matching


def test_matches_recover_correct_instances(frames):
    # Frame 1 -> Frame 2: linear motion + clustered embeddings make the correct
    # identity assignment (A->A, B->B) the unique optimum.
    nodes_t, nodes_t1 = frames[1], frames[2]
    out = build_cost_matrix(nodes_t, nodes_t1, alpha=0.4, beta=0.4, gamma=0.2)
    row_ind, col_ind = solve_matching(out["cost"])
    for r, c in zip(row_ind, col_ind):
        assert nodes_t[r]["instance"] == nodes_t1[c]["instance"]


def test_indices_within_bounds(frames):
    nodes_t, nodes_t1 = frames[0], frames[1]
    out = build_cost_matrix(nodes_t, nodes_t1, alpha=0.4, beta=0.4, gamma=0.2)
    row_ind, col_ind = solve_matching(out["cost"])
    assert all(0 <= r < len(nodes_t) for r in row_ind)
    assert all(0 <= c < len(nodes_t1) for c in col_ind)


def test_components_are_bounded(frames):
    # motion and uncertainty components are normalized to [0, 1].
    out = build_cost_matrix(frames[0], frames[1], alpha=0.4, beta=0.4, gamma=0.2)
    assert (out["motion"] >= 0).all() and (out["motion"] <= 1).all()
    assert (out["uncertainty"] >= 0).all() and (out["uncertainty"] <= 1).all()

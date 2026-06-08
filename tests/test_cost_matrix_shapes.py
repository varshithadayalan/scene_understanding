from matcher.hybrid_core import build_cost_matrix


def test_cost_matrix_is_NxM(frames):
    nodes_t, nodes_t1 = frames[0], frames[1]
    out = build_cost_matrix(nodes_t, nodes_t1, alpha=0.4, beta=0.4, gamma=0.2)
    n, m = len(nodes_t), len(nodes_t1)
    assert out["cost"].shape == (n, m)
    # Every per-component matrix has the same shape as the cost matrix.
    for key in ("embedding", "motion", "uncertainty"):
        assert out[key].shape == (n, m)


def test_cost_matrix_non_square(frames):
    # 2 nodes in t, 1 node in t1 -> shape (2, 1).
    nodes_t = frames[0]
    nodes_t1 = frames[1][:1]
    out = build_cost_matrix(nodes_t, nodes_t1, alpha=0.5, beta=0.3, gamma=0.2)
    assert out["cost"].shape == (2, 1)

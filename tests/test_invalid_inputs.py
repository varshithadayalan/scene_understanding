import copy

import pytest

from matcher.hybrid_core import (
    EmbeddingMissingError,
    HybridConfigError,
    build_cost_matrix,
    validate_weights,
)


@pytest.mark.parametrize("weights", [
    (-0.1, 0.5, 0.5),       # negative
    (0.0, 0.0, 0.0),        # all zero
    (float("nan"), 0.4, 0.2),  # nan
    (float("inf"), 0.4, 0.2),  # inf
    ("0.5", 0.4, 0.2),      # wrong type
])
def test_invalid_weights_raise(weights):
    with pytest.raises(HybridConfigError):
        validate_weights(*weights)


def test_valid_weights_pass():
    assert validate_weights(0.6, 0.3, 0.1) == (0.6, 0.3, 0.1)


def test_missing_embedding_raises_when_required(frames):
    nodes_t = copy.deepcopy(frames[0])
    nodes_t1 = copy.deepcopy(frames[1])
    del nodes_t1[0]["embedding"]  # drop one embedding
    with pytest.raises(EmbeddingMissingError):
        build_cost_matrix(nodes_t, nodes_t1, alpha=0.4, beta=0.4, gamma=0.2,
                          require_embeddings=True)


def test_missing_embedding_tolerated_when_not_required(frames):
    nodes_t = copy.deepcopy(frames[0])
    nodes_t1 = copy.deepcopy(frames[1])
    del nodes_t1[0]["embedding"]
    out = build_cost_matrix(nodes_t, nodes_t1, alpha=0.4, beta=0.4, gamma=0.2,
                            require_embeddings=False)
    assert out["cost"].shape == (len(nodes_t), len(nodes_t1))

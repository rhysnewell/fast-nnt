"""Shared fixtures: one deterministic 5-taxon distance matrix (the `small_5`
reference) keeps the tests reproducible and fast."""

import numpy as np
import pytest

LABELS = list("abcde")

DIST = np.array(
    [
        [0, 5, 9, 9, 8],
        [5, 0, 10, 10, 9],
        [9, 10, 0, 8, 7],
        [9, 10, 8, 0, 3],
        [8, 9, 7, 3, 0],
    ],
    dtype=float,
)


@pytest.fixture
def dist():
    return DIST.copy()


@pytest.fixture
def labels():
    return list(LABELS)


@pytest.fixture
def net(dist, labels):
    import fastnntpy as fn

    return fn.run_neighbour_net(dist, labels=labels)

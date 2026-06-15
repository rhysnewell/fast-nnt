"""Regression tests for the fastnntpy public API.

These pin the structure and index conventions every downstream consumer (and
the NEXUS/networx exporters) depends on:

* split records use 1-based taxon indices,
* node ids in translations/positions/edges are 0-based,
* edge ``split_id`` is 1-based into the split-records list.
"""

import math

import numpy as np
import pytest

import fastnntpy as fn


def test_labels_roundtrip(net, labels):
    assert net.get_labels() == labels


def test_split_records_partition_the_taxa(net, labels):
    recs = net.get_splits_records()
    ntax = len(labels)
    full = set(range(1, ntax + 1))  # 1-based taxon ids

    assert len(recs) > 0
    for label, weight, _conf, a_side, b_side in recs:
        assert isinstance(label, str)
        assert math.isfinite(weight)
        a, b = set(a_side), set(b_side)
        assert a and b                      # both sides non-empty
        assert a.isdisjoint(b)              # a proper bipartition...
        assert a | b == full               # ...covering every taxon


def test_node_positions_are_finite_and_0based(net):
    pos = net.get_node_positions()
    ids = [p[0] for p in pos]

    assert len(pos) > 0
    assert min(ids) == 0                    # 0-based node ids
    assert len(set(ids)) == len(ids)       # unique
    for _node_id, x, y in pos:
        assert math.isfinite(x) and math.isfinite(y)


def test_graph_edges_reference_valid_nodes_and_splits(net):
    node_ids = {p[0] for p in net.get_node_positions()}
    nsplits = len(net.get_splits_records())
    edges = net.get_graph_edges()

    assert len(edges) > 0
    for _eid, u, v, split_id, weight in edges:
        assert u in node_ids and v in node_ids
        assert 1 <= split_id <= nsplits
        assert math.isfinite(weight)


def test_translations_cover_all_taxa(net, labels):
    trans = net.get_node_translations()
    node_ids = {p[0] for p in net.get_node_positions()}

    assert sorted(lbl for _node, lbl in trans) == sorted(labels)
    assert all(node in node_ids for node, _lbl in trans)


def test_pandas_dataframe_input_infers_labels(dist, labels):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(dist, columns=labels)
    n = fn.run_neighbour_net(df)
    assert n.get_labels() == labels


@pytest.mark.parametrize("ordering", ["multiway", "closest-pair"])
@pytest.mark.parametrize("inference", ["active-set", "cg"])
def test_ordering_and_inference_options(dist, labels, ordering, inference):
    n = fn.run_neighbour_net(
        dist, labels=labels, ordering_method=ordering, inference_method=inference
    )
    assert len(n.get_splits_records()) > 0


def test_us_spelling_alias(dist, labels):
    a = fn.run_neighbour_net(dist, labels=labels)
    b = fn.run_neighbor_net(dist, labels=labels)
    assert a.get_labels() == b.get_labels()
    assert len(a.get_splits_records()) == len(b.get_splits_records())


def test_non_square_matrix_is_rejected():
    with pytest.raises(ValueError):
        fn.run_neighbour_net(np.zeros((2, 3)))

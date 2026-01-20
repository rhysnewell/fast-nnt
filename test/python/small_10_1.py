#! /usr/bin/env python

import fastnntpy as fn
import pandas as pd
data = pd.read_csv("test/data/small_10/small_10_1.csv")
n = fn.run_neighbour_net(
    data,
    ordering_method="splitstree4",
    inference_method="active-set",
)
n_alt = fn.run_neighbour_net(
    data,
    ordering_method="huson2023",
    inference_method="splitstree4",
)
print("Labels")
print(n.get_labels())
print("Alt Labels")
print(n_alt.get_labels())
print("Splits Records")
print(n.get_splits_records())
print("Node Translations")
print(n.get_node_translations())
print("Node Positions")
print(n.get_node_positions())
print("Graph Edges")
print(n.get_graph_edges())

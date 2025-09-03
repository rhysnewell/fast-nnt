#! /usr/bin/env python

import fastnntpy as fn
import pandas as pd
import os
from collections import Counter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


data = pd.read_csv("test/data/large/large_dist_matrix.csv")
n = fn.run_neighbor_net(data, max_iterations=5000, ordering_method="splitstree4", labels=data.columns)
n = fn.run_neighbour_net(data, max_iterations=5000, ordering_method="splitstree4", labels=data.columns)
print("Labels")
print(len(n.get_labels()))
print("Splits Records")
print(len(n.get_splits_records()))
print("Node Translations")
print(len(n.get_node_translations()))
print("Node Positions")
print(len(n.get_node_positions()))
print("Graph Edges")
print(len(n.get_graph_edges()))



def plot_fast_nnt_networkx(nx_obj, out_path="test/plots/fast_nnt_nx.png",
                           shift=0, node_size=10, font_size=7,
                           scale_width_by_weight=False, dpi=300):
    # -- data from PyNexus --
    labels = {i + shift: s for i, s in nx_obj.get_node_translations()}
    pos    = {i + shift: (x, y) for i, x, y in nx_obj.get_node_positions()}
    # corrected parsing order: (edge_id, u, v, sid, w)
    edges_raw = [ (u + shift, v + shift, w)
                  for (_eid, u, v, _sid, w) in nx_obj.get_graph_edges() ]

    # only keep edges whose endpoints have positions
    edges = [(u, v, w) for (u, v, w) in edges_raw if u in pos and v in pos]
    if not edges:
        raise ValueError("No drawable edges (endpoints missing positions).")

    # -- build graph --
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # leaves only (degree == 1)
    leaves = [n for n, d in G.degree() if d == 1]

    # edge widths (optional)
    if scale_width_by_weight:
        ws = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        wmax = max(ws) if ws else 1.0
        widths = [0.5 + 2.5 * (w / wmax) for w in ws]
    else:
        widths = 0.8

    # -- draw (no layout) --
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 8), dpi=dpi)

    nx.draw_networkx_edges(G, pos, width=widths, edge_color="black", alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=leaves, node_size=node_size, node_color="black")

    leaf_labels = {n: labels.get(n, str(n)) for n in leaves}
    nx.draw_networkx_labels(G, pos, labels=leaf_labels, font_size=font_size)

    plt.axis("equal"); plt.axis("off"); plt.tight_layout(pad=0.02)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    base, _ = os.path.splitext(out_path)
    plt.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.01)
    plt.close()
    return out_path

plot_fast_nnt_networkx(n, out_path="test/plots/fast_nnt_graph_networkx.png")

def save_neighbornet_plot(nx_obj,
                          out_path="test/plots/fast_nnt_neighbornet.png",
                          shift=0,
                          label_leaves=True,
                          scale_width_by_weight=True,
                          node_size=8,
                          font_size=6,
                          dpi=300):
    """
    Render NeighborNet as straight segments from Rust-provided positions.
    - Labels ONLY leaf nodes (degree==1).
    - No internal node markers, no edge labels.
    - Set `shift` if your IDs are 1/2-based etc.
    """
    # --- data from PyNexus / Rust ---
    labels = {i + shift: s for i, s in nx_obj.get_node_translations()}
    pos     = {i + shift: (x, y) for i, x, y in nx_obj.get_node_positions()}
    edges   = [(u + shift, v + shift, w) for (_eid, u, v, _sid, w) in nx_obj.get_graph_edges()]

    # keep only edges whose endpoints have positions
    edges = [(u, v, w) for (u, v, w) in edges if u in pos and v in pos]
    if not edges:
        raise ValueError("No drawable edges (endpoints missing positions).")

    # degrees → leaf detection
    deg = Counter()
    for u, v, _ in edges:
        deg[u] += 1; deg[v] += 1
    leaves = [n for n, d in deg.items() if d == 1]

    # segments for fast draw
    segs = [[pos[u], pos[v]] for (u, v, _w) in edges]

    # optional edge widths by weight
    if scale_width_by_weight:
        ws = np.array([w for *_ , w in edges], dtype=float)
        wmin, wptp = np.nanmin(ws), np.ptp(ws)
        widths = 0.5 + 2.5 * ((ws - wmin) / (wptp if wptp else 1.0))
    else:
        widths = 0.8

    # --- draw ---
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    ax.add_collection(LineCollection(segs, colors="black", linewidths=widths, alpha=0.9))

    # draw & label leaves only
    if leaves:
        xs = [pos[n][0] for n in leaves]
        ys = [pos[n][1] for n in leaves]
        ax.scatter(xs, ys, s=node_size, c="black", zorder=3)
        for n in leaves:
            x, y = pos[n]
            ax.text(x, y, labels.get(n, str(n)),
                    fontsize=font_size, ha="center", va="center", zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Fast-NNT NeighborNet")
    ax.autoscale(); ax.margins(0.02); ax.grid(True, alpha=0.15)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.splitext(out_path)[0] + ".svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path

save_neighbornet_plot(n, "test/plots/fast_nnt_graph_matplotlib.png", shift=2)
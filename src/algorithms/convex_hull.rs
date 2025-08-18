use fixedbitset::FixedBitSet;
use petgraph::prelude::{EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

use crate::phylo::phylo_graph::PhyloGraph;
use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;
use crate::splits::asplit::ASplit; // adjust path if different

/// Apply the convex hull algorithm to build a split network from `splits`.
/// `n_tax` is 1-based number of taxa; `taxa_labels` is 1-based in spirit (use taxa_labels[t-1]).
pub fn convex_hull_apply(
    n_tax: usize,
    taxa_labels: &[String],
    splits: &[ASplit], // 0-based Rust slice, we treat indices as 1-based where needed
    graph: &mut PhyloSplitsGraph,
) -> anyhow::Result<()> {
    // make an empty used-splits bitset (1..=nsplits)
    let mut used = FixedBitSet::with_capacity(splits.len() + 1);
    convex_hull_apply_with_used(n_tax, taxa_labels, splits, graph, &mut used)
}

/// Same as `convex_hull_apply` but reuses `used_splits` which are already in the graph.
/// `used_splits` is 1-based: bit s means split s (1..=nsplits).
pub fn convex_hull_apply_with_used(
    n_tax: usize,
    taxa_labels: &[String],
    splits: &[ASplit],
    graph: &mut PhyloSplitsGraph,
    used_splits: &mut FixedBitSet, // 1-based
) -> anyhow::Result<()> {
    if used_splits.count_ones(..) == splits.len() {
        return Ok(());
    }

    // Initialize graph if empty: single node holding all taxa
    if graph.base.graph.node_count() == 0 {
        let v = graph.base.new_node();
        graph.base.add_taxon(v, 1);
        for t in 2..=n_tax {
            graph.base.add_taxon(v, t);
        }
    } else {
        // Sanity checks (like Java prints)
        for t in 1..=n_tax {
            if graph.base.get_taxon_node(t).is_none() {
                log::warn!(
                    "ConvexHull: incomplete taxa mapping, taxon {} has no node",
                    t
                );
            }
        }
        for e in graph.base.graph.edge_indices() {
            if graph.get_split(e) == 0 {
                log::warn!("ConvexHull: edge without split: {:?}", e);
            }
        }
    }

    // Process splits in increasing size order (ties by id)
    let order = get_order_to_process_splits_in(splits, used_splits);

    for &j in &order {
        let sj = &splits[j - 1]; // 1-based -> 0-based
        let part_a = sj.get_a(); // FixedBitSet (1-based semantics)

        // Node hull membership: 0 => side-0(B), 1 => side-1(A), 2 => intersection; missing => unset
        let mut hulls: HashMap<NodeIndex, i32> = HashMap::new();
        let mut intersection_nodes: Vec<NodeIndex> = Vec::new();

        // allowed-splits sets for each side
        let mut splits0 = FixedBitSet::with_capacity(splits.len() + 1);
        let mut splits1 = FixedBitSet::with_capacity(splits.len() + 1);

        // Find splits that divide side-0 (the complement of A)
        for i in 1..=splits.len() {
            if !used_splits.contains(i) {
                continue;
            }
            if intersect2_cardinality(sj, false, &splits[i - 1], true) > 0
                && intersect2_cardinality(sj, false, &splits[i - 1], false) > 0
            {
                splits0.insert(i);
            }
        }

        // Find splits that divide side-1 (A)
        for i in 1..=splits.len() {
            if !used_splits.contains(i) {
                continue;
            }
            if intersect2_cardinality(sj, true, &splits[i - 1], true) > 0
                && intersect2_cardinality(sj, true, &splits[i - 1], false) > 0
            {
                splits1.insert(i);
            }
        }

        // Find start nodes: one from side-0 (not in A), one from side-1 (in A)
        let (mut start0, mut start1) = (None, None);
        for t in 1..=n_tax {
            if !part_a.contains(t) {
                start0 = graph.base.get_taxon_node(t);
            } else {
                start1 = graph.base.get_taxon_node(t);
            }
            if start0.is_some() && start1.is_some() {
                break;
            }
        }
        let start0 = start0.expect("ConvexHull: missing start0");
        let start1 = start1.expect("ConvexHull: missing start1");

        hulls.insert(start0, 0);
        if start0 == start1 {
            hulls.insert(start1, 2);
            intersection_nodes.push(start1);
        } else {
            hulls.insert(start1, 1);
        }

        // Build convex hulls via DFS over allowed splits
        convex_hull_path(
            graph,
            start0,
            &mut hulls,
            &splits0,
            &mut intersection_nodes,
            0,
        );
        convex_hull_path(
            graph,
            start1,
            &mut hulls,
            &splits1,
            &mut intersection_nodes,
            1,
        );

        // Duplicate each intersection node; connect duplicate v1--v with (split=j, weight=weight_j, label=j)
        for &v in &intersection_nodes {
            let v1 = graph.base.new_node();
            let e = graph.base.new_edge(v1, v)?;

            graph.set_split(e, j as i32);
            graph.base.set_weight(e, sj.weight);
            graph.base.set_edge_label(e, j.to_string());

            // Move taxa of A to v1, others remain on v
            let taxa_old = graph.base.get_node_taxon(v).unwrap_or(&[]);
            let taxa_list: Vec<usize> = taxa_old.to_vec();
            graph.base.clear_taxa_for_node(v);
            for taxon in taxa_list {
                if part_a.contains(taxon) {
                    graph.base.add_taxon(v1, taxon);
                } else {
                    graph.base.add_taxon(v, taxon);
                }
            }
        }

        // Rewire edges around each intersection node
        for &v in &intersection_nodes {
            // find duplicate v1 via the edge with split j
            let to_v1 = find_edge_with_split(graph, v, j).expect("duplicate edge not found");
            let v1 = graph.base.get_opposite(v, to_v1);

            // snapshot adjacency to avoid iterator invalidation on deletions
            let adj: Vec<EdgeIndex> = graph.base.graph.edges(v).map(|re| re.id()).collect();
            for consider in adj {
                if consider == to_v1 {
                    continue;
                }
                let w = graph.base.get_opposite(v, consider);

                let mark = hulls.get(&w).copied().unwrap_or(-1);
                if mark == -1 {
                    // do nothing
                } else if mark == 1 {
                    // belongs to the other side: add v1--w, copy attributes, delete v--w
                    let consider_dup = graph.base.new_edge(v1, w)?;
                    let sid = graph.get_split(consider);
                    graph.set_split(consider_dup, sid);
                    graph
                        .base
                        .set_weight(consider_dup, graph.base.weight(consider));
                    let copied_label: Option<String> =
                        { graph.base.edge_label(consider).map(|s| s.to_string()) }; // immutable borrow ends here

                    if let Some(lbl) = copied_label {
                        graph.base.set_edge_label(consider_dup, lbl);
                    }
                    graph.base.graph.remove_edge(consider);
                } else if mark == 2 {
                    // w is also intersection: connect duplicates if not already connected
                    let w1 = {
                        let e_to_w1 =
                            find_edge_with_split(graph, w, j).expect("w1 duplicate edge not found");
                        graph.base.get_opposite(w, e_to_w1)
                    };
                    if graph.base.graph.find_edge(v1, w1).is_none() {
                        let consider_dup = graph.base.new_edge(v1, w1)?;
                        let sid = graph.get_split(consider);
                        graph.set_split(consider_dup, sid);
                        graph
                            .base
                            .set_weight(consider_dup, graph.base.weight(consider));
                        let copied_label: Option<String> =
                            { graph.base.edge_label(consider).map(|s| s.to_string()) }; // immutable borrow ends here

                        if let Some(lbl) = copied_label {
                            graph.base.set_edge_label(consider_dup, lbl);
                        }
                    }
                }
            }
        }

        // mark split used
        used_splits.insert(j);
    }

    // Clear node labels, then assign leaf labels from taxa_labels (like GraphUtils.addLabels)
    let node_indices = graph.base.graph.node_indices().collect::<Vec<_>>();
    for v in node_indices {
        graph.base.set_node_label(v, "");
    }
    add_leaf_labels_from_taxa(graph, taxa_labels);

    // Optional: emulate Java’s temporary edge labeling for first edge per split, then clear all again
    let edge_indices = graph.base.graph.edge_indices().collect::<Vec<_>>();
    {
        let mut seen = FixedBitSet::with_capacity(splits.len() + 1);
        for e in &edge_indices {
            let s = graph.get_split(*e);
            if s > 0 && !seen.contains(s as usize) {
                seen.insert(s as usize);
                graph.base.set_edge_label(*e, s.to_string());
            } else {
                graph.base.set_edge_label(*e, "");
            }
        }
    }
    // Java clears all edge labels at the very end:
    for e in edge_indices {
        graph.base.set_edge_label(e, "");
    }

    Ok(())
}

/* ---------------- helpers ---------------- */

/// Build the convex hull path for one side (`side` is 0 or 1)
fn convex_hull_path(
    g: &mut PhyloSplitsGraph,
    start: NodeIndex,
    hulls: &mut HashMap<NodeIndex, i32>,
    allowed_splits: &FixedBitSet, // bits are 1-based split ids
    intersection_nodes: &mut Vec<NodeIndex>,
    side: i32, // 0 or 1
) {
    let mut visited: HashSet<EdgeIndex> = HashSet::new();
    let mut stack: Vec<NodeIndex> = vec![start];

    while let Some(v) = stack.pop() {
        // iterate over incident edges
        let adj: Vec<EdgeIndex> = g.base.graph.edges(v).map(|re| re.id()).collect();
        for f in adj {
            if visited.contains(&f) {
                continue;
            }
            let s_id = g.get_split(f);
            if s_id > 0 && allowed_splits.contains(s_id as usize) {
                visited.insert(f);
                let w = g.base.get_opposite(v, f);

                match hulls.get(&w).copied() {
                    None => {
                        hulls.insert(w, side);
                        stack.push(w);
                    }
                    Some(mark) if mark == (1 - side) => {
                        hulls.insert(w, 2);
                        intersection_nodes.push(w);
                        stack.push(w);
                    }
                    _ => { /* already on our hull or intersection, nothing */ }
                }
            }
        }
    }
}

/// Find the (unique) edge incident to `v` that carries split id `split_id`
fn find_edge_with_split(g: &PhyloSplitsGraph, v: NodeIndex, split_id: usize) -> Option<EdgeIndex> {
    for re in g.base.graph.edges(v) {
        let e = re.id();
        if g.get_split(e) == split_id as i32 {
            return Some(e);
        }
    }
    None
}

/// Order splits by increasing size, then by id (1-based), and exclude already-used ones
fn get_order_to_process_splits_in(
    splits: &[ASplit],
    used: &FixedBitSet, // 1-based
) -> Vec<usize> {
    let mut items: Vec<(usize, usize)> = Vec::with_capacity(splits.len());
    for s in 1..=splits.len() {
        if !used.contains(s) {
            items.push((splits[s - 1].size(), s)); // (size, id)
        }
    }
    items.sort_unstable(); // sort by size, then id
    items.into_iter().map(|(_, s)| s).collect()
}

/// Intersection cardinality of chosen sides: if `side_a=true` use A-part else B-part
fn intersect2_cardinality(a: &ASplit, side_a: bool, b: &ASplit, side_b: bool) -> usize {
    let pa = if side_a { a.get_a() } else { a.get_b() };
    let pb = if side_b { b.get_a() } else { b.get_b() };
    // count intersection; both are 1-based bitsets of size >= ntax+1
    let mut cnt = 0usize;
    // iterate over set bits of smaller
    if pa.count_ones(..) <= pb.count_ones(..) {
        for i in pa.ones() {
            if pb.contains(i) {
                cnt += 1;
            }
        }
    } else {
        for i in pb.ones() {
            if pa.contains(i) {
                cnt += 1;
            }
        }
    }
    cnt
}

/// Assign node labels for leaves (degree==1) if they map to exactly one taxon id.
fn add_leaf_labels_from_taxa(g: &mut PhyloSplitsGraph, taxa_labels: &[String]) {
    let base: &mut PhyloGraph = &mut g.base;
    let node_indices = base.graph.node_indices().collect::<Vec<_>>();
    for v in node_indices {
        let deg = base.graph.neighbors(v).count();
        if deg == 1 {
            if let Some(n2t) = base.node2taxa() {
                if let Some(list) = n2t.get(&v) {
                    if list.len() == 1 {
                        let t = list[0];
                        if t >= 1 && t <= taxa_labels.len() {
                            base.set_node_label(v, &taxa_labels[t - 1]);
                        }
                    }
                }
            }
        }
    }
}

use anyhow::{Result, anyhow};
use fixedbitset::FixedBitSet;
use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use std::collections::{HashMap, HashSet};

use crate::phylo::phylo_graph::PhyloGraph;
use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;
use crate::splits::asplit::ASplit;

/* ------------------------------------------------------------------------- */
/* Convex hull network construction (rotation/embedding aware)               */
/* ------------------------------------------------------------------------- */

/// Apply the convex hull algorithm to build a split network from `splits`.
/// `n_tax` is the number of taxa (1-based semantics).
/// `splits` is 0-based but we treat ids as 1..=nsplits in bitsets and logs.
pub fn convex_hull_apply(
    n_tax: usize,
    taxa_labels: &[String],
    splits: &[ASplit],
    graph: &mut PhyloSplitsGraph,
) -> Result<()> {
    let mut used = FixedBitSet::with_capacity(splits.len() + 1);
    convex_hull_apply_with_used(n_tax, taxa_labels, splits, graph, &mut used)
}

/// Same as `convex_hull_apply` but reuses `used_splits` (1-based bits).
pub fn convex_hull_apply_with_used(
    n_tax: usize,
    taxa_labels: &[String],
    splits: &[ASplit],
    graph: &mut PhyloSplitsGraph,
    used_splits: &mut FixedBitSet,
) -> Result<()> {
    // finish early if everything is already used
    if used_splits.count_ones(..) == splits.len() {
        return Ok(());
    }
    used_splits.grow(splits.len() + 1);

    // Initialize graph if empty: one node holding all taxa
    if graph.base.graph.node_count() == 0 {
        let v = graph.base.new_node();
        for t in 1..=n_tax {
            graph.base.add_taxon(v, t);
        }
    } else {
        // lightweight sanity (optional)
        for t in 1..=n_tax {
            if graph.base.get_taxon_node(t).is_none() {
                log::warn!("ConvexHull: taxon {} has no node", t);
            }
        }
        for e in graph.base.graph.edge_indices() {
            if graph.get_split(e) == 0 {
                log::warn!("ConvexHull: edge {:?} has no split id", e);
            }
        }
    }

    // Process splits in increasing size order (ties by id), skipping ones already used
    let order = get_order_to_process_splits_in(splits, used_splits);

    for &sid in &order {
        let s = &splits[sid - 1];
        let part_a = s.get_a(); // 1-based membership of side A

        // Marking per node: 0 => in B, 1 => in A, 2 => intersection A∩B
        let mut hulls: HashMap<NodeIndex, i32> = HashMap::new();
        let mut intersections: Vec<NodeIndex> = Vec::new();

        // Determine which already-placed splits divide each side
        let mut splits0 = FixedBitSet::with_capacity(splits.len() + 1);
        let mut splits1 = FixedBitSet::with_capacity(splits.len() + 1);

        for i in 1..=splits.len() {
            if !used_splits.contains(i) {
                continue;
            }
            // splits that cut the complement of A (side 0)
            if intersect2_cardinality(s, false, &splits[i - 1], true) > 0
                && intersect2_cardinality(s, false, &splits[i - 1], false) > 0
            {
                splits0.insert(i);
            }
            // splits that cut A (side 1)
            if intersect2_cardinality(s, true, &splits[i - 1], true) > 0
                && intersect2_cardinality(s, true, &splits[i - 1], false) > 0
            {
                splits1.insert(i);
            }
        }

        // Pick one start node from each side by taxon membership
        let (mut start0, mut start1) = (None, None);
        for t in 1..=n_tax {
            if !part_a.contains(t) && start0.is_none() {
                start0 = graph.base.get_taxon_node(t);
            }
            if part_a.contains(t) && start1.is_none() {
                start1 = graph.base.get_taxon_node(t);
            }
            if start0.is_some() && start1.is_some() {
                break;
            }
        }
        let start0 = start0.ok_or_else(|| anyhow!("ConvexHull: missing start node for side B"))?;
        let start1 = start1.ok_or_else(|| anyhow!("ConvexHull: missing start node for side A"))?;

        hulls.insert(start0, 0);
        if start0 == start1 {
            hulls.insert(start1, 2);
            intersections.push(start1);
        } else {
            hulls.insert(start1, 1);
        }

        // Grow both hulls via DFS over allowed splits, using rotation order
        convex_hull_path(graph, start0, &mut hulls, &splits0, &mut intersections, 0);
        convex_hull_path(graph, start1, &mut hulls, &splits1, &mut intersections, 1);

        // Make intersection list unique (can be discovered from both sides)
        {
            let mut seen: HashSet<NodeIndex> = HashSet::new();
            intersections.retain(|v| seen.insert(*v));
        }

        // Duplicate each intersection node: add v1--v carrying split sid
        for &v in &intersections {
            let v1 = graph.base.new_node();

            // Place the hook edge deterministically: insert at `v` AFTER its first adjacent edge (if any)
            let e_hook = if let Some(f0) = graph.first_adjacent_edge(v) {
                graph.new_edge_after(v1, v, f0)?
            } else {
                graph.new_edge(v1, v)?
            };
            graph.set_split(e_hook, sid as i32);
            graph.base.set_weight(e_hook, s.weight);
            graph.base.set_edge_label(e_hook, sid.to_string());

            // Move taxa in A to v1; others stay on v
            let taxa_old = graph.base.get_node_taxon(v).unwrap_or(&[]);
            let list = taxa_old.to_vec();
            graph.base.clear_taxa_for_node(v);
            for taxon in list {
                if part_a.contains(taxon) {
                    graph.base.add_taxon(v1, taxon);
                } else {
                    graph.base.add_taxon(v, taxon);
                }
            }
        }

        // Rewire edges around each intersection node v, preserving rotation:
        // - For neighbor w marked side 1 (A), move v--w to v1--w:
        //   insert the new edge on w's rotation AFTER `consider` then delete `consider`.
        for &v in &intersections {
            let e_to_v1 = find_edge_with_split(graph, v, sid).expect("duplicate hook missing");
            let v1 = graph.base.get_opposite(v, e_to_v1);

            // snapshot current rotation at v to avoid iterator invalidation
            let adj_v: Vec<EdgeIndex> = graph.rotation(v).to_vec();
            for consider in adj_v {
                if consider == e_to_v1 {
                    continue;
                }
                let w = graph.base.get_opposite(v, consider);
                let mark = *hulls.get(&w).unwrap_or(&-1);

                if mark == 1 {
                    // Move consider to connect v1--w. Preserve w’s local order by placing AFTER `consider`.
                    let consider_dup = graph.new_edge_after(v1, w, consider)?;
                    // copy attributes (split/weight/label)
                    let sid_old = graph.get_split(consider);
                    if sid_old != 0 {
                        graph.set_split(consider_dup, sid_old);
                    }
                    let wgt = graph.base.weight(consider);
                    if wgt != crate::phylo::phylo_graph::DEFAULT_WEIGHT {
                        graph.base.set_weight(consider_dup, wgt);
                    }
                    if let Some(lbl) = graph.base.edge_label(consider).map(|s| s.to_string()) {
                        graph.base.set_edge_label(consider_dup, lbl);
                    }
                    // delete original edge; rotation is updated by wrapper
                    graph.remove_edge(consider);
                } else if mark == 2 {
                    // w is also an intersection: connect v1--w1 if not present
                    let e_w_to_w1 =
                        find_edge_with_split(graph, w, sid).expect("peer duplicate missing");
                    let w1 = graph.base.get_opposite(w, e_w_to_w1);

                    if graph.base.graph.find_edge(v1, w1).is_none() {
                        // Place on w1 side AFTER e_w_to_w1 for stability
                        let e_new = graph.new_edge_after(v1, w1, e_w_to_w1)?;
                        let sid_old = graph.get_split(consider);
                        if sid_old != 0 {
                            graph.set_split(e_new, sid_old);
                        }
                        let wgt = graph.base.weight(consider);
                        if wgt != crate::phylo::phylo_graph::DEFAULT_WEIGHT {
                            graph.base.set_weight(e_new, wgt);
                        }
                        if let Some(lbl) = graph.base.edge_label(consider).map(|s| s.to_string()) {
                            graph.base.set_edge_label(e_new, lbl);
                        }
                    }
                }
            }
        }

        used_splits.insert(sid);
    }

    // Clear node labels, then label leaves from taxa
    for v in graph.base.graph.node_indices().collect::<Vec<_>>() {
        graph.base.set_node_label(v, "");
    }
    add_leaf_labels_from_taxa(graph, taxa_labels);

    // Optional edge label trick (first edge per split) — then clear (matches Java behavior)
    let edges = graph.base.graph.edge_indices().collect::<Vec<_>>();
    {
        let mut seen = FixedBitSet::with_capacity(splits.len() + 1);
        for &e in &edges {
            let s = graph.get_split(e);
            if s > 0 && !seen.contains(s as usize) {
                seen.insert(s as usize);
                graph.base.set_edge_label(e, s.to_string());
            } else {
                graph.base.set_edge_label(e, "");
            }
        }
    }
    for e in edges {
        graph.base.set_edge_label(e, "");
    }

    Ok(())
}

/* ---------------- rotation-aware helpers ---------------- */

/// Grow one hull from `start` over edges whose split ids are in `allowed_splits`.
/// Traversal order follows the node’s rotation for determinism.
fn convex_hull_path(
    g: &mut PhyloSplitsGraph,
    start: NodeIndex,
    hulls: &mut HashMap<NodeIndex, i32>,
    allowed_splits: &FixedBitSet, // 1-based split ids
    intersections: &mut Vec<NodeIndex>,
    side: i32, // 0 (B) or 1 (A)
) {
    let mut seen_edges: HashSet<EdgeIndex> = HashSet::new();
    let mut stack: Vec<NodeIndex> = vec![start];

    while let Some(v) = stack.pop() {
        // iterate incident edges in rotation order
        for &f in g.rotation(v) {
            if seen_edges.contains(&f) {
                continue;
            }
            let sid = g.get_split(f);
            if sid <= 0 || !allowed_splits.contains(sid as usize) {
                continue;
            }

            seen_edges.insert(f);
            let w = g.base.get_opposite(v, f);

            match hulls.get(&w).copied() {
                None => {
                    hulls.insert(w, side);
                    stack.push(w);
                }
                Some(mark) if mark == (1 - side) => {
                    if *hulls.entry(w).or_insert(2) != 2 {
                        hulls.insert(w, 2);
                    }
                    intersections.push(w);
                    stack.push(w);
                }
                _ => { /* already on this hull or intersection */ }
            }
        }
    }
}

/// Find the (unique) edge incident to `v` carrying split id `split_id` (use rotation for determinism).
fn find_edge_with_split(g: &PhyloSplitsGraph, v: NodeIndex, split_id: usize) -> Option<EdgeIndex> {
    for &e in g.rotation(v) {
        if g.get_split(e) == split_id as i32 {
            return Some(e);
        }
    }
    None
}

/// Order splits by increasing size then id, skipping already-used ones.
fn get_order_to_process_splits_in(splits: &[ASplit], used: &FixedBitSet) -> Vec<usize> {
    let mut items: Vec<(usize, usize)> = Vec::with_capacity(splits.len());
    for s in 1..=splits.len() {
        if !used.contains(s) {
            items.push((splits[s - 1].size(), s));
        }
    }
    items.sort_unstable();
    items.into_iter().map(|(_, s)| s).collect()
}

/// Intersection cardinality of chosen sides: if `side_a=true` use A-part else B-part.
fn intersect2_cardinality(a: &ASplit, side_a: bool, b: &ASplit, side_b: bool) -> usize {
    let pa = if side_a { a.get_a() } else { a.get_b() };
    let pb = if side_b { b.get_a() } else { b.get_b() };

    let (small, large) = if pa.count_ones(..) <= pb.count_ones(..) {
        (pa, pb)
    } else {
        (pb, pa)
    };
    small.ones().filter(|&i| large.contains(i)).count()
}

/// Assign node labels for leaves (degree==1) if exactly one taxon maps to that node.
fn add_leaf_labels_from_taxa(g: &mut PhyloSplitsGraph, taxa_labels: &[String]) {
    let base: &mut PhyloGraph = &mut g.base;
    for v in base.graph.node_indices().collect::<Vec<_>>() {
        if base.graph.neighbors(v).count() == 1 {
            if let Some(n2t) = base.node2taxa() {
                if let Some(list) = n2t.get(&v) {
                    if list.len() == 1 {
                        let t = list[0];
                        if (1..=taxa_labels.len()).contains(&t) {
                            base.set_node_label(v, &taxa_labels[t - 1]);
                        }
                    }
                }
            }
        }
    }
}

/* ---------------- tiny access shim ---------------- */

/// Expose rotation slice for iteration (read-only). Add this to `impl PhyloSplitsGraph`.
trait RotationAccess {
    fn rotation(&self, v: NodeIndex) -> &[EdgeIndex];
}
impl RotationAccess for PhyloSplitsGraph {
    #[inline]
    fn rotation(&self, v: NodeIndex) -> &[EdgeIndex] {
        self.rot(v)
    }
}

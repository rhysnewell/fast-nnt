use anyhow::{Result, anyhow, bail};
use fixedbitset::FixedBitSet;
use petgraph::{
    csr::IndexType,
    prelude::{EdgeIndex, NodeIndex},
    visit::{EdgeRef, NodeIndexable},
};
use std::collections::{HashMap, HashSet};

use crate::{
    algorithms::convex_hull::convex_hull_apply,
    data::splits_blocks::{SplitsBlock, SplitsProvider},
    phylo::phylo_splits_graph::PhyloSplitsGraph,
    splits::asplit::ASplitView,
};
use log::info;

#[derive(Copy, Clone, Debug)]
pub struct Pt(pub f64, pub f64);

/// Options for EqualAngle
#[derive(Copy, Clone, Debug)]
pub struct EqualAngleOpts {
    /// Use edge weights when laying out coordinates (true = weighted steps)
    pub use_weights: bool,
    /// Total circle angle for split directions (default 360.0)
    pub total_angle_deg: f64,
    /// Root split id for tiny offsets when !use_weights (0/neg = none)
    pub root_split: i32,
}

impl Default for EqualAngleOpts {
    fn default() -> Self {
        Self {
            use_weights: true,
            total_angle_deg: 360.0,
            root_split: 0,
        }
    }
}

/// Compute a split network with the (ported) Equal-Angle workflow.
/// - `taxa_labels` is your “TaxaBlock” (1-based semantic: label for taxon `t` is `taxa_labels[t-1]`)
/// - `splits` is the circular `SplitsBlock` with a cycle
/// - `graph` is mutated in-place
/// - `forbidden_splits` skips setting angles for these split ids (1-based); pass `None` to set all
/// - `used_splits` is cleared and then bits set for each non-trivial circular split we consume
///
/// Returns `Ok(all_used)` where `all_used == true` iff every non-trivial split is circular
pub fn equal_angle_apply(
    opts: EqualAngleOpts,
    taxa_labels: &[String],
    splits: &SplitsBlock,
    graph: &mut PhyloSplitsGraph,
    forbidden_splits: Option<&FixedBitSet>,
    used_splits: &mut FixedBitSet,
) -> Result<bool> {
    let t0 = std::time::Instant::now();
    graph.clear();
    used_splits.clear();

    let ntax = taxa_labels.len();
    if ntax == 0 {
        info!("EqualAngle: no taxa; nothing to do");
        return Ok(true);
    }

    let cycle = {
        let c = splits
            .cycle()
            .ok_or_else(|| anyhow::anyhow!("SplitsBlock has no cycle"))?;
        normalize_cycle_1based(c)?
    };

    // 1) init star graph with center and trivial split edges
    init_graph_star(taxa_labels, splits, &cycle, graph)?;

    // 2) non-trivial splits ordered by |partContaining(1)|
    let ordered = get_nontrivial_splits_ordered(splits);

    // 3) wrap each circular split (TODO: full wrapping)
    let mut all_used = true;
    for &sid in &ordered {
        if is_circular_by_cycle(splits, sid, &cycle) {
            // Wrap split sid around the cycle
            wrap_split(ntax, splits, sid, &cycle, graph)?;
            // Placeholder: log and mark used. We don’t modify graph topology here yet.
            used_splits.grow(splits.nsplits() + 1);
            used_splits.insert(sid);
        } else {
            all_used = false;
        }
    }

    // 4) remove temporary trivial edges (none if we had trivial splits, some if we didn’t)
    remove_temporary_trivial_edges(graph);

    if !all_used {
        // assign remaining with ConvexHull
        convex_hull_apply(ntax, taxa_labels, splits.get_splits(), graph)?;
    }
    // 5) assign angles to edges from split directions
    assign_angles_to_edges(
        ntax,
        splits,
        &cycle,
        graph,
        forbidden_splits,
        opts.total_angle_deg,
    );

    // 6) rotate so the edge leaving taxon 1 points to 9 o’clock
    let (_align, _extra) = rotate_angles_align_then_offset(graph, 1, 180.0, 0.0);

    // 7) assign coordinates from angles (DFS through graph)
    let _coords = assign_coordinates_to_nodes(opts.use_weights, graph, 1, opts.root_split);

    // 8) add leaf node labels from taxa
    add_leaf_labels_from_taxa(graph, taxa_labels);

    info!(
        "EqualAngle: initialized network with {} nodes, {} edges in {:?} (use_weights={})",
        graph.base.graph.node_count(),
        graph.base.graph.edge_count(),
        t0.elapsed(),
        opts.use_weights
    );

    Ok(all_used)
}

/* ----------------------- Step helpers, mostly 1:1 ports ----------------------- */

/// normalize cycle so that cycle[1] == 1; keep 1-based convention with index 0 unused
pub fn normalize_cycle_1based(cycle_1based: &[usize]) -> Result<Vec<usize>> {
    if cycle_1based.is_empty() || cycle_1based[0] != 0 {
        bail!("cycle must be 1-based with index 0 unused");
    }
    let n = cycle_1based.len() - 1;
    let mut k = 1usize;
    while k <= n && cycle_1based[k] != 1 {
        k += 1;
    }
    if k > n {
        bail!("cycle does not contain taxon 1");
    }

    let mut out = vec![0usize; n + 1];
    let mut i = 1;
    let mut j = k;
    while j <= n {
        out[i] = cycle_1based[j];
        i += 1;
        j += 1;
    }
    j = 1;
    while i <= n {
        out[i] = cycle_1based[j];
        i += 1;
        j += 1;
    }
    Ok(out)
}

/// Build the initial star: center node connected to each taxon leaf in cycle order.
/// Set edge split to trivial split id if present; else split=-1 (temporary).
fn init_graph_star(
    taxa: &[String],
    splits: &SplitsBlock,
    cycle: &[usize], // 1-based
    g: &mut PhyloSplitsGraph,
) -> Result<()> {
    g.clear();

    let ntax = taxa.len();
    let mut taxon2trivial: Vec<i32> = vec![0; ntax + 1];
    for s in 1..=splits.nsplits() {
        let sp = splits.split(s);
        if sp.size() == 1 {
            if let Some(t) = first_member(sp.smaller_part()) {
                taxon2trivial[t] = s as i32;
            }
        }
    }

    let center = g.base.new_node();
    // remember leaf node per taxon
    for i in 1..=ntax {
        let t = cycle[i];
        let leaf = g.base.new_node();
        g.base.add_taxon(leaf, t);
        let e = g.new_edge(center, leaf)?;
        let sid = taxon2trivial[t];
        if sid != 0 {
            let w = splits.split(sid as usize).weight();
            g.base.set_weight(e, w);
            g.set_split(e, sid);
        } else {
            g.set_split(e, -1); // temporary trivial edge
        }
    }
    Ok(())
}

/// Gather non-trivial split ids sorted by |partContaining(1)|
fn get_nontrivial_splits_ordered(splits: &SplitsBlock) -> Vec<usize> {
    let mut pairs: Vec<(usize, usize)> = Vec::new(); // (size, sid)
    for s in 1..=splits.nsplits() {
        let sp = splits.split(s);
        if sp.size() > 1 {
            let size = sp.part_containing(1).count_ones(..);
            pairs.push((size, s));
        }
    }
    pairs.sort_by_key(|p| p.0);
    pairs.into_iter().map(|p| p.1).collect()
}

/// Simple circularity check: split part not containing cycle[1] must be a contiguous arc in cycle
fn is_circular_by_cycle(splits: &SplitsBlock, sid: usize, cycle: &[usize]) -> bool {
    let sp = splits.split(sid);
    let ntax = cycle.len() - 1;
    let part = sp.part_not_containing(cycle[1]);

    // collect cycle positions (2..=n) for taxa in part
    let mut pos: Vec<usize> = Vec::new();
    for i in 2..=ntax {
        let t = cycle[i];
        if part.contains(t) {
            pos.push(i);
        }
    }
    if pos.is_empty() {
        return true;
    } // degenerate
    // must be contiguous (allow wrap-around handled by cycle normalization)
    for w in pos.windows(2) {
        if w[1] != w[0] + 1 {
            return false;
        }
    }
    true
}

/// Remove edges with split == -1 by merging leaf node into its neighbor (move taxa labels)
fn remove_temporary_trivial_edges(g: &mut PhyloSplitsGraph) {
    let mut to_remove: Vec<EdgeIndex> = Vec::new();
    for e in g.base.graph.edge_indices() {
        if g.get_split(e) == -1 {
            to_remove.push(e);
        }
    }
    for e in to_remove {
        if let Some((u, v)) = g.base.graph.edge_endpoints(e) {
            // choose leaf node (degree 1) to delete; move its taxa to the other node
            let du = g.base.graph.neighbors(u).count();
            let dv = g.base.graph.neighbors(v).count();
            let (leaf, keep) = if du == 1 {
                (u, v)
            } else if dv == 1 {
                (v, u)
            } else {
                continue;
            };

            if let Some(list) = g
                .base
                .node2taxa()
                .expect("No taxa mapping")
                .get(&leaf)
                .cloned()
            {
                for t in list {
                    g.base.add_taxon(keep, t);
                }
                // clear on leaf
                g.base.clear_taxa_for_node(leaf);
            }
            g.base.graph.remove_edge(e);
            // remove leaf node completely
            if g.base.graph.neighbors(leaf).count() == 0 {
                g.base.graph.remove_node(leaf);
            }
        }
    }
}

/// Assign angles to edges: compute split directions (per split id) on a circle,
/// then stamp on edges unless forbidden.
pub fn assign_angles_to_edges(
    ntaxa: usize,
    splits: &SplitsBlock,
    cycle: &[usize],
    g: &mut PhyloSplitsGraph,
    forbidden: Option<&fixedbitset::FixedBitSet>,
    total_angle: f64,
) {
    let split2angle = assign_angles_to_splits(ntaxa, splits, cycle, total_angle);

    // Collect edge ids first to avoid holding an immutable borrow during mutation.
    let edges: Vec<petgraph::prelude::EdgeIndex> = g.base.graph.edge_indices().collect();

    for e in edges {
        let sid = g.get_split(e);
        if sid > 0 {
            let sidu = sid as usize;
            let is_forbidden = forbidden.map_or(false, |bs| sidu < bs.len() && bs.contains(sidu));
            if !is_forbidden {
                if let Some(&theta) = split2angle.get(sidu) {
                    g.set_angle(e, theta);
                }
            }
        }
    }
}

/// Compute split direction for each split id (1-based) by “mid-angle” between the first/last
/// taxa (in cycle order) that lie on the part not containing cycle[1].
pub fn assign_angles_to_splits(
    ntaxa: usize,
    splits: &SplitsBlock,
    cycle: &[usize],
    total_angle: f64,
) -> Vec<f64> {
    // angles for taxa positions (1..=ntax), centered so taxon 1 points at 270 - total/2
    let mut taxa_angles = vec![0.0f64; ntaxa + 1];
    for tpos in 1..=ntaxa {
        taxa_angles[tpos] =
            total_angle * ((tpos - 1) as f64) / (ntaxa as f64) + (270.0 - 0.5 * total_angle);
    }

    // split angles (1-based)
    let mut split2angle = vec![0.0f64; splits.nsplits() + 1];

    for s in 1..=splits.nsplits() {
        let part = splits.split(s).part_not_containing(cycle[1]);
        let mut xp = 0usize;
        let mut xq = 0usize;
        for i in 2..=ntaxa {
            let t = cycle[i];
            if part.contains(t) {
                if xp == 0 {
                    xp = i;
                }
                xq = i;
            }
        }
        if xp == 0 {
            // degenerate; aim at first taxon
            split2angle[s] = modulo360(taxa_angles[1]);
        } else {
            split2angle[s] = modulo360(0.5 * (taxa_angles[xp] + taxa_angles[xq]));
        }
    }
    split2angle
}

/// Align the first edge incident to `taxon_id` to `target_deg` **then**
/// apply a global rotation of `extra_offset_deg`.
///
/// Returns `(align_delta, extra_offset_deg)` where:
/// - `align_delta` is `Some(deg)` if alignment was performed, else `None`
///   (e.g., if the taxon or leaf edge wasn’t found).
pub fn rotate_angles_align_then_offset(
    g: &mut PhyloSplitsGraph,
    taxon_id: usize,
    target_deg: f64,
    extra_offset_deg: f64,
) -> (Option<f64>, f64) {
    // Try to align this leaf’s incident edge
    let align_delta = rotate_angles_align_leaf(g, taxon_id, target_deg);

    // Always apply the extra global rotation afterward
    if extra_offset_deg != 0.0 {
        rotate_angles_in_place(g, extra_offset_deg);
    }
    (align_delta, extra_offset_deg)
}

/// Add `delta_deg` (in degrees) to every stored edge angle, modulo 360.
pub fn rotate_angles_in_place(g: &mut PhyloSplitsGraph, delta_deg: f64) {
    let edges: Vec<petgraph::prelude::EdgeIndex> = g.base.graph.edge_indices().collect();

    for e in edges {
        let a = g.get_angle(e);
        g.set_angle(e, modulo360(a + delta_deg));
    }
}

/// Rotate all edge angles so the *first* edge incident to `taxon_id` is aligned to `target_deg`.
/// Returns the delta that was applied (in degrees), or `None` if that leaf/edge couldn't be found.
pub fn rotate_angles_align_leaf(
    g: &mut PhyloSplitsGraph,
    taxon_id: usize,
    target_deg: f64,
) -> Option<f64> {
    let v = g
        .base
        .taxon2node()
        .expect("No taxon2node map")
        .get(&taxon_id)?;
    let e = g.base.graph.edges(*v).next()?.id(); // take first incident edge
    let cur = g.get_angle(e);
    let delta = modulo360(target_deg - cur);
    rotate_angles_in_place(g, delta);
    Some(delta)
}

pub fn assign_coordinates_to_nodes(
    use_weights: bool,
    g: &PhyloSplitsGraph,
    start_taxon_id: usize,
    root_split: i32,
) -> HashMap<NodeIndex, Pt> {
    let mut pts = HashMap::new();
    let Some(&v0) = g.base.taxon2node().expect("taxon map").get(&start_taxon_id) else {
        return pts;
    };
    pts.insert(v0, Pt(0.0, 0.0));

    let mut visited = FixedBitSet::with_capacity(g.base.graph.node_bound().index());
    let mut splits_in_path = FixedBitSet::with_capacity((g.max_split_id().max(0) as usize) + 2);

    dfs_coords(
        use_weights,
        g,
        v0,
        &mut visited,
        &mut splits_in_path,
        &mut pts,
        root_split,
    );
    pts
}

fn dfs_coords(
    use_weights: bool,
    g: &PhyloSplitsGraph,
    v: NodeIndex,
    visited: &mut FixedBitSet,
    splits_in_path: &mut FixedBitSet,
    pts: &mut HashMap<NodeIndex, Pt>,
    root_split: i32,
) {
    if visited.contains(v.index()) {
        return;
    }
    visited.insert(v.index());

    let nbrs: Vec<(EdgeIndex, NodeIndex)> = g
        .base
        .graph
        .edges(v)
        .map(|e| (e.id(), e.target()))
        .collect();

    for (e, w) in nbrs {
        let sid = g.get_split(e);
        if sid <= 0 {
            continue;
        }
        let s = sid as usize;
        if splits_in_path.len() <= s {
            splits_in_path.grow(s + 1);
        }
        if splits_in_path.contains(s) {
            continue;
        } // KEY guard
        splits_in_path.insert(s);

        let step = if use_weights {
            g.base.weight(e)
        } else if sid == root_split {
            0.1
        } else {
            1.0
        };
        let Pt(x, y) = *pts.get(&v).unwrap();
        let a = g.get_angle(e).to_radians();
        pts.insert(w, Pt(x + step * a.cos(), y + step * a.sin()));

        dfs_coords(use_weights, g, w, visited, splits_in_path, pts, root_split);
        splits_in_path.set(s, false); // backtrack
    }
}

/// Apply leaf labels from taxa labels (if exactly one taxon maps to the leaf), else keep node label
fn add_leaf_labels_from_taxa(g: &mut PhyloSplitsGraph, taxa: &[String]) {
    // give labels to leaf nodes with exactly one taxon id
    let mut updates: Vec<(NodeIndex, String)> = Vec::new();
    for v in g.base.graph.node_indices() {
        if g.base.graph.neighbors(v).count() == 1 {
            if let Some(list) = g.base.node2taxa().expect("No node2taxa map").get(&v) {
                if list.len() == 1 {
                    let t = list[0];
                    if t >= 1 && t <= taxa.len() {
                        updates.push((v, taxa[t - 1].clone()));
                    }
                }
            }
        }
    }
    for (v, lab) in updates {
        g.base.set_node_label(v, lab);
    }
}

/// Translate Java: EqualAngle.wrapSplit(...)
/// - `ntax`: number of taxa (1-based domain, i.e., taxa ids are 1..=ntax)
/// - `cycle`: 1-based array [ _, t1, t2, ..., t_ntax ] (index 0 ignored), already normalized
pub fn wrap_split(
    ntax: usize,
    splits: &SplitsBlock,
    s: usize,        // split id (1-based)
    cycle: &[usize], // 1-based cycle
    graph: &mut PhyloSplitsGraph,
) -> Result<()> {
    // Part of split s that does NOT contain taxon 1 (1-based membership)
    let part = splits.get(s).part_not_containing(1);

    // xp = first taxon in that part along the cycle
    // xq = last  taxon in that part along the cycle
    let (mut xp, mut xq) = (0usize, 0usize);
    for i in 1..=ntax {
        let t = cycle[i];
        if part.contains(t) {
            if xp == 0 {
                xp = t;
            }
            xq = t;
        }
    }
    if xp == 0 || xq == 0 {
        bail!(
            "wrap_split: split {} has empty non-1 part after cycle filtering",
            s
        );
    }

    // vp and vq: leaf nodes for taxa xp and xq
    let vp = graph
        .base
        .get_taxon_node(xp)
        .ok_or_else(|| anyhow!("wrap_split: no node for taxon {}", xp))?;
    let vq = graph
        .base
        .get_taxon_node(xq)
        .ok_or_else(|| anyhow!("wrap_split: no node for taxon {}", xq))?;

    let e_vp = graph
        .first_adjacent_edge(vp)
        .ok_or_else(|| anyhow!("wrap_split: taxon node vp ({:?}) has no incident edge", vp))?;

    let e_vq = graph
        .first_adjacent_edge(vq)
        .ok_or_else(|| anyhow!("wrap_split: taxon node vq ({:?}) has no incident edge", vq))?;

    let target_leaf_edge = e_vq; // Java: vq.getFirstAdjacentEdge()

    // Start from vp’s leaf edge and step to the inner neighbor
    let mut e = e_vp;
    let mut v = graph.base.get_opposite(vp, e);

    // Leaf edges we’ll copy at the current step (Java accumulates then clears each round)
    let mut leaf_edges: Vec<EdgeIndex> = Vec::with_capacity(ntax);
    leaf_edges.push(e);
    // "nodesVisited" to detect cycles (should not happen)
    let mut nodes_visited: HashSet<NodeIndex> = HashSet::new();

    // prevU = previous node on the new boundary path
    let mut prev_u: Option<NodeIndex> = None;

    // Boundary walk
    loop {
        debug!(
            "wrap_split: entering node {:?} via edge {:?} (nodes visits {})",
            v,
            e,
            nodes_visited.len()
        );
        if !nodes_visited.insert(v) {
            // already present
            bail!("wrap_split: node revisited during wrapping: {:?}", v);
        }

        // f0 is the edge by which we enter node v
        let f0 = e;

        // Gather leaf edges incident to v; detect if we reached target_leaf_edge
        // Also choose the "next" non-leaf boundary edge (not equal to f0).
        let mut reached_end = false;

        let mut next_e: Option<EdgeIndex> = None;

        // Collect adjacency first to avoid borrow issues while mutating later
        if let Some(mut f) = graph.next_adjacent_edge_cyclic(v, e) {
            while graph.is_leaf_edge(f) {
                leaf_edges.push(f);
                if f == target_leaf_edge {
                    break;
                }
                if f == f0 {
                    return Err(anyhow!(
                        "wrap_split: next adjacent edge cyclic after f0 ({:?}) is also f0; cycle detected at node {:?}",
                        f0,
                        v
                    ));
                }
                f = graph.next_adjacent_edge_cyclic(v, f).ok_or_else(|| {
                    anyhow!(
                        "wrap_split: no next leaf edge at node {:?} (after entering via {:?})",
                        v,
                        e
                    )
                })?;
            }

            if f == target_leaf_edge {
                next_e = None; // no next edge, we reached the target leaf edge
                reached_end = true;
            } else {
                next_e = Some(f);
            }
            debug!(
                "wrap_split: reached node {:?} edge {:?} (via {:?})",
                v, f, f0
            );
        };

        // Create new node on the NEW path
        let u = graph.base.new_node();

        // Edge from new node `u` to old node `v` (inserted "after f0" in Java — we can’t control embedding)
        let f_uv = graph.new_edge_after(u, v, f0)?;
        graph.set_split(f_uv, s as i32);
        graph.base.set_weight(f_uv, splits.get(s).weight().into());

        // If we already had a previous `u`, connect it to the current `u` carrying split/weight from `e`
        if let Some(prev) = prev_u {
            // Snapshot split/weight on `e` before any mutation
            let e_split = graph.get_split(e);
            let e_w = graph.base.weight(e);
            let f_uprev = graph.new_edge(u, prev)?;
            graph.set_split(f_uprev, e_split);
            graph.base.set_weight(f_uprev, e_w);
        }

        // Copy each leaf edge (v -- w) to (u -- w), copying split/weight; then delete old edge
        // Snapshot required info before mutation to avoid borrow conflicts
        let mut copies: Vec<(NodeIndex, i32, f64, EdgeIndex)> =
            Vec::with_capacity(leaf_edges.len());
        for f in leaf_edges.iter().copied() {
            let w = graph.base.get_opposite(v, f);
            let s_id = graph.get_split(f);
            let wgt = graph.base.weight(f);
            copies.push((w, s_id, wgt, f));
        }
        for (w, s_id, wgt, f_old) in copies {
            let f_new = graph.new_edge(u, w)?;
            graph.set_split(f_new, s_id);
            graph.base.set_weight(f_new, wgt);
            graph.remove_edge(f_old); // delete original leaf edge
        }
        leaf_edges.clear();
        // Advance along boundary or stop if we encountered the target leaf edge
        if reached_end {
            break;
        } else {
            let ne = next_e.ok_or_else(|| {
                anyhow!(
                    "wrap_split: no next boundary edge at node {:?} (after entering via {:?})",
                    v,
                    f0
                )
            })?;
            v = graph.base.get_opposite(v, ne);
            e = ne;
            prev_u = Some(u);
        }
    }

    Ok(())
}

/* ----------------------------- tiny utilities ------------------------------ */

fn first_member(bs: &fixedbitset::FixedBitSet) -> Option<usize> {
    // our FixedBitSet is 1-based usage; find first set from 1
    for i in 1..bs.len() {
        if bs.contains(i) {
            return Some(i);
        }
    }
    None
}

fn modulo360(a: f64) -> f64 {
    let mut x = a % 360.0;
    if x < 0.0 {
        x += 360.0;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use fixedbitset::FixedBitSet;
    use ndarray::arr2;

    use crate::{
        algorithms::equal_angle::{
            get_nontrivial_splits_ordered, init_graph_star, normalize_cycle_1based,
        },
        data::splits_blocks::SplitsBlock,
        ordering::ordering_huson2023::compute_order_huson_2023,
        phylo::phylo_splits_graph::PhyloSplitsGraph,
        utils::compute_least_squares_fit,
        weights::active_set_weights::{NNLSParams, compute_asplits},
    };

    #[test]
    fn smoke_10_1() {
        let d = arr2(&[
            [0.0, 5.0, 12.0, 7.0, 3.0, 9.0, 11.0, 6.0, 4.0, 10.0],
            [5.0, 0.0, 8.0, 2.0, 14.0, 5.0, 13.0, 7.0, 12.0, 1.0],
            [12.0, 8.0, 0.0, 4.0, 9.0, 3.0, 8.0, 2.0, 5.0, 6.0],
            [7.0, 2.0, 4.0, 0.0, 11.0, 7.0, 10.0, 4.0, 6.0, 9.0],
            [3.0, 14.0, 9.0, 11.0, 0.0, 8.0, 1.0, 13.0, 2.0, 7.0],
            [9.0, 5.0, 3.0, 7.0, 8.0, 0.0, 12.0, 5.0, 3.0, 4.0],
            [11.0, 13.0, 8.0, 10.0, 1.0, 12.0, 0.0, 6.0, 2.0, 8.0],
            [6.0, 7.0, 2.0, 4.0, 13.0, 5.0, 6.0, 0.0, 9.0, 7.0],
            [4.0, 12.0, 5.0, 6.0, 2.0, 3.0, 2.0, 9.0, 0.0, 5.0],
            [10.0, 1.0, 6.0, 9.0, 7.0, 4.0, 8.0, 7.0, 5.0, 0.0],
        ]);

        let cycle = compute_order_huson_2023(&d);
        let mut params = NNLSParams::default();
        let progress = None; // No progress tracking in this test
        let labels = vec![
            "t1".to_string(),
            "t2".to_string(),
            "t3".to_string(),
            "t4".to_string(),
            "t5".to_string(),
            "t6".to_string(),
            "t7".to_string(),
            "t8".to_string(),
            "t9".to_string(),
            "t10".to_string(),
        ];

        let splits = compute_asplits(&cycle, &d, &mut params, progress).expect("NNLS solve");
        let fit = compute_least_squares_fit(&d, &splits);
        let mut splits_blocks = SplitsBlock::new();
        splits_blocks.set_splits(splits);
        splits_blocks.set_fit(fit);
        splits_blocks.set_cycle(cycle, true).expect("set cycle");

        let mut graph = PhyloSplitsGraph::new();
        // then:
        let mut used_splits = FixedBitSet::with_capacity(splits_blocks.nsplits() + 1);

        let cycle_exp = vec![0, 1, 5, 7, 9, 3, 8, 4, 2, 10, 6];
        let c = splits_blocks
            .cycle()
            .ok_or_else(|| anyhow::anyhow!("SplitsBlock has no cycle"))
            .expect("get cycle");
        let norm_cycle = normalize_cycle_1based(&c).expect("normalize cycle");
        assert_eq!(norm_cycle, cycle_exp);
        init_graph_star(&labels, &splits_blocks, &norm_cycle, &mut graph).expect("init graph star");
        assert!(graph.base.graph.node_count() == 11);
        assert!(graph.base.graph.edge_count() == 10);

        let ordered = get_nontrivial_splits_ordered(&splits_blocks);
        assert_eq!(ordered, vec![2, 3, 11, 9, 16, 4, 8, 15, 7, 14, 19, 21]);
        let ntax = labels.len();
        for &sid in &ordered {
            if is_circular_by_cycle(&splits_blocks, sid, &norm_cycle) {
                // Wrap split sid around the cycle
                wrap_split(ntax, &splits_blocks, sid, &norm_cycle, &mut graph).expect("wrap split");
                // Placeholder: log and mark used. We don’t modify graph topology here yet.
                used_splits.grow(splits_blocks.nsplits() + 1);
                used_splits.insert(sid);
            }
        }
        assert!(graph.base.graph.node_count() == 41);
        assert!(graph.base.graph.edge_count() == 58);

        let exp_angles = vec![
            0.0, 270.0, 288.0, 324.0, 18.0, 54.0, 126.0, 144.0, 162.0, 180.0, 162.0, 234.0, 198.0,
            234.0, 252.0, 270.0, 288.0, 270.0, 306.0, 324.0, 342.0, 0.0, 18.0,
        ];
        let split2angle = assign_angles_to_splits(ntax, &splits_blocks, &norm_cycle, 360.);
        assert_eq!(split2angle, exp_angles);
        // equal_angle_apply(
        //     EqualAngleOpts::default(),
        //     &labels,
        //     &splits_blocks,
        //     &mut graph,
        //     None,
        //     &mut used_splits,
        // ).expect("EqualAngle apply");
        // println!("{}", graph.base.graph.node_count());
        // println!("{}", graph.base.graph.edge_count());
        // assert!(graph.base.graph.node_count() == 42);
        // assert!(graph.base.graph.edge_count() == 60);
    }
}

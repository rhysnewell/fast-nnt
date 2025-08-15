use fixedbitset::FixedBitSet;
use petgraph::visit::{EdgeRef, NodeIndexable};

use crate::splits::asplit::ASplit;
use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;

/// Small trait so we can accept your SplitsBlock-like container.
/// If your project has a concrete SplitsBlock type, just implement this for it.
pub trait SplitsProvider {
    /// number of splits
    fn nsplits(&self) -> usize;
    /// 1-based get: split id in [1..=nsplits]
    fn split(&self, id: usize) -> &ASplit;
    /// optional: raw cycle [0, t1, t2, ..., tn]
    fn cycle(&self) -> Option<&[usize]> {
        None
    }
}

/// Wrap to [0,360)
#[inline]
pub fn modulo_360(mut ang: f64) -> f64 {
    ang = ang % 360.0;
    if ang < 0.0 {
        ang += 360.0;
    }
    ang
}

/// Normalize cycle so that cycle[1] == 1 (keeps 1-based contract; cycle[0] is ignored).
pub fn normalize_cycle(cycle: &[usize]) -> Vec<usize> {
    assert!(cycle.len() >= 2, "cycle must be [0, t1, ..., tn]");
    let n = cycle.len() - 1;
    let mut res = vec![0usize; n + 1];
    // find i with cycle[i] == 1
    let mut i = 1;
    while i <= n && cycle[i] != 1 {
        i += 1;
    }
    // rotate
    let mut j = 1;
    let mut k = i;
    while k <= n {
        res[j] = cycle[k];
        j += 1;
        k += 1;
    }
    k = 1;
    while j <= n {
        res[j] = cycle[k];
        j += 1;
        k += 1;
    }
    res
}

/// Compute equal-angle directions for all splits (degrees).
/// - `ntaxa`: number of taxa (size of cycle, excluding the 0 sentinel)
/// - `splits`: provides ASplit for ids 1..=nsplits
/// - `cycle`: 1-based cycle [0, t1..tn]
/// - `total_angle`: arc length to lay the taxa on (typical 360.0)
///
/// Returns a vector `split2angle` indexed by split id (len = nsplits + 1).
pub fn assign_angles_to_splits<S: SplitsProvider>(
    ntaxa: usize,
    splits: &S,
    cycle: &[usize],
    total_angle: f64,
) -> Vec<f64> {
    assert_eq!(
        cycle.len(),
        ntaxa + 1,
        "cycle must be [0, t1..tn] with ntaxa entries"
    );

    // Taxon angular positions on the circle (1-based indices)
    // angles[taxon-position-in-cycle] = angle
    // Java: angles[t] = totalAngle * (t-1)/ntaxa + 270 - 0.5*totalAngle
    let mut taxa_angles = vec![0.0f64; ntaxa + 1];
    for t in 1..=ntaxa {
        taxa_angles[t] = total_angle * ((t - 1) as f64) / (ntaxa as f64)
            + 270.0
            - 0.5 * total_angle;
    }

    // Assign each split the midpoint angle between the first and last taxon
    // of the part not containing cycle[1] (i.e. the "opposite" block).
    let ns = splits.nsplits();
    let mut split2angle = vec![0.0f64; ns + 1];

    // `anchor` is the taxon at cycle[1] after normalization
    let anchor = cycle[1];

    for s in 1..=ns {
        let sp = splits.split(s);

        // The block not containing anchor taxon (1-based taxa ids in ASplit)
        let part_not = sp.part_not_containing(anchor);

        // xp, xq are positions (indices in the cycle array) of the first/last
        // taxon belonging to the part_not, scanning i = 2..=ntaxa (Java parity)
        let mut xp = 0usize;
        let mut xq = 0usize;
        for i in 2..=ntaxa {
            let tax = cycle[i];
            if part_not.contains(tax) {
                if xp == 0 {
                    xp = i;
                }
                xq = i;
            }
        }
        if xp == 0 || xq == 0 {
            // Split might be trivial or degenerate w.r.t cycle; keep 0.0 angle
            continue;
        }

        let ang = 0.5 * (taxa_angles[xp] + taxa_angles[xq]);
        split2angle[s] = modulo_360(ang);
    }

    split2angle
}

/// Push split-angles to edges (skip forbidden split ids).
/// - `forbidden_splits`: bitset where index = split id; true means "don't change"
pub fn assign_angles_to_edges<S: SplitsProvider>(
    ntaxa: usize,
    splits: &S,
    cycle: &[usize],
    graph: &mut PhyloSplitsGraph,
    forbidden_splits: Option<&FixedBitSet>,
    total_angle: f64,
) {
    let split2angle = assign_angles_to_splits(ntaxa, splits, cycle, total_angle);

    // Collect edges first to avoid borrow overlap
    let edges: Vec<_> = graph.base.graph.edge_indices().collect();
    for e in edges {
        let sid = graph.get_split(e);
        if sid <= 0 {
            continue; // skip temporary/invalid split ids like -1 or 0
        }
        let sid_u = sid as usize;
        if let Some(fs) = forbidden_splits {
            if sid_u < fs.len() && fs.contains(sid_u) {
                continue;
            }
        }
        if sid_u < split2angle.len() {
            graph.set_angle(e, split2angle[sid_u]);
        }
    }
}

/* ---------------- Optional: simple coordinate assignment ---------------- */

/// Simple 2D point (x, y).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pt(pub f64, pub f64);

#[inline]
fn translate_by_angle(p: Pt, angle_deg: f64, dist: f64) -> Pt {
    let th = angle_deg.to_radians();
    Pt(p.0 + dist * th.cos(), p.1 + dist * th.sin())
}

/// Assign coordinates from angles & (optionally) weights, like the Java helper.
/// - Places `start_taxon_id` at (0,0), then DFS-extends using edge angle and
///   step length = weight (or 1.0 if `use_weights=false`, except root split scaled).
///
/// This is a *very* simple layout helper (not a full equal-angle embedder).
pub fn assign_coordinates_to_nodes(
    use_weights: bool,
    graph: &PhyloSplitsGraph,
    start_taxon_id: usize,
    root_split: Option<i32>,
) -> std::collections::HashMap<NodeIndex, Pt> {
    use petgraph::stable_graph::NodeIndex;
    use std::collections::HashMap as Map;

    let mut coords: Map<NodeIndex, Pt> = Map::new();
    let Some(start) = graph.base.get_taxon_node(start_taxon_id) else {
        return coords;
    };
    coords.insert(start, Pt(0.0, 0.0));

    // track edges (by split id) used on current path
    let mut used_splits = FixedBitSet::with_capacity((graph.max_split_id().max(0) as usize) + 1);
    let mut seen_nodes = FixedBitSet::with_capacity(graph.base.graph.node_bound());

    fn dfs(
        g: &PhyloSplitsGraph,
        v: NodeIndex,
        use_weights: bool,
        root_split: Option<i32>,
        used: &mut FixedBitSet,
        seen: &mut FixedBitSet,
        coords: &mut std::collections::HashMap<NodeIndex, Pt>,
    ) {
        if seen.contains(v.index()) {
            return;
        }
        seen.insert(v.index());
        let pv = *coords.get(&v).unwrap_or(&Pt(0.0, 0.0));
        for er in g.base.graph.edges(v) {
            let e = er.id();
            let sid = g.get_split(e);
            let idx = sid.max(0) as usize;
            if !used.contains(idx) {
                let w = g
                    .base
                    .graph
                    .edge_endpoints(e)
                    .map(|(a, b)| if a == v { b } else { a })
                    .unwrap();
                let angle = g.get_angle(e);
                let step = if use_weights {
                    g.base.weight(e)
                } else if Some(sid) == root_split {
                    0.1
                } else {
                    1.0
                };
                let pw = translate_by_angle(pv, angle, step);
                coords.insert(w, pw);
                if used.len() <= idx {
                    used.grow(idx + 1);
                }
                used.insert(idx);
                dfs(g, w, use_weights, root_split, used, seen, coords);
                used.set(idx, false);
            }
        }
    }

    dfs(
        graph,
        start,
        use_weights,
        root_split,
        &mut used_splits,
        &mut seen_nodes,
        &mut coords,
    );
    coords
}

/* ---------------- Utilities for tests or pipelines ---------------- */

/// Order non-trivial split ids by increasing size of the block containing taxon `anchor_taxon`.
pub fn get_non_trivial_splits_ordered<S: SplitsProvider>(
    splits: &S,
    anchor_taxon: usize,
) -> Vec<usize> {
    use std::cmp::Ordering;
    let mut ids: Vec<(usize, usize)> = Vec::new(); // (size, id)
    for s in 1..=splits.nsplits() {
        let sp = splits.split(s);
        let size = sp.part_containing(anchor_taxon).count_ones_excluding_zero();
        if size > 1 && size < sp.ntax() {
            ids.push((size, s));
        }
    }
    ids.sort_by(|a, b| {
        let o = a.0.cmp(&b.0);
        if o == Ordering::Equal {
            a.1.cmp(&b.1)
        } else {
            o
        }
    });
    ids.into_iter().map(|(_, s)| s).collect()
}

/* ---------------- ASplit helpers we rely on ---------------- */

use petgraph::stable_graph::NodeIndex;

trait PartOps {
    fn contains(&self, taxon: usize) -> bool;
    fn count_ones_excluding_zero(&self) -> usize;
}

impl PartOps for FixedBitSet {
    #[inline]
    fn contains(&self, taxon: usize) -> bool {
        taxon > 0 && taxon < self.len() && self.contains(taxon)
    }
    #[inline]
    fn count_ones_excluding_zero(&self) -> usize {
        // FixedBitSet is 0-based; we ignore bit 0 as sentinel
        let mut c = 0usize;
        // iterate over set bits
        let mut idx = self.ones().peekable();
        while let Some(i) = idx.next() {
            if i != 0 {
                c += 1;
            }
        }
        c
    }
}


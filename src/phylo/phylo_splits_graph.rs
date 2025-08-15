use fixedbitset::FixedBitSet;
use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, NodeIndexable};
use std::collections::HashMap;

use crate::phylo::phylo_graph::{DEFAULT_WEIGHT, PhyloGraph};

#[inline]
fn fb_ensure_len(bs: &mut fixedbitset::FixedBitSet, needed: usize) {
    if bs.len() < needed {
        bs.grow(needed);
    }
}

#[inline]
fn fb_insert(bs: &mut fixedbitset::FixedBitSet, idx: usize) {
    fb_ensure_len(bs, idx + 1);
    bs.insert(idx);
}

#[inline]
fn fb_set(bs: &mut fixedbitset::FixedBitSet, idx: usize, val: bool) {
    fb_ensure_len(bs, idx + 1);
    bs.set(idx, val);
}

/// Splits graph: extends PhyloGraph with per-edge split ids, angles, and a 1-based taxon→cycle map
#[derive(Debug, Clone, Default)]
pub struct PhyloSplitsGraph {
    pub base: PhyloGraph,

    // Edge attribute maps (lazy-allocated, like Java's EdgeIntArray/EdgeDoubleArray)
    edge_split_map: Option<HashMap<EdgeIndex, i32>>,
    edge_angle_map: Option<HashMap<EdgeIndex, f64>>,

    // 1-based taxon id → cycle position (1..=ntax); index 0 is unused
    taxon_cycle_map: Option<Vec<usize>>,
}

impl PhyloSplitsGraph {
    /* ---------------- construction / housekeeping ---------------- */

    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.base.clear();
        self.edge_split_map = None;
        self.edge_angle_map = None;
        self.taxon_cycle_map = None;
    }

    pub fn name(&self) -> Option<&str> {
        self.base.name()
    }
    pub fn set_name<S: Into<String>>(&mut self, s: S) {
        self.base.set_name(s);
    }

    /* ---------------- edge split-id / angle maps ---------------- */

    fn splits_mut(&mut self) -> &mut HashMap<EdgeIndex, i32> {
        if self.edge_split_map.is_none() {
            self.edge_split_map = Some(HashMap::new());
        }
        self.edge_split_map.as_mut().unwrap()
    }
    fn angles_mut(&mut self) -> &mut HashMap<EdgeIndex, f64> {
        if self.edge_angle_map.is_none() {
            self.edge_angle_map = Some(HashMap::new());
        }
        self.edge_angle_map.as_mut().unwrap()
    }

    pub fn set_split(&mut self, e: EdgeIndex, id: i32) {
        self.splits_mut().insert(e, id);
    }
    pub fn get_split(&self, e: EdgeIndex) -> i32 {
        self.edge_split_map
            .as_ref()
            .and_then(|m| m.get(&e).copied())
            .unwrap_or(0)
    }

    pub fn set_angle(&mut self, e: EdgeIndex, val: f64) {
        self.angles_mut().insert(e, val);
    }
    pub fn get_angle(&self, e: EdgeIndex) -> f64 {
        self.edge_angle_map
            .as_ref()
            .and_then(|m| m.get(&e).copied())
            .unwrap_or(0.0)
    }

    pub fn get_split_ids(&self) -> Vec<i32> {
        let mut ids: Vec<i32> = self
            .base
            .graph
            .edge_indices()
            .filter_map(|e| {
                let id = self.get_split(e);
                (id != 0).then_some(id)
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /* ---------------- copy / clone ---------------- */

    /// Deep-copy from another PhyloSplitsGraph. Returns (old→new node map, old→new edge map).
    pub fn copy_from(
        &mut self,
        src: &PhyloSplitsGraph,
    ) -> (HashMap<NodeIndex, NodeIndex>, HashMap<EdgeIndex, EdgeIndex>) {
        self.clear();
        self.set_name(src.name().unwrap_or_default());

        // 1) Copy base graph while building maps
        let mut old2new_node = HashMap::default();
        for v in src.base.graph.node_indices() {
            let w = if let Some(lbl) = src.base.node_label(v) {
                self.base.new_node_with_label(lbl.to_string())
            } else {
                self.base.new_node()
            };
            // copy taxa for v to w
            if let Some(n2t) = src.base.node2taxa() {
                if let Some(tlist) = n2t.get(&v) {
                    for &t in tlist {
                        self.base.add_taxon(w, t);
                    }
                }
            }
            old2new_node.insert(v, w);
        }

        let mut old2new_edge = HashMap::default();
        for e in src.base.graph.edge_indices() {
            let (u_old, v_old) = src.base.graph.edge_endpoints(e).unwrap();
            let u = old2new_node[&u_old];
            let v = old2new_node[&v_old];
            let f = if let Some(l) = src.base.edge_label(e) {
                self.base.new_edge_with_label(u, v, l.to_string()).unwrap()
            } else {
                self.base.new_edge(u, v).unwrap()
            };
            self.base.set_weight(f, src.base.weight(e));
            if let Some(conf) = src
                .base
                .has_edge_confidences()
                .then(|| src.base.confidence(e))
            {
                self.base.set_confidence(f, conf);
            }
            if let Some(p) = src
                .base
                .has_edge_probabilities()
                .then(|| src.base.probability(e))
            {
                self.base.set_probability(f, p);
            }
            // Copy split/angle
            let sid = src.get_split(e);
            if sid != 0 {
                self.set_split(f, sid);
            }
            let ang = src.get_angle(e);
            if ang != 0.0 {
                self.set_angle(f, ang);
            }
            old2new_edge.insert(e, f);
        }

        // Copy taxon->cycle map
        if let Some(t2c) = &src.taxon_cycle_map {
            self.taxon_cycle_map = Some(t2c.clone());
        }

        (old2new_node, old2new_edge)
    }

    pub fn clone_graph(&self) -> Self {
        let mut out = PhyloSplitsGraph::new();
        out.copy_from(self);
        out
    }

    /* ---------------- split removal / separators ---------------- */

    /// Remove an entire split by contracting all edges tagged with that split id.
    /// This follows the Java implementation closely.
    pub fn remove_split(&mut self, split_id: i32) {
        let one = self.base.get_taxon_node(1);
        if one.is_none() {
            return;
        }
        let start = one.unwrap();

        // 1) gather separators (v, e) along any path from 'start' where edge has split_id
        let mut separators: Vec<(NodeIndex, EdgeIndex)> = Vec::new();
        let mut seen: FixedBitSet = FixedBitSet::with_capacity(self.base.graph.node_bound());
        self.get_all_separators(split_id, start, None, &mut seen, &mut separators);

        if separators.is_empty() {
            return;
        }

        // 2) opposites set for quick membership tests
        let mut opposites: FixedBitSet = FixedBitSet::with_capacity(self.base.graph.node_bound());
        for &(v, e) in &separators {
            if let Some(w) = self.opposite_of(v, e) {
                opposites.insert(w.index());
            }
        }

        // 3) for each separator (v --e-- w), rewire w's incident edges to v where allowed, then merge labels/taxa, delete w
        for (v, e) in separators {
            let w = match self.opposite_of(v, e) {
                Some(node) => node,
                None => continue,
            };

            // collect current incident edges of w (so we can mutate the graph safely)
            let mut w_incident: Vec<EdgeIndex> =
                self.base.graph.edges(w).map(|er| er.id()).collect();

            for f in w_incident.iter().copied() {
                if f == e {
                    continue;
                }
                let maybe_u = self.opposite_of(w, f);
                if maybe_u.is_none() {
                    continue;
                }
                let u = maybe_u.unwrap();

                if u == v {
                    continue;
                }
                if opposites.contains(u.index()) {
                    continue;
                }

                // create new edge g: (u, v)
                if let Ok(g) = self.base.new_edge(u, v) {
                    // copy split, weight, angle from f
                    let sid = self.get_split(f);
                    if sid != 0 {
                        self.set_split(g, sid);
                    }
                    let wgt = self.base.weight(f);
                    if wgt != DEFAULT_WEIGHT {
                        self.base.set_weight(g, wgt);
                    }
                    let ang = self.get_angle(f);
                    if ang != 0.0 {
                        self.set_angle(g, ang);
                    }
                    // copy edge label if present
                    if let Some(lbl) = self.base.edge_label(f) {
                        self.base.set_edge_label(g, lbl.to_string());
                    }
                }
            }

            // merge labels w -> v
            let vlab = self.base.node_label(v).map(|s| s.to_string());
            let wlab = self.base.node_label(w).map(|s| s.to_string());
            match (vlab, wlab) {
                (None, Some(wl)) if !wl.is_empty() => self.base.set_node_label(v, wl),
                (Some(vl), Some(wl)) if !wl.is_empty() => {
                    let mut merged = vl;
                    if !merged.is_empty() {
                        merged.push_str(", ");
                    }
                    merged.push_str(&wl);
                    self.base.set_node_label(v, merged);
                }
                _ => {}
            }

            // merge labels w -> v (unchanged)
            let vlab = self.base.node_label(v).map(|s| s.to_string());
            let wlab = self.base.node_label(w).map(|s| s.to_string());
            match (vlab, wlab) {
                (None, Some(wl)) if !wl.is_empty() => self.base.set_node_label(v, wl),
                (Some(vl), Some(wl)) if !wl.is_empty() => {
                    let mut merged = vl;
                    if !merged.is_empty() {
                        merged.push_str(", ");
                    }
                    merged.push_str(&wl);
                    self.base.set_node_label(v, merged);
                }
                _ => {}
            }

            // --- FIX: copy taxa first to end the immutable borrow before mutating
            let taxa_to_move: Vec<usize> = self
                .base
                .node2taxa()
                .and_then(|n2t| n2t.get(&w).cloned())
                .unwrap_or_default();

            for t in taxa_to_move {
                self.base.add_taxon(v, t);
            }

            self.base.clear_taxa_for_node(w);

            // delete w
            self.base.remove_node_and_cleanup(w);
        }
    }

    /// DFS: collect all (node, edge) where edge has the given split id.
    fn get_all_separators(
        &self,
        split_id: i32,
        v: NodeIndex,
        parent_edge: Option<EdgeIndex>,
        seen: &mut FixedBitSet,
        out: &mut Vec<(NodeIndex, EdgeIndex)>,
    ) {
        if seen.contains(v.index()) {
            return;
        }
        seen.insert(v.index());
        for er in self.base.graph.edges(v) {
            let e = er.id();
            if Some(e) == parent_edge {
                continue;
            }
            if self.get_split(e) == split_id {
                out.push((v, e));
            } else if let Some(w) = self.opposite_of(v, e) {
                self.get_all_separators(split_id, w, Some(e), seen, out);
            }
        }
    }

    /// Find one (node, edge) separator for split_id on a path from v
    pub fn get_separator(
        &self,
        split_id: i32,
        v: NodeIndex,
        parent_edge: Option<EdgeIndex>,
        seen: &mut FixedBitSet,
    ) -> Option<(NodeIndex, EdgeIndex)> {
        if seen.contains(v.index()) {
            return None;
        }
        seen.insert(v.index());
        for er in self.base.graph.edges(v) {
            let e = er.id();
            if Some(e) == parent_edge {
                continue;
            }
            if self.get_split(e) == split_id {
                return Some((v, e));
            } else if let Some(w) = self.opposite_of(v, e) {
                if let Some(pair) = self.get_separator(split_id, w, Some(e), seen) {
                    return Some(pair);
                }
            }
        }
        None
    }

    #[inline]
    fn opposite_of(&self, v: NodeIndex, e: EdgeIndex) -> Option<NodeIndex> {
        let (a, b) = self.base.graph.edge_endpoints(e)?;
        if a == v {
            Some(b)
        } else if b == v {
            Some(a)
        } else {
            None
        }
    }

    /* ---------------- node sequence labeling ----------------
       Rough port of labelNodesBySequences / labelNodesBySequencesRec:
       - split2chars: map split id -> bitset of characters that flip at that split
       - first_chars: 1-based char array ['\0','0'/'1', ...]
       Returns a map: node -> "0101...".
    ---------------------------------------------------------------- */

    pub fn label_nodes_by_sequences(
        &self,
        split2chars: &HashMap<i32, FixedBitSet>,
        first_chars: &[u8], // length m+1, index 0 unused; values b'0' or b'1'
    ) -> HashMap<NodeIndex, String> {
        let mut labels: HashMap<NodeIndex, String> = HashMap::default();
        let start = match self.base.get_taxon_node(1) {
            Some(v) => v,
            None => return labels,
        };
        let cap = (self.max_split_id().max(0) as usize) + 1;
        let mut used_splits = FixedBitSet::with_capacity(cap);

        self.label_nodes_by_sequences_rec(
            start,
            &mut used_splits,
            split2chars,
            first_chars,
            &mut labels,
        );
        labels
    }

    fn label_nodes_by_sequences_rec(
        &self,
        v: NodeIndex,
        used: &mut FixedBitSet, // splits used on the path
        split2chars: &HashMap<i32, FixedBitSet>,
        first_chars: &[u8],
        out: &mut HashMap<NodeIndex, String>,
    ) {
        if out.contains_key(&v) {
            return;
        }
        // flips := OR of all character-sets for splits in 'used'
        let mut flips = FixedBitSet::with_capacity(first_chars.len());
        // iterate split ids present in 'used' (we don't know max; iterate keys)
        if let Some(sm) = &self.edge_split_map {
            for (&sid, _) in sm.iter() {
                if used.contains(sid.index()) {
                    if let Some(bits) = split2chars.get(&(sid.index() as i32)) {
                        let mut tmp = flips.clone();
                        tmp.union_with(bits);
                        flips = tmp;
                    }
                }
            }
        }

        // build label by flipping chars where needed (1..m)
        let mut sb = String::with_capacity(first_chars.len().saturating_sub(1));
        for c in 1..first_chars.len() {
            let bit = flips.contains(c);
            let first_is_one = first_chars[c] == b'1';
            // If flip==first_is_one then 0 else 1 (match Java logic)
            let ch = if bit == first_is_one { '0' } else { '1' };
            sb.push(ch);
        }
        out.insert(v, sb);

        // DFS to neighbors
        for er in self.base.graph.edges(v) {
            let e = er.id();
            let sid = self.get_split(e);
            if sid >= 0 {
                let idx = sid as usize;
                if !used.contains(idx) {
                    // GROW before insert
                    fb_insert(used, idx);
                    if let Some(w) = self.opposite_of(v, e) {
                        self.label_nodes_by_sequences_rec(w, used, split2chars, first_chars, out);
                    }
                    // GROW before set(false) (might be no-op if already grown)
                    fb_set(used, idx, false);
                }
            }
        }
    }

    /* ---------------- cycle mapping ---------------- */

    /// 1-based get: taxon id -> cycle index
    pub fn get_taxon2_cycle(&self, tax_id: usize) -> i32 {
        if let Some(vec) = &self.taxon_cycle_map {
            if tax_id > 0 && tax_id <= vec.len() {
                return vec[tax_id - 1] as i32;
            }
        }
        // mirror Java: return -1 if absent
        -1
    }

    /// Set (1-based) taxon → cycle index
    pub fn set_taxon2_cycle(&mut self, tax_id: usize, cycle_index: usize) {
        if self.taxon_cycle_map.is_none() {
            self.taxon_cycle_map = Some(Vec::new());
        }
        let v = self.taxon_cycle_map.as_mut().unwrap();
        if tax_id > v.len() {
            v.resize(tax_id, 0);
        }
        v[tax_id - 1] = cycle_index;
    }

    /// Build the cycle array [0, t1, t2, ..., tn] (1-based taxa)
    pub fn get_cycle(&self) -> Vec<usize> {
        let ntax = self.base.number_of_taxa();
        let mut cycle = vec![0usize; ntax + 1];
        for t in 1..=ntax {
            let idx = self.get_taxon2_cycle(t);
            if idx > 0 && (idx as usize) < cycle.len() {
                cycle[idx as usize] = t;
            }
        }
        cycle
    }

    /* ---------------- counters ---------------- */

    pub fn count_splits(&self) -> usize {
        use std::collections::BTreeSet;
        let mut seen: BTreeSet<i32> = BTreeSet::new();
        for e in self.base.graph.edge_indices() {
            let id = self.get_split(e);
            if id != 0 {
                seen.insert(id);
            }
        }
        seen.len()
    }

    pub fn max_split_id(&self) -> i32 {
        let mut max_id = 0;
        for e in self.base.graph.edge_indices() {
            let id = self.get_split(e);
            if id > max_id {
                max_id = id;
            }
        }
        max_id
    }

    pub fn count_nodes(&self) -> usize {
        self.base.graph.node_count()
    }

    pub fn count_edges(&self) -> usize {
        self.base.graph.edge_count()
    }

    pub fn network_type(&self) -> &str {
        "phylo-splits"
    }
}

#[cfg(test)]
mod phylo_splits_graph_tests {
    use super::*;
    use fixedbitset::FixedBitSet;
    use petgraph::stable_graph::{EdgeIndex, NodeIndex};
    use std::collections::HashMap;

    /// Convenience: build a tiny 3-node graph A—B—C where:
    /// - edge AB has split=99
    /// - edge BC has split=5
    /// - taxon 1 is on A; taxon 2 on B; taxon 3 on C
    /// - labels "A","B","C"
    fn make_abc() -> (
        PhyloSplitsGraph,
        NodeIndex,
        NodeIndex,
        NodeIndex,
        EdgeIndex,
        EdgeIndex,
    ) {
        let mut g = PhyloSplitsGraph::new();

        let a = g.base.new_node_with_label("A");
        let b = g.base.new_node_with_label("B");
        let c = g.base.new_node_with_label("C");

        // taxa
        g.base.add_taxon(a, 1);
        g.base.add_taxon(b, 2);
        g.base.add_taxon(c, 3);

        let e_ab = g.base.new_edge(a, b).unwrap();
        g.set_split(e_ab, 99);
        g.set_angle(e_ab, 0.1);
        g.base.set_weight(e_ab, 2.0);

        let e_bc = g.base.new_edge(b, c).unwrap();
        g.set_split(e_bc, 5);
        g.set_angle(e_bc, 0.2);
        g.base.set_weight(e_bc, 3.0);

        (g, a, b, c, e_ab, e_bc)
    }

    #[test]
    fn split_and_angle_maps_work() {
        let (g, _a, _b, _c, e_ab, e_bc) = make_abc();
        assert_eq!(g.get_split(e_ab), 99);
        assert_eq!(g.get_split(e_bc), 5);
        assert_eq!(g.get_angle(e_ab), 0.1);
        assert_eq!(g.get_angle(e_bc), 0.2);

        // id set should be unique & sorted
        let ids = g.get_split_ids();
        assert_eq!(ids, vec![5, 99]);
        assert_eq!(g.max_split_id(), 99);
        assert_eq!(g.count_splits(), 2);
    }

    #[test]
    fn remove_split_rewires_and_moves_taxa() {
        let (mut g, a, b, c, e_ab, e_bc) = make_abc();

        // preconditions
        assert!(g.base.graph.contains_node(a));
        assert!(g.base.graph.contains_node(b));
        assert!(g.base.graph.contains_node(c));
        assert!(g.base.graph.edge_endpoints(e_ab).is_some());
        assert!(g.base.graph.edge_endpoints(e_bc).is_some());
        assert_eq!(g.base.get_taxon_node(2), Some(b));
        assert_eq!(g.base.node_label(a), Some("A"));
        assert_eq!(g.base.node_label(b), Some("B"));

        // remove split 99 i.e., contract around AB
        g.remove_split(99);

        // B should be deleted
        assert!(g.base.graph.node_weight(b).is_none());

        // There should now be a direct edge between A and C, inheriting split from BC (5)
        // Find any edge connecting A and C
        let mut ac_found = false;
        for e in g.base.graph.edge_indices() {
            if let Some((u, v)) = g.base.graph.edge_endpoints(e) {
                if (u == a && v == c) || (u == c && v == a) {
                    ac_found = true;
                    // split id of the new edge should be 5 (copied from BC)
                    assert_eq!(g.get_split(e), 5);
                    // weight/angle copied from BC as well (if your impl copies them)
                    // Can't guarantee exact edge identity, so we just assert presence
                }
            }
        }
        assert!(ac_found, "Expected A—C edge after splitting");

        // Taxon 2 should have moved from B to A
        assert_eq!(g.base.get_taxon_node(2), Some(a));

        // Labels should have merged "A, B" (order may vary depending on your merge logic)
        let lbl = g.base.node_label(a).unwrap().to_string();
        assert!(lbl.contains('A'));
        assert!(lbl.contains('B'));
    }

    #[test]
    fn cycle_mapping_roundtrip() {
        let (mut g, _a, _b, _c, _e_ab, _e_bc) = make_abc();
        // set cycle positions: 1→1, 2→3, 3→2
        g.set_taxon2_cycle(1, 1);
        g.set_taxon2_cycle(2, 3);
        g.set_taxon2_cycle(3, 2);

        assert_eq!(g.get_taxon2_cycle(1), 1);
        assert_eq!(g.get_taxon2_cycle(2), 3);
        assert_eq!(g.get_taxon2_cycle(3), 2);

        // build cycle [0, t1, t2, t3]
        let cycle = g.get_cycle();
        // At index 1 we expect taxon 1; at 2 -> taxon 3; at 3 -> taxon 2
        assert_eq!(cycle.len(), 4);
        assert_eq!(cycle[1], 1);
        assert_eq!(cycle[2], 3);
        assert_eq!(cycle[3], 2);
    }

    #[test]
    fn label_nodes_by_sequences_basic() {
        let (g, a, b, c, e_ab, e_bc) = make_abc();

        // Define a simple mapping: split 99 flips char 1; split 5 flips char 2
        // first_chars is 1-based: index 0 unused; positions 1..m are '0'/'1'
        let mut split2chars: HashMap<i32, FixedBitSet> = HashMap::default();
        let mut bits99 = FixedBitSet::with_capacity(4);
        bits99.insert(1);
        let mut bits5 = FixedBitSet::with_capacity(4);
        bits5.insert(2);
        split2chars.insert(99, bits99);
        split2chars.insert(5, bits5);

        let first = vec![0u8, b'0', b'0', b'0']; // "000" baseline

        let labels = g.label_nodes_by_sequences(&split2chars, &first);

        // Sanity: we computed a label for each reachable node
        assert!(labels.contains_key(&a));
        assert!(labels.contains_key(&b));
        assert!(labels.contains_key(&c));

        // Each label length is m = 3
        for s in labels.values() {
            assert_eq!(s.len(), 3);
            assert!(s.chars().all(|ch| ch == '0' || ch == '1'));
        }
    }

    #[test]
    fn deep_copy_preserves_structure_and_annotations() {
        let (mut g, a, b, c, e_ab, e_bc) = make_abc();
        g.set_taxon2_cycle(1, 1);
        g.set_taxon2_cycle(2, 2);
        g.set_taxon2_cycle(3, 3);

        let cloned = g.clone_graph();

        // base graph node count should match (3 nodes)
        let g_nodes = g.base.graph.node_indices().count();
        let c_nodes = cloned.base.graph.node_indices().count();
        assert_eq!(g_nodes, c_nodes);

        // splits present
        let mut ids = g.get_split_ids();
        ids.sort_unstable();
        let mut ids2 = cloned.get_split_ids();
        ids2.sort_unstable();
        assert_eq!(ids, ids2);

        // angles copied
        let mut has_angles = false;
        for e in g.base.graph.edge_indices() {
            if g.get_angle(e) != 0.0 {
                has_angles = true;
            }
        }
        if has_angles {
            // sampling: ensure cloned has non-zero angles on some edge
            let cloned_has_nonzero = cloned
                .base
                .graph
                .edge_indices()
                .any(|e| cloned.get_angle(e) != 0.0);
            assert!(cloned_has_nonzero);
        }

        // taxa mapping preserved: which node hosts taxon 2 in original vs clone?
        let orig_t2 = g.base.get_taxon_node(2);
        let clone_t2 = cloned.base.get_taxon_node(2);
        assert!(orig_t2.is_some() && clone_t2.is_some());

        // labels preserved (up to node-index renaming)
        // Check that the set of node labels is equal
        let mut labs_g: Vec<String> = g
            .base
            .graph
            .node_indices()
            .filter_map(|v| g.base.node_label(v).map(|s| s.to_string()))
            .collect();
        labs_g.sort();
        let mut labs_c: Vec<String> = cloned
            .base
            .graph
            .node_indices()
            .filter_map(|v| cloned.base.node_label(v).map(|s| s.to_string()))
            .collect();
        labs_c.sort();
        assert_eq!(labs_g, labs_c);

        // cycle mapping preserved
        assert_eq!(g.get_cycle(), cloned.get_cycle());
    }
}

use anyhow::Result;
use fixedbitset::FixedBitSet;
use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, NodeIndexable};
use std::collections::{BTreeMap, HashMap};

use crate::algorithms::equal_angle::{Pt, assign_coordinates_to_nodes};
use crate::nexus::network_writer::{leaf_label, node_degree};
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
    // rotation system (cyclic order of incident edges)
    rotation: HashMap<NodeIndex, Vec<EdgeIndex>>,
    pub node_ids: BTreeMap<NodeIndex, usize>,
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
        self.rotation.clear();
    }

    pub fn name(&self) -> Option<&str> {
        self.base.name()
    }
    pub fn set_name<S: Into<String>>(&mut self, s: S) {
        self.base.set_name(s);
    }

    /* ---------------- edge split-id / angle maps ---------------- */

    fn splits_mut(&mut self) -> &mut HashMap<EdgeIndex, i32> {
        self.edge_split_map.get_or_insert_with(HashMap::new)
    }
    fn angles_mut(&mut self) -> &mut HashMap<EdgeIndex, f64> {
        self.edge_angle_map.get_or_insert_with(HashMap::new)
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

    pub fn create_node_ids(&mut self) {
        self.node_ids.clear();
        let mut next = 1usize;
        for v in self.base.graph.node_indices() {
            self.node_ids.insert(v, next);
            next += 1;
        }
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
                self.new_edge_with_label(u, v, l.to_string()).unwrap()
            } else {
                self.new_edge(u, v).unwrap()
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
            let w_incident: Vec<EdgeIndex> = self.base.graph.edges(w).map(|er| er.id()).collect();

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
                if let Ok(g) = self.new_edge(u, v) {
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
            self.remove_node_and_cleanup(w); // not base.remove_node_and_cleanup
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
        for s in used.ones() {
            if let Some(bits) = split2chars.get(&(s as i32)) {
                let mut tmp = flips.clone();
                tmp.union_with(bits);
                flips = tmp;
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

    /// Create (u,v) and append on both rotations.
    pub fn new_edge(&mut self, u: NodeIndex, v: NodeIndex) -> anyhow::Result<EdgeIndex> {
        let e = self.base.new_edge(u, v)?;
        self.rot_append(u, e);
        self.rot_append(v, e);
        Ok(e)
    }

    pub fn new_edge_with_label(
        &mut self,
        u: NodeIndex,
        v: NodeIndex,
        lbl: String,
    ) -> anyhow::Result<EdgeIndex> {
        let e = self.base.new_edge_with_label(u, v, lbl)?;
        self.rot_append(u, e);
        self.rot_append(v, e);
        Ok(e)
    }

    /// Create (u,v) and splice its v-side halfedge AFTER `f0_at_v` (matches Java’s AFTER).
    pub fn new_edge_after(
        &mut self,
        u: NodeIndex,
        v: NodeIndex,
        f0_at_v: EdgeIndex,
    ) -> anyhow::Result<EdgeIndex> {
        let e = self.base.new_edge(u, v)?;
        self.rot_append(u, e);
        self.rot_insert_after(v, f0_at_v, e);
        Ok(e)
    }

    /// Remove an edge and keep rotations in sync.
    pub fn remove_edge(&mut self, e: EdgeIndex) -> bool {
        if let Some((u, v)) = self.base.graph.edge_endpoints(e) {
            self.rot_remove(u, e);
            self.rot_remove(v, e);
            self.base.graph.remove_edge(e).is_some()
        } else {
            false
        }
    }

    /// Remove a node and all incident edges, updating rotations.
    pub fn remove_node_and_cleanup(&mut self, v: NodeIndex) {
        if let Some(rs) = self.rotation.remove(&v) {
            for e in rs {
                if let Some((a, b)) = self.base.graph.edge_endpoints(e) {
                    let other = if a == v { b } else { a };
                    self.rot_remove(other, e);
                    self.base.graph.remove_edge(e);
                }
            }
        }
        self.base.remove_node_and_cleanup(v);
    }

    /* --------------- rotation helpers --------------- */
    #[inline]
    pub fn rot_mut(&mut self, v: NodeIndex) -> &mut Vec<EdgeIndex> {
        self.rotation.entry(v).or_default()
    }
    #[inline]
    pub fn rot(&self, v: NodeIndex) -> &[EdgeIndex] {
        self.rotation.get(&v).map(|r| r.as_slice()).unwrap_or(&[])
    }
    #[inline]
    pub fn rot_insert_after(&mut self, v: NodeIndex, after: EdgeIndex, e_new: EdgeIndex) {
        let r = self.rot_mut(v);
        if let Some(i) = r.iter().position(|&x| x == after) {
            r.insert(i + 1, e_new);
        } else {
            r.push(e_new);
        }
    }
    #[inline]
    pub fn rot_append(&mut self, v: NodeIndex, e_new: EdgeIndex) {
        self.rot_mut(v).push(e_new);
    }
    #[inline]
    pub fn rot_remove(&mut self, v: NodeIndex, e: EdgeIndex) {
        if let Some(r) = self.rotation.get_mut(&v) {
            if let Some(i) = r.iter().position(|&x| x == e) {
                r.remove(i);
            }
            if r.is_empty() {
                self.rotation.remove(&v);
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

    #[inline]
    pub fn first_adjacent_edge(&self, v: NodeIndex) -> Option<EdgeIndex> {
        self.rot(v).first().copied()
    }
    #[inline]
    pub fn last_adjacent_edge(&self, v: NodeIndex) -> Option<EdgeIndex> {
        self.rot(v).last().copied()
    }
    pub fn next_adjacent_edge_cyclic(&self, v: NodeIndex, e: EdgeIndex) -> Option<EdgeIndex> {
        let r = self.rot(v);
        if r.is_empty() {
            return None;
        }
        if let Some(i) = r.iter().position(|&x| x == e) {
            Some(r[(i + 1) % r.len()])
        } else {
            r.first().copied()
        }
    }
    pub fn prev_adjacent_edge_cyclic(&self, v: NodeIndex, e: EdgeIndex) -> Option<EdgeIndex> {
        let r = self.rot(v);
        if r.is_empty() {
            return None;
        }
        if let Some(i) = r.iter().position(|&x| x == e) {
            Some(r[(i + r.len() - 1) % r.len()])
        } else {
            r.last().copied()
        }
    }

    #[inline]
    pub fn opposite(&self, v: NodeIndex, e: EdgeIndex) -> anyhow::Result<NodeIndex> {
        let (a, b) = self
            .base
            .graph
            .edge_endpoints(e)
            .ok_or_else(|| anyhow::anyhow!("opposite: edge {:?} has no endpoints", e))?;
        Ok(if a == v { b } else { a })
    }

    pub fn is_leaf_edge(&self, e: EdgeIndex) -> bool {
        if let Some((a, b)) = self.base.graph.edge_endpoints(e) {
            let da = self.base.graph.neighbors(a).count();
            let db = self.base.graph.neighbors(b).count();
            da == 1 || db == 1
        } else {
            false
        }
    }

    /* ------ Nexus Based Funcs ------- */
    pub fn get_node_translations(
        &self,
        taxa_labels_1based: &[String],
    ) -> Result<Vec<(usize, String)>> {
        let mut translations = Vec::new();
        for v in self.base.graph.node_indices() {
            if node_degree(&self.base, v) == 1 {
                if let Some(lbl) = leaf_label(&self.base, v, taxa_labels_1based) {
                    translations.push((self.node_ids[&v], lbl));
                }
            }
        }
        Ok(translations)
    }

    pub fn get_node_positions(&self) -> Result<Vec<(usize, f64, f64)>> {
        let coords = assign_coordinates_to_nodes(true, &self, 1, 0);
        let mut positions = Vec::new();
        for v in self.base.graph.node_indices() {
            let id = self.node_ids[&v];
            let pt = coords.get(&v).copied().unwrap_or(Pt(0.0, 0.0));
            positions.push((id, pt.0, pt.1));
        }
        Ok(positions)
    }

    pub fn get_graph_edges(&self) -> Result<Vec<(usize, usize, usize, i32, f64)>> {
        let mut eid = 1usize;
        let mut edges = Vec::new();
        for e in self.base.graph.edge_indices() {
            let (u, v) = self.base.graph.edge_endpoints(e).expect("valid endpoints");
            let su = self.node_ids[&u];
            let sv = self.node_ids[&v];
            let sid = self.get_split(e);
            let wgt = self.base.weight(e);
            edges.push((eid, su, sv, sid, wgt));
            eid += 1;
        }
        Ok(edges)
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
        let (g, a, b, c, _e_ab, _e_bc) = make_abc();

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
        let (mut g, _a, _b, _c, _e_ab, _e_bc) = make_abc();
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

    // ============================================================
    //  Additional comprehensive tests
    // ============================================================

    // ---- Helper: build a diamond graph A—B—C—D—A with an extra center node E ----
    //
    //       A
    //      / \
    //     B   D
    //      \ /
    //       C
    //
    //  Edges: A-B (split=1), B-C (split=2), C-D (split=3), A-D (split=4)
    //  Taxa: 1->A, 2->B, 3->C, 4->D
    fn make_diamond() -> (
        PhyloSplitsGraph,
        NodeIndex,
        NodeIndex,
        NodeIndex,
        NodeIndex,
        EdgeIndex,
        EdgeIndex,
        EdgeIndex,
        EdgeIndex,
    ) {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node_with_label("A");
        let b = g.base.new_node_with_label("B");
        let c = g.base.new_node_with_label("C");
        let d = g.base.new_node_with_label("D");

        g.base.add_taxon(a, 1);
        g.base.add_taxon(b, 2);
        g.base.add_taxon(c, 3);
        g.base.add_taxon(d, 4);

        let e_ab = g.new_edge(a, b).unwrap();
        g.set_split(e_ab, 1);
        g.base.set_weight(e_ab, 1.0);

        let e_bc = g.new_edge(b, c).unwrap();
        g.set_split(e_bc, 2);
        g.base.set_weight(e_bc, 2.0);

        let e_cd = g.new_edge(c, d).unwrap();
        g.set_split(e_cd, 3);
        g.base.set_weight(e_cd, 3.0);

        let e_ad = g.new_edge(a, d).unwrap();
        g.set_split(e_ad, 4);
        g.base.set_weight(e_ad, 4.0);

        (g, a, b, c, d, e_ab, e_bc, e_cd, e_ad)
    }

    // ---- copy_from: returns correct node/edge maps ----

    #[test]
    fn copy_from_returns_valid_maps() {
        let (src, a, _b, _c, e_ab, e_bc) = make_abc();
        let mut dst = PhyloSplitsGraph::new();
        let (node_map, edge_map) = dst.copy_from(&src);

        // every source node should appear in the map
        assert!(node_map.contains_key(&a));
        assert_eq!(node_map.len(), 3);

        // every source edge should appear in the map
        assert!(edge_map.contains_key(&e_ab));
        assert!(edge_map.contains_key(&e_bc));
        assert_eq!(edge_map.len(), 2);

        // mapped edges should have the same split ids
        let new_ab = edge_map[&e_ab];
        let new_bc = edge_map[&e_bc];
        assert_eq!(dst.get_split(new_ab), 99);
        assert_eq!(dst.get_split(new_bc), 5);

        // mapped edges should have the same weights
        assert!((dst.base.weight(new_ab) - 2.0).abs() < 1e-12);
        assert!((dst.base.weight(new_bc) - 3.0).abs() < 1e-12);

        // mapped edges should have the same angles
        assert!((dst.get_angle(new_ab) - 0.1).abs() < 1e-12);
        assert!((dst.get_angle(new_bc) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn copy_from_preserves_taxa_mapping() {
        let (src, _a, _b, _c, _e_ab, _e_bc) = make_abc();
        let mut dst = PhyloSplitsGraph::new();
        let (node_map, _edge_map) = dst.copy_from(&src);

        // taxon 1 should map to the new node corresponding to A
        for tax in 1..=3 {
            let src_node = src.base.get_taxon_node(tax).unwrap();
            let dst_node = dst.base.get_taxon_node(tax).unwrap();
            assert_eq!(dst_node, node_map[&src_node]);
        }
    }

    #[test]
    fn copy_from_preserves_name() {
        let (mut src, _, _, _, _, _) = make_abc();
        src.set_name("test_graph");
        let mut dst = PhyloSplitsGraph::new();
        dst.copy_from(&src);
        assert_eq!(dst.name(), Some("test_graph"));
    }

    // ---- clone_graph independence ----

    #[test]
    fn clone_graph_is_independent() {
        let (mut src, _a, _b, _c, e_ab, _e_bc) = make_abc();
        src.set_name("original");
        let cloned = src.clone_graph();

        // mutate original
        src.set_split(e_ab, 999);
        src.set_name("mutated");

        // clone should be unaffected
        assert_eq!(cloned.name(), Some("original"));
        // The clone's edges have different indices, so check by iterating
        let clone_splits = cloned.get_split_ids();
        assert!(clone_splits.contains(&99)); // original value
        assert!(!clone_splits.contains(&999)); // mutation should not propagate
    }

    // ---- remove_split: contract on diamond ----

    #[test]
    fn remove_split_on_diamond() {
        let (mut g, _a, _b, _c, _d, _e_ab, _e_bc, _e_cd, _e_ad) = make_diamond();

        let nodes_before = g.base.graph.node_count();
        let edges_before = g.base.graph.edge_count();
        assert_eq!(nodes_before, 4);
        assert_eq!(edges_before, 4);

        // Remove split 1 (edge A-B). This contracts that edge, merging its endpoints.
        g.remove_split(1);

        // After contracting one split (which may appear on one edge), we lose
        // one node and at least one edge.
        let nodes_after = g.base.graph.node_count();
        let edges_after = g.base.graph.edge_count();
        assert!(
            nodes_after < nodes_before,
            "should have fewer nodes after split removal"
        );
        assert!(
            edges_after < edges_before,
            "should have fewer edges after split removal"
        );

        // Taxon 2 (originally on B) should still be mapped somewhere
        assert!(g.base.get_taxon_node(2).is_some());

        // Split 1 should no longer be in the graph
        let remaining_splits = g.get_split_ids();
        assert!(!remaining_splits.contains(&1), "split 1 should be removed");
    }

    #[test]
    fn remove_split_no_op_when_split_absent() {
        let (mut g, _a, _b, _c, _e_ab, _e_bc) = make_abc();
        let node_count_before = g.base.graph.node_count();
        let edge_count_before = g.base.graph.edge_count();

        // Remove a split id that doesn't exist
        g.remove_split(777);

        assert_eq!(g.base.graph.node_count(), node_count_before);
        assert_eq!(g.base.graph.edge_count(), edge_count_before);
    }

    // ---- get_separator ----

    #[test]
    fn get_separator_finds_edge_with_target_split() {
        let (g, a, _b, _c, _e_ab, _e_bc) = make_abc();
        let mut seen = FixedBitSet::with_capacity(g.base.graph.node_bound());

        let result = g.get_separator(5, a, None, &mut seen);
        assert!(result.is_some());
        let (node, edge) = result.unwrap();
        assert_eq!(g.get_split(edge), 5);
        // The node is on the side we approached from
        let (u, v) = g.base.graph.edge_endpoints(edge).unwrap();
        assert!(u == node || v == node);
    }

    #[test]
    fn get_separator_returns_none_for_absent_split() {
        let (g, a, _b, _c, _e_ab, _e_bc) = make_abc();
        let mut seen = FixedBitSet::with_capacity(g.base.graph.node_bound());

        let result = g.get_separator(777, a, None, &mut seen);
        assert!(result.is_none());
    }

    #[test]
    fn get_separator_respects_seen_set() {
        let (g, a, b, _c, _e_ab, _e_bc) = make_abc();
        let mut seen = FixedBitSet::with_capacity(g.base.graph.node_bound());
        // Mark B as already seen, so DFS from A should not cross to C
        seen.insert(b.index());

        // Split 5 is on edge B-C, but we can't reach it because B is already seen
        let result = g.get_separator(5, a, None, &mut seen);
        assert!(result.is_none());
    }

    // ---- label_nodes_by_sequences ----

    #[test]
    fn label_nodes_by_sequences_flip_logic() {
        // Build a simple path: A -- B -- C
        // Split 1 on edge A-B flips character 1
        // Split 2 on edge B-C flips character 2
        // Baseline: all '0'
        // A sees no flips: "00"
        // B sees split 1 flip: char 1 flips -> "10"
        // C sees splits 1 and 2: chars 1 and 2 flip -> "11"
        let (g, a, b, c, _e_ab, _e_bc) = make_abc();

        let mut split2chars: HashMap<i32, FixedBitSet> = HashMap::new();
        let mut bits99 = FixedBitSet::with_capacity(3);
        bits99.insert(1); // split 99 flips char 1
        split2chars.insert(99, bits99);

        let mut bits5 = FixedBitSet::with_capacity(3);
        bits5.insert(2); // split 5 flips char 2
        split2chars.insert(5, bits5);

        // first_chars: 1-based, indices 1..2 are '0'
        let first = vec![0u8, b'0', b'0'];
        let labels = g.label_nodes_by_sequences(&split2chars, &first);

        // A: start node (taxon 1), no splits traversed => "00" (no flips)
        assert_eq!(labels[&a], "00");
        // B: traversed split 99, flips char 1 => "10"
        assert_eq!(labels[&b], "10");
        // C: traversed splits 99 and 5, flips chars 1 and 2 => "11"
        assert_eq!(labels[&c], "11");
    }

    #[test]
    fn label_nodes_by_sequences_returns_empty_without_taxon1() {
        // Build a graph without taxon 1
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node_with_label("A");
        g.base.add_taxon(a, 2); // taxon 2, not 1

        let split2chars: HashMap<i32, FixedBitSet> = HashMap::new();
        let first = vec![0u8, b'0'];
        let labels = g.label_nodes_by_sequences(&split2chars, &first);
        assert!(labels.is_empty());
    }

    // ---- new_edge_with_label / new_edge_after ----

    #[test]
    fn new_edge_with_label_stores_label_and_updates_rotation() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node_with_label("A");
        let b = g.base.new_node_with_label("B");

        let e = g.new_edge_with_label(a, b, "my_edge".to_string()).unwrap();
        assert_eq!(g.base.edge_label(e), Some("my_edge"));

        // rotation should contain the edge on both sides
        assert!(g.rot(a).contains(&e));
        assert!(g.rot(b).contains(&e));
    }

    #[test]
    fn new_edge_self_loop_rejected() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node_with_label("A");

        let result = g.new_edge(a, a);
        assert!(result.is_err());

        let result2 = g.new_edge_with_label(a, a, "loop".to_string());
        assert!(result2.is_err());
    }

    #[test]
    fn new_edge_after_inserts_in_correct_rotation_position() {
        // new_edge_after(u, v, f0_at_v) inserts the new edge AFTER f0_at_v
        // in v's rotation, while u gets a plain append.
        let mut g = PhyloSplitsGraph::new();
        let center = g.base.new_node_with_label("center");
        let a = g.base.new_node_with_label("A");
        let b = g.base.new_node_with_label("B");
        let c = g.base.new_node_with_label("C");

        let e_ca = g.new_edge(center, a).unwrap(); // center rot: [e_ca]
        let e_cb = g.new_edge(center, b).unwrap(); // center rot: [e_ca, e_cb]

        // new_edge_after(c, center, e_ca) means:
        //   u=c, v=center => rot_append(c, e) and rot_insert_after(center, e_ca, e)
        let e_cc = g.new_edge_after(c, center, e_ca).unwrap();

        // center rotation: e_ca was at index 0, so e_cc inserted at index 1 => [e_ca, e_cc, e_cb]
        let rot = g.rot(center);
        assert_eq!(rot.len(), 3);
        assert_eq!(rot[0], e_ca);
        assert_eq!(rot[1], e_cc);
        assert_eq!(rot[2], e_cb);

        // c gets a plain append
        let rot_c = g.rot(c);
        assert_eq!(rot_c.len(), 1);
        assert_eq!(rot_c[0], e_cc);
    }

    #[test]
    fn new_edge_after_falls_back_to_append_when_ref_not_found() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node_with_label("A");
        let b = g.base.new_node_with_label("B");
        let c = g.base.new_node_with_label("C");

        let e_ab = g.new_edge(a, b).unwrap();
        // Use a bogus reference edge index
        let bogus = EdgeIndex::new(999);
        let e_ac = g.new_edge_after(a, c, bogus).unwrap();

        // Since the bogus edge isn't in A's rotation, e_ac should just be appended
        let rot = g.rot(a);
        assert_eq!(rot.len(), 2);
        assert_eq!(rot[0], e_ab);
        assert_eq!(rot[1], e_ac);
    }

    // ---- remove_edge ----

    #[test]
    fn remove_edge_cleans_up_rotation() {
        let (g, a, _b, _c, _d) = make_diamond_via_new_edge();

        // Verify edges exist and rotation is populated
        assert!(!g.rot(a).is_empty());

        let mut g = g;
        let edges_at_a: Vec<EdgeIndex> = g.rot(a).to_vec();
        let e = edges_at_a[0];

        let removed = g.remove_edge(e);
        assert!(removed);

        // Edge should not be in rotation of either endpoint anymore
        assert!(!g.rot(a).contains(&e));
        // Just verify the edge is gone from the graph
        assert!(g.base.graph.edge_endpoints(e).is_none());
    }

    // Helper that builds edges through PhyloSplitsGraph::new_edge (updating rotation)
    fn make_diamond_via_new_edge() -> (PhyloSplitsGraph, NodeIndex, NodeIndex, NodeIndex, NodeIndex)
    {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node_with_label("A");
        let b = g.base.new_node_with_label("B");
        let c = g.base.new_node_with_label("C");
        let d = g.base.new_node_with_label("D");

        g.base.add_taxon(a, 1);
        g.base.add_taxon(b, 2);
        g.base.add_taxon(c, 3);
        g.base.add_taxon(d, 4);

        g.new_edge(a, b).unwrap();
        g.new_edge(b, c).unwrap();
        g.new_edge(c, d).unwrap();
        g.new_edge(a, d).unwrap();

        (g, a, b, c, d)
    }

    #[test]
    fn remove_edge_returns_false_for_nonexistent() {
        let mut g = PhyloSplitsGraph::new();
        let bogus = EdgeIndex::new(42);
        assert!(!g.remove_edge(bogus));
    }

    // ---- remove_node_and_cleanup ----

    #[test]
    fn remove_node_and_cleanup_removes_incident_edges_and_rotation() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node_with_label("A");
        let b = g.base.new_node_with_label("B");
        let c = g.base.new_node_with_label("C");
        g.base.add_taxon(b, 1);

        let e_ab = g.new_edge(a, b).unwrap();
        let e_bc = g.new_edge(b, c).unwrap();

        assert_eq!(g.base.graph.node_count(), 3);
        assert_eq!(g.base.graph.edge_count(), 2);

        g.remove_node_and_cleanup(b);

        // B is gone
        assert!(g.base.graph.node_weight(b).is_none());
        // Both incident edges are gone
        assert!(g.base.graph.edge_endpoints(e_ab).is_none());
        assert!(g.base.graph.edge_endpoints(e_bc).is_none());
        // Rotation for B is gone
        assert!(g.rot(b).is_empty());
        // Rotations for A and C no longer reference the removed edges
        assert!(!g.rot(a).contains(&e_ab));
        assert!(!g.rot(c).contains(&e_bc));
        // Taxon 1 should no longer map to anything (node was removed)
        assert!(g.base.get_taxon_node(1).is_none());
    }

    // ---- rotation system ----

    #[test]
    fn rot_append_builds_order() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();

        let e1 = g.base.new_edge(v, a).unwrap();
        let e2 = g.base.new_edge(v, b).unwrap();
        let e3 = g.base.new_edge(v, c).unwrap();

        g.rot_append(v, e1);
        g.rot_append(v, e2);
        g.rot_append(v, e3);

        let r = g.rot(v);
        assert_eq!(r, &[e1, e2, e3]);
    }

    #[test]
    fn rot_insert_after_middle() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();

        let e1 = g.base.new_edge(v, a).unwrap();
        let e2 = g.base.new_edge(v, b).unwrap();
        let e3 = g.base.new_edge(v, c).unwrap();

        g.rot_append(v, e1);
        g.rot_append(v, e3);

        // Insert e2 after e1
        g.rot_insert_after(v, e1, e2);

        let r = g.rot(v);
        assert_eq!(r, &[e1, e2, e3]);
    }

    #[test]
    fn rot_insert_after_at_end() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();

        let e1 = g.base.new_edge(v, a).unwrap();
        let e2 = g.base.new_edge(v, b).unwrap();

        g.rot_append(v, e1);
        // Insert e2 after e1 (at the end)
        g.rot_insert_after(v, e1, e2);

        let r = g.rot(v);
        assert_eq!(r, &[e1, e2]);
    }

    #[test]
    fn rot_insert_after_nonexistent_ref_falls_back_to_push() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();

        let e1 = g.base.new_edge(v, a).unwrap();
        let e2 = g.base.new_edge(v, b).unwrap();

        g.rot_append(v, e1);
        let bogus = EdgeIndex::new(999);
        g.rot_insert_after(v, bogus, e2);

        // e2 should be pushed to the end
        let r = g.rot(v);
        assert_eq!(r, &[e1, e2]);
    }

    #[test]
    fn rot_remove_cleans_up_empty_entry() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();

        let e = g.base.new_edge(v, a).unwrap();
        g.rot_append(v, e);

        assert_eq!(g.rot(v).len(), 1);

        g.rot_remove(v, e);

        // After removing the only edge, the rotation entry should be removed entirely
        assert!(g.rot(v).is_empty());
        assert!(!g.rotation.contains_key(&v));
    }

    #[test]
    fn rot_remove_partial() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();

        let e1 = g.base.new_edge(v, a).unwrap();
        let e2 = g.base.new_edge(v, b).unwrap();
        g.rot_append(v, e1);
        g.rot_append(v, e2);

        g.rot_remove(v, e1);

        let r = g.rot(v);
        assert_eq!(r, &[e2]);
        assert!(g.rotation.contains_key(&v)); // still has an entry
    }

    #[test]
    fn rot_remove_nonexistent_edge_is_no_op() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();

        let e = g.base.new_edge(v, a).unwrap();
        g.rot_append(v, e);

        let bogus = EdgeIndex::new(999);
        g.rot_remove(v, bogus); // should not panic

        assert_eq!(g.rot(v), &[e]);
    }

    // ---- taxon2cycle / get_cycle ----

    #[test]
    fn get_taxon2_cycle_returns_neg1_for_absent() {
        let g = PhyloSplitsGraph::new();
        assert_eq!(g.get_taxon2_cycle(1), -1);
        assert_eq!(g.get_taxon2_cycle(0), -1);
        assert_eq!(g.get_taxon2_cycle(100), -1);
    }

    #[test]
    fn set_taxon2_cycle_grows_vector() {
        let mut g = PhyloSplitsGraph::new();
        // Set taxon 5 to cycle position 3 (should grow the internal vec to size 5)
        g.set_taxon2_cycle(5, 3);
        assert_eq!(g.get_taxon2_cycle(5), 3);
        // Intermediate taxa should be 0 (which get_taxon2_cycle returns as 0)
        assert_eq!(g.get_taxon2_cycle(1), 0);
        assert_eq!(g.get_taxon2_cycle(4), 0);
    }

    #[test]
    fn set_taxon2_cycle_overwrites() {
        let mut g = PhyloSplitsGraph::new();
        g.set_taxon2_cycle(2, 10);
        assert_eq!(g.get_taxon2_cycle(2), 10);
        g.set_taxon2_cycle(2, 20);
        assert_eq!(g.get_taxon2_cycle(2), 20);
    }

    #[test]
    fn get_cycle_builds_correct_1based_array() {
        let (mut g, _a, _b, _c, _e_ab, _e_bc) = make_abc();
        // cycle: taxon 1 at pos 3, taxon 2 at pos 1, taxon 3 at pos 2
        g.set_taxon2_cycle(1, 3);
        g.set_taxon2_cycle(2, 1);
        g.set_taxon2_cycle(3, 2);

        let cycle = g.get_cycle();
        assert_eq!(cycle.len(), 4); // [0, t_at_pos1, t_at_pos2, t_at_pos3]
        assert_eq!(cycle[0], 0); // index 0 unused
        assert_eq!(cycle[1], 2); // position 1 -> taxon 2
        assert_eq!(cycle[2], 3); // position 2 -> taxon 3
        assert_eq!(cycle[3], 1); // position 3 -> taxon 1
    }

    #[test]
    fn get_cycle_empty_graph() {
        let g = PhyloSplitsGraph::new();
        let cycle = g.get_cycle();
        // no taxa => cycle is [0]
        assert_eq!(cycle, vec![0]);
    }

    // ---- count_splits / max_split_id ----

    #[test]
    fn count_splits_skips_zero_ids() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();

        let _e1 = g.new_edge(a, b).unwrap();
        let e2 = g.new_edge(b, c).unwrap();

        // _e1 has split 0 (default), e2 has split 7
        g.set_split(e2, 7);

        assert_eq!(g.count_splits(), 1); // only split 7 counted
        assert_eq!(g.max_split_id(), 7);
    }

    #[test]
    fn count_splits_deduplicates() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();
        let d = g.base.new_node();

        let e1 = g.new_edge(a, b).unwrap();
        let e2 = g.new_edge(b, c).unwrap();
        let e3 = g.new_edge(c, d).unwrap();

        // Two edges share the same split id
        g.set_split(e1, 3);
        g.set_split(e2, 3);
        g.set_split(e3, 5);

        assert_eq!(g.count_splits(), 2); // split 3 and split 5
        assert_eq!(g.max_split_id(), 5);
    }

    #[test]
    fn max_split_id_empty_graph() {
        let g = PhyloSplitsGraph::new();
        assert_eq!(g.max_split_id(), 0);
        assert_eq!(g.count_splits(), 0);
    }

    // ---- opposite / is_leaf_edge ----

    #[test]
    fn opposite_returns_correct_endpoint() {
        let (g, a, b, _c, e_ab, _e_bc) = make_abc();
        assert_eq!(g.opposite(a, e_ab).unwrap(), b);
        assert_eq!(g.opposite(b, e_ab).unwrap(), a);
    }

    #[test]
    fn opposite_errors_on_invalid_edge() {
        let g = PhyloSplitsGraph::new();
        let bogus_node = NodeIndex::new(0);
        let bogus_edge = EdgeIndex::new(999);
        let result = g.opposite(bogus_node, bogus_edge);
        assert!(result.is_err());
    }

    #[test]
    fn is_leaf_edge_detects_leaf() {
        // In make_abc, A—B—C: A has degree 1, C has degree 1, B has degree 2
        // e_ab is a leaf edge (A has degree 1)
        // e_bc is a leaf edge (C has degree 1)
        let (g, _a, _b, _c, e_ab, e_bc) = make_abc();
        assert!(g.is_leaf_edge(e_ab));
        assert!(g.is_leaf_edge(e_bc));
    }

    #[test]
    fn is_leaf_edge_internal_edge() {
        // In a path A-B-C-D, edge B-C is internal (both endpoints have degree 2)
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();
        let d = g.base.new_node();

        g.new_edge(a, b).unwrap();
        let e_bc = g.new_edge(b, c).unwrap();
        g.new_edge(c, d).unwrap();

        assert!(!g.is_leaf_edge(e_bc));
    }

    #[test]
    fn is_leaf_edge_invalid_returns_false() {
        let g = PhyloSplitsGraph::new();
        assert!(!g.is_leaf_edge(EdgeIndex::new(999)));
    }

    // ---- get_node_translations / get_graph_edges ----

    #[test]
    fn get_node_translations_returns_leaf_nodes() {
        // Build A—B—C: A and C are leaves (degree 1), B is internal
        let (mut g, _a, _b, _c, _e_ab, _e_bc) = make_abc();
        g.create_node_ids();

        // taxa_labels_1based: index 0 = taxon 1 label, index 1 = taxon 2 label, etc.
        let labels = vec![
            "Alpha".to_string(), // taxon 1
            "Beta".to_string(),  // taxon 2
            "Gamma".to_string(), // taxon 3
        ];

        let translations = g.get_node_translations(&labels).unwrap();

        // Should have 2 leaf nodes (A=taxon 1, C=taxon 3)
        // B has degree 2, so it is NOT a leaf in the petgraph sense
        assert_eq!(translations.len(), 2);

        // Check that translations contain the correct labels
        let labels_found: Vec<String> = translations.iter().map(|(_, l)| l.clone()).collect();
        assert!(labels_found.contains(&"Alpha".to_string()));
        assert!(labels_found.contains(&"Gamma".to_string()));
    }

    #[test]
    fn get_graph_edges_returns_all_edges_with_metadata() {
        let (mut g, _a, _b, _c, _e_ab, _e_bc) = make_abc();
        g.create_node_ids();

        let edges = g.get_graph_edges().unwrap();
        assert_eq!(edges.len(), 2);

        // edges are (eid, src_node_id, dst_node_id, split_id, weight)
        // Check that split IDs are correct (unordered since edge iteration order varies)
        let split_ids: Vec<i32> = edges.iter().map(|e| e.3).collect();
        assert!(split_ids.contains(&99));
        assert!(split_ids.contains(&5));

        // Check that weights are correct
        let weights: Vec<f64> = edges.iter().map(|e| e.4).collect();
        assert!(weights.contains(&2.0));
        assert!(weights.contains(&3.0));

        // Edge IDs should be sequential starting from 1
        let eids: Vec<usize> = edges.iter().map(|e| e.0).collect();
        assert!(eids.contains(&1));
        assert!(eids.contains(&2));
    }

    // ---- create_node_ids ----

    #[test]
    fn create_node_ids_assigns_sequential_ids() {
        let (mut g, a, b, c, _e_ab, _e_bc) = make_abc();
        g.create_node_ids();

        // All nodes should have IDs
        assert!(g.node_ids.contains_key(&a));
        assert!(g.node_ids.contains_key(&b));
        assert!(g.node_ids.contains_key(&c));

        // IDs should be sequential starting from 1
        let ids: Vec<usize> = g.node_ids.values().copied().collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert_eq!(ids.len(), 3);
    }

    // ---- cyclic adjacency traversal ----

    #[test]
    fn first_last_adjacent_edge() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();

        let e1 = g.new_edge(v, a).unwrap();
        let e2 = g.new_edge(v, b).unwrap();

        assert_eq!(g.first_adjacent_edge(v), Some(e1));
        assert_eq!(g.last_adjacent_edge(v), Some(e2));
    }

    #[test]
    fn first_last_adjacent_edge_empty() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();

        assert_eq!(g.first_adjacent_edge(v), None);
        assert_eq!(g.last_adjacent_edge(v), None);
    }

    #[test]
    fn next_adjacent_edge_cyclic_wraps_around() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();

        let e1 = g.new_edge(v, a).unwrap();
        let e2 = g.new_edge(v, b).unwrap();
        let e3 = g.new_edge(v, c).unwrap();

        // Next after e1 is e2
        assert_eq!(g.next_adjacent_edge_cyclic(v, e1), Some(e2));
        // Next after e2 is e3
        assert_eq!(g.next_adjacent_edge_cyclic(v, e2), Some(e3));
        // Next after e3 wraps to e1
        assert_eq!(g.next_adjacent_edge_cyclic(v, e3), Some(e1));
    }

    #[test]
    fn prev_adjacent_edge_cyclic_wraps_around() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();

        let e1 = g.new_edge(v, a).unwrap();
        let e2 = g.new_edge(v, b).unwrap();
        let e3 = g.new_edge(v, c).unwrap();

        // Prev of e1 wraps to e3
        assert_eq!(g.prev_adjacent_edge_cyclic(v, e1), Some(e3));
        // Prev of e3 is e2
        assert_eq!(g.prev_adjacent_edge_cyclic(v, e3), Some(e2));
        // Prev of e2 is e1
        assert_eq!(g.prev_adjacent_edge_cyclic(v, e2), Some(e1));
    }

    #[test]
    fn next_prev_adjacent_edge_empty_rotation() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let bogus = EdgeIndex::new(0);

        assert_eq!(g.next_adjacent_edge_cyclic(v, bogus), None);
        assert_eq!(g.prev_adjacent_edge_cyclic(v, bogus), None);
    }

    #[test]
    fn next_adjacent_edge_cyclic_unknown_edge_returns_first() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let e1 = g.new_edge(v, a).unwrap();

        let bogus = EdgeIndex::new(999);
        // When edge is not found in rotation, falls back to first
        assert_eq!(g.next_adjacent_edge_cyclic(v, bogus), Some(e1));
    }

    #[test]
    fn prev_adjacent_edge_cyclic_unknown_edge_returns_last() {
        let mut g = PhyloSplitsGraph::new();
        let v = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let _e1 = g.new_edge(v, a).unwrap();
        let e2 = g.new_edge(v, b).unwrap();

        let bogus = EdgeIndex::new(999);
        // When edge is not found in rotation, falls back to last
        assert_eq!(g.prev_adjacent_edge_cyclic(v, bogus), Some(e2));
    }

    // ---- clear ----

    #[test]
    fn clear_resets_everything() {
        let (mut g, _a, _b, _c, _e_ab, _e_bc) = make_abc();
        g.set_name("myname");
        g.set_taxon2_cycle(1, 1);

        g.clear();

        assert_eq!(g.base.graph.node_count(), 0);
        assert_eq!(g.base.graph.edge_count(), 0);
        assert_eq!(g.count_splits(), 0);
        assert_eq!(g.max_split_id(), 0);
        assert_eq!(g.get_taxon2_cycle(1), -1);
        assert!(g.rotation.is_empty());
    }

    // ---- count_nodes / count_edges / network_type ----

    #[test]
    fn count_nodes_and_edges() {
        let (g, _a, _b, _c, _e_ab, _e_bc) = make_abc();
        assert_eq!(g.count_nodes(), 3);
        assert_eq!(g.count_edges(), 2);
    }

    #[test]
    fn network_type_returns_expected() {
        let g = PhyloSplitsGraph::new();
        assert_eq!(g.network_type(), "phylo-splits");
    }

    // ---- get_split_ids ----

    #[test]
    fn get_split_ids_sorted_and_deduped() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();
        let d = g.base.new_node();
        let e = g.base.new_node();

        let e1 = g.new_edge(a, b).unwrap();
        let e2 = g.new_edge(b, c).unwrap();
        let e3 = g.new_edge(c, d).unwrap();
        let e4 = g.new_edge(d, e).unwrap();

        g.set_split(e1, 10);
        g.set_split(e2, 3);
        g.set_split(e3, 10); // duplicate
        g.set_split(e4, 7);

        let ids = g.get_split_ids();
        assert_eq!(ids, vec![3, 7, 10]);
    }

    // ---- edge split/angle defaults ----

    #[test]
    fn default_split_and_angle() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let e = g.new_edge(a, b).unwrap();

        // Default split is 0, default angle is 0.0
        assert_eq!(g.get_split(e), 0);
        assert_eq!(g.get_angle(e), 0.0);
    }

    // ---- copy_from clears destination ----

    #[test]
    fn copy_from_clears_destination_first() {
        let (src, _, _, _, _, _) = make_abc();

        // Build a destination with some existing state
        let mut dst = PhyloSplitsGraph::new();
        let x = dst.base.new_node_with_label("X");
        let y = dst.base.new_node_with_label("Y");
        dst.new_edge(x, y).unwrap();
        dst.set_name("old_name");

        // Copy from src (which has 3 nodes)
        dst.copy_from(&src);

        // Destination should now have exactly 3 nodes and 2 edges (not 5 nodes, 3 edges)
        assert_eq!(dst.base.graph.node_count(), 3);
        assert_eq!(dst.base.graph.edge_count(), 2);
    }

    // ---- rotation consistency with new_edge ----

    #[test]
    fn new_edge_populates_both_endpoint_rotations() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node();
        let b = g.base.new_node();

        let e = g.new_edge(a, b).unwrap();

        assert_eq!(g.rot(a), &[e]);
        assert_eq!(g.rot(b), &[e]);
    }

    #[test]
    fn multiple_edges_accumulate_in_rotation() {
        let mut g = PhyloSplitsGraph::new();
        let center = g.base.new_node();
        let a = g.base.new_node();
        let b = g.base.new_node();
        let c = g.base.new_node();

        let e1 = g.new_edge(center, a).unwrap();
        let e2 = g.new_edge(center, b).unwrap();
        let e3 = g.new_edge(center, c).unwrap();

        let rot = g.rot(center);
        assert_eq!(rot.len(), 3);
        assert_eq!(rot, &[e1, e2, e3]);
    }

    // ---- remove_split on empty / single-node graphs ----

    #[test]
    fn remove_split_on_empty_graph_does_not_panic() {
        let mut g = PhyloSplitsGraph::new();
        g.remove_split(1); // should be a no-op without panicking
    }

    #[test]
    fn remove_split_when_no_taxon1_does_nothing() {
        let mut g = PhyloSplitsGraph::new();
        let a = g.base.new_node();
        let b = g.base.new_node();
        g.base.add_taxon(a, 2); // taxon 2 only, no taxon 1
        let e = g.new_edge(a, b).unwrap();
        g.set_split(e, 1);

        let nodes_before = g.base.graph.node_count();
        g.remove_split(1);
        // Without taxon 1, remove_split early-returns
        assert_eq!(g.base.graph.node_count(), nodes_before);
    }

    // ---- opposite_of (internal helper, tested via opposite public API) ----

    #[test]
    fn opposite_works_from_either_side() {
        let (g, a, b, c, e_ab, e_bc) = make_abc();
        assert_eq!(g.opposite(a, e_ab).unwrap(), b);
        assert_eq!(g.opposite(b, e_ab).unwrap(), a);
        assert_eq!(g.opposite(b, e_bc).unwrap(), c);
        assert_eq!(g.opposite(c, e_bc).unwrap(), b);
    }
}

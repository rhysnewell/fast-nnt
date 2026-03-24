use anyhow::Result;
use petgraph::Undirected;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableGraph};
use std::collections::HashMap;

/// Default values (as in Java version)
pub const DEFAULT_WEIGHT: f64 = 1.0;
pub const DEFAULT_CONFIDENCE: f64 = 1.0;
pub const DEFAULT_PROBABILITY: f64 = 1.0;

/// Node payload: we store an optional label
#[derive(Debug, Clone, Default)]
pub struct NodeData {
    pub label: Option<String>,
}

/// Edge payload: we store an optional label
#[derive(Debug, Clone, Default)]
pub struct EdgeData {
    pub label: Option<String>,
}

/// Phylogenetic graph built on petgraph::StableGraph
#[derive(Debug, Clone, Default)]
pub struct PhyloGraph {
    name: Option<String>,
    pub graph: StableGraph<NodeData, EdgeData, Undirected>,
    // edge attributes kept in side maps keyed by EdgeIndex
    edge_weights: Option<HashMap<EdgeIndex, f64>>,
    edge_confidences: Option<HashMap<EdgeIndex, f64>>,
    edge_probabilities: Option<HashMap<EdgeIndex, f64>>,
    // taxa mappings
    taxon2node: Option<HashMap<usize, NodeIndex>>,
    node2taxa: Option<HashMap<NodeIndex, Vec<usize>>>,
}

impl PhyloGraph {
    /* ---------------- ctor / housekeeping ---------------- */

    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.graph = StableGraph::default();
        self.name = None;
        self.edge_weights = None;
        self.edge_confidences = None;
        self.edge_probabilities = None;
        self.taxon2node = None;
        self.node2taxa = None;
    }

    pub fn set_name<S: Into<String>>(&mut self, s: S) {
        self.name = Some(s.into());
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /* ---------------- node/edge creation & labels ---------------- */

    pub fn new_node(&mut self) -> NodeIndex {
        self.graph.add_node(NodeData::default())
    }

    pub fn new_node_with_label<S: Into<String>>(&mut self, label: S) -> NodeIndex {
        self.graph.add_node(NodeData {
            label: Some(label.into()),
        })
    }

    pub fn set_node_label<S: Into<String>>(&mut self, v: NodeIndex, label: S) {
        if let Some(nd) = self.graph.node_weight_mut(v) {
            nd.label = Some(label.into());
        }
    }

    pub fn node_label(&self, v: NodeIndex) -> Option<&str> {
        self.graph.node_weight(v).and_then(|nd| nd.label.as_deref())
    }

    pub fn new_edge(&mut self, u: NodeIndex, v: NodeIndex) -> Result<EdgeIndex> {
        if u == v {
            return Err(anyhow::anyhow!("Illegal self-edge"));
        }
        Ok(self.graph.add_edge(u, v, EdgeData::default()))
    }

    pub fn new_edge_with_label<S: Into<String>>(
        &mut self,
        u: NodeIndex,
        v: NodeIndex,
        label: S,
    ) -> Result<EdgeIndex> {
        if u == v {
            return Err(anyhow::anyhow!("Illegal self-edge"));
        }
        Ok(self.graph.add_edge(
            u,
            v,
            EdgeData {
                label: Some(label.into()),
            },
        ))
    }

    pub fn set_edge_label<S: Into<String>>(&mut self, e: EdgeIndex, label: S) {
        if let Some(ed) = self.graph.edge_weight_mut(e) {
            ed.label = Some(label.into());
        }
    }

    pub fn edge_label(&self, e: EdgeIndex) -> Option<&str> {
        self.graph.edge_weight(e).and_then(|ed| ed.label.as_deref())
    }

    /// Remove node and incident edges, and clean any taxon mappings for this node.
    pub fn remove_node_and_cleanup(&mut self, v: NodeIndex) -> bool {
        // clear taxa for this node first (mirrors Java's listener)
        self.clear_taxa_for_node(v);
        self.graph.remove_node(v).is_some()
    }

    /* ---------------- edge attributes: weight / confidence / prob ---------------- */

    fn weights_mut(&mut self) -> &mut HashMap<EdgeIndex, f64> {
        self.edge_weights.get_or_insert_with(HashMap::new)
    }
    fn confidences_mut(&mut self) -> &mut HashMap<EdgeIndex, f64> {
        self.edge_confidences.get_or_insert_with(HashMap::new)
    }
    fn probabilities_mut(&mut self) -> &mut HashMap<EdgeIndex, f64> {
        self.edge_probabilities.get_or_insert_with(HashMap::new)
    }

    pub fn has_edge_weights(&self) -> bool {
        self.edge_weights.is_some()
    }
    pub fn has_edge_confidences(&self) -> bool {
        self.edge_confidences.is_some()
    }
    pub fn has_edge_probabilities(&self) -> bool {
        self.edge_probabilities.is_some()
    }

    pub fn set_weight(&mut self, e: EdgeIndex, val: f64) {
        if val == DEFAULT_WEIGHT && self.edge_weights.is_none() {
            return; // match Java: don't allocate for default
        }
        self.weights_mut().insert(e, val);
    }
    pub fn weight(&self, e: EdgeIndex) -> f64 {
        self.edge_weights
            .as_ref()
            .and_then(|m| m.get(&e).copied())
            .unwrap_or(DEFAULT_WEIGHT)
    }

    pub fn set_confidence(&mut self, e: EdgeIndex, val: f64) {
        self.confidences_mut().insert(e, val);
    }
    pub fn confidence(&self, e: EdgeIndex) -> f64 {
        self.edge_confidences
            .as_ref()
            .and_then(|m| m.get(&e).copied())
            .unwrap_or(DEFAULT_CONFIDENCE)
    }

    pub fn set_probability(&mut self, e: EdgeIndex, val: f64) {
        self.probabilities_mut().insert(e, val);
    }
    pub fn probability(&self, e: EdgeIndex) -> f64 {
        self.edge_probabilities
            .as_ref()
            .and_then(|m| m.get(&e).copied())
            .unwrap_or(DEFAULT_PROBABILITY)
    }

    /* ---------------- taxa mapping ---------------- */

    fn taxon2node_mut(&mut self) -> &mut HashMap<usize, NodeIndex> {
        self.taxon2node.get_or_insert_with(HashMap::new)
    }
    fn node2taxa_mut(&mut self) -> &mut HashMap<NodeIndex, Vec<usize>> {
        self.node2taxa.get_or_insert_with(HashMap::new)
    }

    pub fn taxon2node(&self) -> Option<&HashMap<usize, NodeIndex>> {
        self.taxon2node.as_ref()
    }
    pub fn node2taxa(&self) -> Option<&HashMap<NodeIndex, Vec<usize>>> {
        self.node2taxa.as_ref()
    }

    pub fn get_taxon_node(&self, taxon: usize) -> Option<NodeIndex> {
        self.taxon2node.as_ref()?.get(&taxon).copied()
    }

    pub fn get_node_taxon(&self, node: NodeIndex) -> Option<&[usize]> {
        self.node2taxa.as_ref()?.get(&node).map(|v| v.as_slice())
    }

    pub fn number_of_taxa(&self) -> usize {
        self.taxon2node.as_ref().map(|m| m.len()).unwrap_or(0)
    }

    pub fn taxa_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.taxon2node
            .as_ref()
            .map(|m| m.keys().copied().collect::<Vec<_>>())
            .unwrap_or_default()
            .into_iter()
    }

    pub fn has_taxa(&self, v: NodeIndex) -> bool {
        self.number_of_taxa_at(v) > 0
    }

    pub fn number_of_taxa_at(&self, v: NodeIndex) -> usize {
        self.node2taxa
            .as_ref()
            .and_then(|m| m.get(&v))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Add a taxon id to a given node (avoids duplicates)
    pub fn add_taxon(&mut self, v: NodeIndex, taxon: usize) {
        self.taxon2node_mut().insert(taxon, v);
        let vec = self.node2taxa_mut().entry(v).or_default();
        if !vec.contains(&taxon) {
            vec.push(taxon);
        }
    }

    /// Clear taxa for a node (remove reverse mappings that point to this node)
    pub fn clear_taxa_for_node(&mut self, v: NodeIndex) {
        if let (Some(t2n), Some(n2t)) = (self.taxon2node.as_mut(), self.node2taxa.as_mut()) {
            if let Some(list) = n2t.get(&v) {
                for &t in list {
                    if t2n.get(&t).copied() == Some(v) {
                        t2n.remove(&t);
                    }
                }
            }
            n2t.remove(&v);
        }
    }

    /// First taxon id on the node (or None)
    pub fn first_taxon(&self, v: NodeIndex) -> Option<usize> {
        self.node2taxa
            .as_ref()
            .and_then(|m| m.get(&v))
            .and_then(|v| v.first().copied())
    }

    /// Remove a single taxon (does not delete node)
    pub fn remove_taxon(&mut self, taxon: usize) {
        if taxon == 0 {
            return;
        }
        if let Some(t2n) = self.taxon2node.as_mut() {
            if let Some(v) = t2n.remove(&taxon) {
                if let Some(n2t) = self.node2taxa.as_mut() {
                    if let Some(vec) = n2t.get_mut(&v) {
                        if let Some(pos) = vec.iter().position(|&x| x == taxon) {
                            vec.remove(pos);
                        }
                        if vec.is_empty() {
                            n2t.remove(&v);
                        }
                    }
                }
            }
        }
    }

    pub fn remove_edge(&mut self, e: EdgeIndex) {
        self.graph.remove_edge(e);
    }

    pub fn clear_all_taxa(&mut self) {
        if let Some(n2t) = self.node2taxa.as_mut() {
            n2t.clear();
        }
        if let Some(t2n) = self.taxon2node.as_mut() {
            t2n.clear();
        }
    }

    /* ---------------- utility / copy / merge ---------------- */

    /// Returns true if node is a leaf (degree <= 1).
    pub fn is_leaf(&self, v: NodeIndex) -> bool {
        self.graph.neighbors(v).count() <= 1
    }

    /// Change node labels using a mapping; if `leaves_only`, only change leaf node labels.
    pub fn change_labels(&mut self, old2new: &HashMap<String, String>, leaves_only: bool) {
        // 1) Collect target nodes under an immutable borrow
        let targets: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|&v| !leaves_only || self.is_leaf(v))
            .collect();

        // 2) Now mutate labels (iterator borrow has ended)
        for v in targets {
            if let Some(nd) = self.graph.node_weight_mut(v) {
                if let Some(cur) = nd.label.as_ref() {
                    if let Some(new_label) = old2new.get(cur) {
                        nd.label = Some(new_label.clone());
                    }
                }
            }
        }
    }

    /// Deep copy from another PhyloGraph (preserves labels and edge attributes; also copies taxa).
    pub fn copy_from(&mut self, src: &PhyloGraph) {
        self.clear();
        self.name = src.name.clone();

        // map old -> new nodes
        let mut map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        for v in src.graph.node_indices() {
            let label = src.node_label(v).map(|s| s.to_string());
            let w = if let Some(lbl) = label {
                self.new_node_with_label(lbl)
            } else {
                self.new_node()
            };
            map.insert(v, w);
        }

        // edges with labels & attributes
        for e in src.graph.edge_indices() {
            let (u_old, v_old) = src
                .graph
                .edge_endpoints(e)
                .expect("edge must have endpoints");
            let u_new = map[&u_old];
            let v_new = map[&v_old];
            let f = if let Some(lab) = src.edge_label(e).map(|s| s.to_string()) {
                self.new_edge_with_label(u_new, v_new, lab)
                    .expect("copied nodes are distinct")
            } else {
                self.new_edge(u_new, v_new)
                    .expect("copied nodes are distinct")
            };
            if src.has_edge_weights() {
                self.set_weight(f, src.weight(e));
            }
            if src.has_edge_confidences() {
                self.set_confidence(f, src.confidence(e));
            }
            if src.has_edge_probabilities() {
                self.set_probability(f, src.probability(e));
            }
        }

        // taxa
        if let Some(t2n) = src.taxon2node.as_ref() {
            for (&tax, &old_v) in t2n.iter() {
                let new_v = map[&old_v];
                self.add_taxon(new_v, tax);
            }
        }
    }

    /// Add (disjoint union) of another graph into this one.
    /// Returns a mapping from other's node indices to the new node indices.
    pub fn add_graph(&mut self, other: &PhyloGraph) -> HashMap<NodeIndex, NodeIndex> {
        let mut old2new: HashMap<NodeIndex, NodeIndex> = HashMap::default();

        for v in other.graph.node_indices() {
            let new_v = if let Some(l) = other.node_label(v) {
                self.new_node_with_label(l.to_string())
            } else {
                self.new_node()
            };
            old2new.insert(v, new_v);
        }
        for e in other.graph.edge_indices() {
            let (u, v) = other
                .graph
                .edge_endpoints(e)
                .expect("edge must have endpoints");
            let u2 = old2new[&u];
            let v2 = old2new[&v];
            let new_e = if let Some(l) = other.edge_label(e) {
                self.new_edge_with_label(u2, v2, l.to_string())
                    .expect("copied nodes are distinct")
            } else {
                self.new_edge(u2, v2).expect("copied nodes are distinct")
            };
            self.set_weight(new_e, other.weight(e));
            if other.has_edge_confidences() {
                self.set_confidence(new_e, other.confidence(e));
            }
            if other.has_edge_probabilities() {
                self.set_probability(new_e, other.probability(e));
            }
        }
        // Java add() did NOT copy taxa; we keep that behavior.
        old2new
    }

    pub fn get_opposite(&self, v: NodeIndex, e: EdgeIndex) -> NodeIndex {
        let endpoints = self.graph.edge_endpoints(e).expect("valid edge endpoints");
        if endpoints.0 == v {
            endpoints.1
        } else {
            endpoints.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::stable_graph::{EdgeIndex, NodeIndex};
    use std::collections::HashMap;

    /// Build a small graph: A--B--C with labels, weights, and taxa.
    fn make_abc() -> (
        PhyloGraph,
        NodeIndex,
        NodeIndex,
        NodeIndex,
        EdgeIndex,
        EdgeIndex,
    ) {
        let mut g = PhyloGraph::new();
        let a = g.new_node_with_label("A");
        let b = g.new_node_with_label("B");
        let c = g.new_node_with_label("C");

        g.add_taxon(a, 1);
        g.add_taxon(b, 2);
        g.add_taxon(c, 3);

        let e_ab = g.new_edge(a, b).unwrap();
        g.set_weight(e_ab, 2.5);

        let e_bc = g.new_edge(b, c).unwrap();
        g.set_weight(e_bc, 3.0);

        (g, a, b, c, e_ab, e_bc)
    }

    // ==================== Node operations ====================

    #[test]
    fn new_node_increments_count() {
        let mut g = PhyloGraph::new();
        assert_eq!(g.graph.node_count(), 0);

        let v1 = g.new_node();
        assert_eq!(g.graph.node_count(), 1);
        assert!(g.graph.contains_node(v1));

        let v2 = g.new_node();
        assert_eq!(g.graph.node_count(), 2);
        assert!(g.graph.contains_node(v2));
        assert_ne!(v1, v2);
    }

    #[test]
    fn new_node_with_label_stores_label() {
        let mut g = PhyloGraph::new();
        let v = g.new_node_with_label("taxon_alpha");
        assert_eq!(g.node_label(v), Some("taxon_alpha"));
    }

    #[test]
    fn node_without_label_returns_none() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        assert_eq!(g.node_label(v), None);
    }

    #[test]
    fn set_node_label_overrides_existing() {
        let mut g = PhyloGraph::new();
        let v = g.new_node_with_label("old");
        assert_eq!(g.node_label(v), Some("old"));
        g.set_node_label(v, "new");
        assert_eq!(g.node_label(v), Some("new"));
    }

    #[test]
    fn set_node_label_on_unlabeled_node() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        assert_eq!(g.node_label(v), None);
        g.set_node_label(v, "now_labeled");
        assert_eq!(g.node_label(v), Some("now_labeled"));
    }

    #[test]
    fn is_leaf_isolated_node() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        // degree 0 => leaf
        assert!(g.is_leaf(v));
    }

    #[test]
    fn is_leaf_with_one_edge() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        g.new_edge(a, b).unwrap();
        // degree 1 => leaf
        assert!(g.is_leaf(a));
        assert!(g.is_leaf(b));
    }

    #[test]
    fn is_leaf_internal_node() {
        let (g, _a, b, _c, _e_ab, _e_bc) = make_abc();
        // b has degree 2 => not a leaf
        assert!(!g.is_leaf(b));
    }

    // ==================== Edge operations ====================

    #[test]
    fn new_edge_creates_edge() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge(a, b).unwrap();
        assert_eq!(g.graph.edge_count(), 1);
        let (u, v) = g.graph.edge_endpoints(e).unwrap();
        assert!((u == a && v == b) || (u == b && v == a));
    }

    #[test]
    fn new_edge_with_label_stores_label() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge_with_label(a, b, "edge_AB").unwrap();
        assert_eq!(g.edge_label(e), Some("edge_AB"));
    }

    #[test]
    fn edge_without_label_returns_none() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge(a, b).unwrap();
        assert_eq!(g.edge_label(e), None);
    }

    #[test]
    fn set_edge_label_overrides_existing() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge_with_label(a, b, "old").unwrap();
        g.set_edge_label(e, "new");
        assert_eq!(g.edge_label(e), Some("new"));
    }

    #[test]
    fn count_edges() {
        let (g, _, _, _, _, _) = make_abc();
        assert_eq!(g.graph.edge_count(), 2);
    }

    #[test]
    fn self_edge_rejected() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let result = g.new_edge(a, a);
        assert!(result.is_err());
        assert_eq!(g.graph.edge_count(), 0);
    }

    #[test]
    fn self_edge_with_label_rejected() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let result = g.new_edge_with_label(a, a, "bad");
        assert!(result.is_err());
        assert_eq!(g.graph.edge_count(), 0);
    }

    #[test]
    fn remove_edge_works() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge(a, b).unwrap();
        assert_eq!(g.graph.edge_count(), 1);
        g.remove_edge(e);
        assert_eq!(g.graph.edge_count(), 0);
        // nodes still present
        assert!(g.graph.contains_node(a));
        assert!(g.graph.contains_node(b));
    }

    // ==================== Weight / confidence / probability ====================

    #[test]
    fn default_weight_when_no_weights_set() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge(a, b).unwrap();
        assert!(!g.has_edge_weights());
        assert_eq!(g.weight(e), DEFAULT_WEIGHT);
    }

    #[test]
    fn set_weight_default_value_does_not_allocate() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge(a, b).unwrap();
        // Setting default value when map doesn't exist should be a no-op
        g.set_weight(e, DEFAULT_WEIGHT);
        assert!(!g.has_edge_weights());
        assert_eq!(g.weight(e), DEFAULT_WEIGHT);
    }

    #[test]
    fn set_weight_non_default_allocates_and_returns() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge(a, b).unwrap();
        g.set_weight(e, 42.0);
        assert!(g.has_edge_weights());
        assert_eq!(g.weight(e), 42.0);
    }

    #[test]
    fn set_weight_after_map_exists_stores_default() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let c = g.new_node();
        let e1 = g.new_edge(a, b).unwrap();
        let e2 = g.new_edge(b, c).unwrap();
        // Force map to exist
        g.set_weight(e1, 5.0);
        assert!(g.has_edge_weights());
        // Now set default on e2 -- map already exists, so it stores it
        g.set_weight(e2, DEFAULT_WEIGHT);
        assert_eq!(g.weight(e2), DEFAULT_WEIGHT);
    }

    #[test]
    fn confidence_default_and_set() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge(a, b).unwrap();
        assert!(!g.has_edge_confidences());
        assert_eq!(g.confidence(e), DEFAULT_CONFIDENCE);

        g.set_confidence(e, 0.95);
        assert!(g.has_edge_confidences());
        assert_eq!(g.confidence(e), 0.95);
    }

    #[test]
    fn probability_default_and_set() {
        let mut g = PhyloGraph::new();
        let a = g.new_node();
        let b = g.new_node();
        let e = g.new_edge(a, b).unwrap();
        assert!(!g.has_edge_probabilities());
        assert_eq!(g.probability(e), DEFAULT_PROBABILITY);

        g.set_probability(e, 0.75);
        assert!(g.has_edge_probabilities());
        assert_eq!(g.probability(e), 0.75);
    }

    #[test]
    fn multiple_edges_have_independent_weights() {
        let (g, _, _, _, e_ab, e_bc) = make_abc();
        assert_eq!(g.weight(e_ab), 2.5);
        assert_eq!(g.weight(e_bc), 3.0);
    }

    // ==================== Taxon mapping ====================

    #[test]
    fn add_taxon_and_retrieve() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        g.add_taxon(v, 1);
        assert_eq!(g.get_taxon_node(1), Some(v));
        assert_eq!(g.get_node_taxon(v), Some(&[1_usize][..]));
    }

    #[test]
    fn number_of_taxa_empty() {
        let g = PhyloGraph::new();
        assert_eq!(g.number_of_taxa(), 0);
    }

    #[test]
    fn number_of_taxa_counts_correctly() {
        let (g, _, _, _, _, _) = make_abc();
        assert_eq!(g.number_of_taxa(), 3);
    }

    #[test]
    fn taxa_iter_returns_all_taxa() {
        let (g, _, _, _, _, _) = make_abc();
        let mut taxa: Vec<usize> = g.taxa_iter().collect();
        taxa.sort();
        assert_eq!(taxa, vec![1, 2, 3]);
    }

    #[test]
    fn taxon2node_map_exposed() {
        let (g, a, b, c, _, _) = make_abc();
        let map = g.taxon2node().unwrap();
        assert_eq!(map[&1], a);
        assert_eq!(map[&2], b);
        assert_eq!(map[&3], c);
    }

    #[test]
    fn get_taxon_node_missing_returns_none() {
        let (g, _, _, _, _, _) = make_abc();
        assert_eq!(g.get_taxon_node(999), None);
    }

    #[test]
    fn get_node_taxon_missing_returns_none() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        // no taxa ever added => node2taxa is None
        assert_eq!(g.get_node_taxon(v), None);
    }

    #[test]
    fn add_taxon_avoids_duplicates() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        g.add_taxon(v, 5);
        g.add_taxon(v, 5); // duplicate
        // should still be just one entry
        assert_eq!(g.get_node_taxon(v), Some(&[5_usize][..]));
        assert_eq!(g.number_of_taxa(), 1);
    }

    #[test]
    fn multiple_taxa_on_one_node() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        g.add_taxon(v, 10);
        g.add_taxon(v, 20);
        assert_eq!(g.number_of_taxa_at(v), 2);
        assert!(g.has_taxa(v));

        let taxa = g.get_node_taxon(v).unwrap();
        assert!(taxa.contains(&10));
        assert!(taxa.contains(&20));
    }

    #[test]
    fn first_taxon_returns_first_added() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        g.add_taxon(v, 7);
        g.add_taxon(v, 3);
        assert_eq!(g.first_taxon(v), Some(7));
    }

    #[test]
    fn first_taxon_no_taxa_returns_none() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        assert_eq!(g.first_taxon(v), None);
    }

    #[test]
    fn has_taxa_false_for_no_taxa() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        assert!(!g.has_taxa(v));
        assert_eq!(g.number_of_taxa_at(v), 0);
    }

    #[test]
    fn remove_taxon_removes_mapping() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        g.add_taxon(v, 5);
        g.add_taxon(v, 6);
        assert_eq!(g.number_of_taxa(), 2);

        g.remove_taxon(5);
        assert_eq!(g.number_of_taxa(), 1);
        assert_eq!(g.get_taxon_node(5), None);
        assert_eq!(g.get_node_taxon(v), Some(&[6_usize][..]));
    }

    #[test]
    fn remove_taxon_zero_is_noop() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        g.add_taxon(v, 1);
        g.remove_taxon(0); // should be ignored per implementation
        assert_eq!(g.number_of_taxa(), 1);
    }

    #[test]
    fn remove_taxon_last_on_node_cleans_node_entry() {
        let mut g = PhyloGraph::new();
        let v = g.new_node();
        g.add_taxon(v, 5);
        g.remove_taxon(5);
        // node2taxa entry should be removed when vec becomes empty
        assert_eq!(g.get_node_taxon(v), None);
    }

    #[test]
    fn clear_all_taxa() {
        let (mut g, _, _, _, _, _) = make_abc();
        assert_eq!(g.number_of_taxa(), 3);
        g.clear_all_taxa();
        assert_eq!(g.number_of_taxa(), 0);
    }

    // ==================== Name ====================

    #[test]
    fn name_default_is_none() {
        let g = PhyloGraph::new();
        assert_eq!(g.name(), None);
    }

    #[test]
    fn set_name_and_get() {
        let mut g = PhyloGraph::new();
        g.set_name("my_graph");
        assert_eq!(g.name(), Some("my_graph"));
    }

    #[test]
    fn set_name_overwrite() {
        let mut g = PhyloGraph::new();
        g.set_name("first");
        g.set_name("second");
        assert_eq!(g.name(), Some("second"));
    }

    // ==================== Clear ====================

    #[test]
    fn clear_resets_everything() {
        let (mut g, _, _, _, _, _) = make_abc();
        g.set_name("before");
        g.clear();

        assert_eq!(g.graph.node_count(), 0);
        assert_eq!(g.graph.edge_count(), 0);
        assert_eq!(g.name(), None);
        assert!(!g.has_edge_weights());
        assert!(!g.has_edge_confidences());
        assert!(!g.has_edge_probabilities());
        assert_eq!(g.number_of_taxa(), 0);
    }

    // ==================== copy_from ====================

    #[test]
    fn copy_from_preserves_structure() {
        let (src, _, _, _, _, _) = make_abc();
        let mut dst = PhyloGraph::new();
        dst.copy_from(&src);

        assert_eq!(dst.graph.node_count(), src.graph.node_count());
        assert_eq!(dst.graph.edge_count(), src.graph.edge_count());
    }

    #[test]
    fn copy_from_preserves_name() {
        let mut src = PhyloGraph::new();
        src.set_name("original");
        src.new_node();

        let mut dst = PhyloGraph::new();
        dst.copy_from(&src);
        assert_eq!(dst.name(), Some("original"));
    }

    #[test]
    fn copy_from_preserves_labels() {
        let (src, _, _, _, _, _) = make_abc();
        let mut dst = PhyloGraph::new();
        dst.copy_from(&src);

        // collect all labels from dst
        let mut labels: Vec<String> = dst
            .graph
            .node_indices()
            .filter_map(|v| dst.node_label(v).map(|s| s.to_string()))
            .collect();
        labels.sort();
        assert_eq!(labels, vec!["A", "B", "C"]);
    }

    #[test]
    fn copy_from_preserves_weights() {
        let (src, _, _, _, _, _) = make_abc();
        let mut dst = PhyloGraph::new();
        dst.copy_from(&src);

        assert!(dst.has_edge_weights());
        // Check that the weights were copied (values 2.5 and 3.0)
        let mut weights: Vec<f64> = dst.graph.edge_indices().map(|e| dst.weight(e)).collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(weights, vec![2.5, 3.0]);
    }

    #[test]
    fn copy_from_preserves_confidence_and_probability() {
        let mut src = PhyloGraph::new();
        let a = src.new_node();
        let b = src.new_node();
        let e = src.new_edge(a, b).unwrap();
        src.set_confidence(e, 0.88);
        src.set_probability(e, 0.77);

        let mut dst = PhyloGraph::new();
        dst.copy_from(&src);

        assert!(dst.has_edge_confidences());
        assert!(dst.has_edge_probabilities());
        let e_dst = dst.graph.edge_indices().next().unwrap();
        assert_eq!(dst.confidence(e_dst), 0.88);
        assert_eq!(dst.probability(e_dst), 0.77);
    }

    #[test]
    fn copy_from_preserves_taxa() {
        let (src, _, _, _, _, _) = make_abc();
        let mut dst = PhyloGraph::new();
        dst.copy_from(&src);

        assert_eq!(dst.number_of_taxa(), 3);
        let mut taxa: Vec<usize> = dst.taxa_iter().collect();
        taxa.sort();
        assert_eq!(taxa, vec![1, 2, 3]);

        // Each taxon maps to a node with the correct label
        for t in 1..=3 {
            let node = dst.get_taxon_node(t).unwrap();
            let label = dst.node_label(node).unwrap();
            let expected = match t {
                1 => "A",
                2 => "B",
                3 => "C",
                _ => unreachable!(),
            };
            assert_eq!(label, expected);
        }
    }

    #[test]
    fn copy_from_preserves_edge_labels() {
        let mut src = PhyloGraph::new();
        let a = src.new_node();
        let b = src.new_node();
        let e = src.new_edge_with_label(a, b, "my_edge").unwrap();
        assert_eq!(src.edge_label(e), Some("my_edge"));

        let mut dst = PhyloGraph::new();
        dst.copy_from(&src);
        let e_dst = dst.graph.edge_indices().next().unwrap();
        assert_eq!(dst.edge_label(e_dst), Some("my_edge"));
    }

    #[test]
    fn copy_from_clears_destination_first() {
        let (src, _, _, _, _, _) = make_abc();
        let mut dst = PhyloGraph::new();
        // Add some junk to dst before copy
        let x = dst.new_node_with_label("X");
        let y = dst.new_node_with_label("Y");
        dst.new_edge(x, y).unwrap();
        dst.set_name("old_name");

        dst.copy_from(&src);

        // dst should match src, not retain old data
        assert_eq!(dst.graph.node_count(), 3);
        assert_eq!(dst.graph.edge_count(), 2);
        assert_eq!(dst.name(), None); // src had no name
    }

    #[test]
    fn copy_is_independent_of_source() {
        let (mut src, a, _, _, e_ab, _) = make_abc();
        let mut dst = PhyloGraph::new();
        dst.copy_from(&src);

        // Mutate source
        src.set_weight(e_ab, 999.0);
        src.set_node_label(a, "CHANGED");

        // dst should be unaffected
        let mut dst_weights: Vec<f64> = dst.graph.edge_indices().map(|e| dst.weight(e)).collect();
        dst_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(dst_weights, vec![2.5, 3.0]);
    }

    // ==================== add_graph ====================

    #[test]
    fn add_graph_disjoint_union() {
        let mut g1 = PhyloGraph::new();
        let a = g1.new_node_with_label("A");
        let b = g1.new_node_with_label("B");
        g1.new_edge(a, b).unwrap();

        let mut g2 = PhyloGraph::new();
        let c = g2.new_node_with_label("C");
        let d = g2.new_node_with_label("D");
        g2.new_edge(c, d).unwrap();

        let mapping = g1.add_graph(&g2);

        assert_eq!(g1.graph.node_count(), 4);
        assert_eq!(g1.graph.edge_count(), 2);

        // mapping should have entries for c and d
        assert_eq!(mapping.len(), 2);
        // mapped nodes should have correct labels
        assert_eq!(g1.node_label(mapping[&c]), Some("C"));
        assert_eq!(g1.node_label(mapping[&d]), Some("D"));
    }

    #[test]
    fn add_graph_preserves_weights() {
        let mut g1 = PhyloGraph::new();
        let a = g1.new_node();
        let b = g1.new_node();
        let e1 = g1.new_edge(a, b).unwrap();
        g1.set_weight(e1, 10.0);

        let mut g2 = PhyloGraph::new();
        let c = g2.new_node();
        let d = g2.new_node();
        let e2 = g2.new_edge(c, d).unwrap();
        g2.set_weight(e2, 20.0);

        g1.add_graph(&g2);

        let mut weights: Vec<f64> = g1.graph.edge_indices().map(|e| g1.weight(e)).collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(weights, vec![10.0, 20.0]);
    }

    #[test]
    fn add_graph_does_not_copy_taxa() {
        let (mut g1, _, _, _, _, _) = make_abc();
        let original_taxa = g1.number_of_taxa();

        let mut g2 = PhyloGraph::new();
        let v = g2.new_node();
        g2.add_taxon(v, 99);

        g1.add_graph(&g2);

        // taxa from g2 should NOT be copied (per Java behavior)
        assert_eq!(g1.number_of_taxa(), original_taxa);
    }

    #[test]
    fn add_empty_graph_is_noop() {
        let (mut g, _, _, _, _, _) = make_abc();
        let empty = PhyloGraph::new();
        let mapping = g.add_graph(&empty);

        assert_eq!(g.graph.node_count(), 3);
        assert_eq!(g.graph.edge_count(), 2);
        assert!(mapping.is_empty());
    }

    // ==================== remove_node_and_cleanup ====================

    #[test]
    fn remove_node_removes_node_and_edges() {
        let (mut g, a, b, c, _, _) = make_abc();
        assert_eq!(g.graph.node_count(), 3);
        assert_eq!(g.graph.edge_count(), 2);

        // Remove b (connected to a and c)
        assert!(g.remove_node_and_cleanup(b));

        assert_eq!(g.graph.node_count(), 2);
        // Both edges incident to b are removed
        assert_eq!(g.graph.edge_count(), 0);
        // a and c still exist
        assert!(g.graph.contains_node(a));
        assert!(g.graph.contains_node(c));
        assert!(!g.graph.contains_node(b));
    }

    #[test]
    fn remove_node_cleans_taxon_mappings() {
        let (mut g, _a, b, _c, _, _) = make_abc();
        assert_eq!(g.get_taxon_node(2), Some(b));

        g.remove_node_and_cleanup(b);

        assert_eq!(g.get_taxon_node(2), None);
        assert_eq!(g.number_of_taxa(), 2); // only taxa 1 and 3 remain
    }

    #[test]
    fn remove_leaf_node() {
        let (mut g, a, _b, _c, _, _) = make_abc();
        g.remove_node_and_cleanup(a);

        assert_eq!(g.graph.node_count(), 2);
        assert_eq!(g.graph.edge_count(), 1); // only B--C remains
        assert_eq!(g.get_taxon_node(1), None);
        assert_eq!(g.number_of_taxa(), 2);
    }

    // ==================== change_labels ====================

    #[test]
    fn change_labels_all_nodes() {
        let (mut g, a, b, c, _, _) = make_abc();
        let mut map = HashMap::new();
        map.insert("A".to_string(), "Alpha".to_string());
        map.insert("B".to_string(), "Beta".to_string());
        map.insert("C".to_string(), "Gamma".to_string());

        g.change_labels(&map, false);

        assert_eq!(g.node_label(a), Some("Alpha"));
        assert_eq!(g.node_label(b), Some("Beta"));
        assert_eq!(g.node_label(c), Some("Gamma"));
    }

    #[test]
    fn change_labels_leaves_only() {
        let (mut g, a, b, c, _, _) = make_abc();
        // a and c are leaves (degree 1), b is internal (degree 2)
        let mut map = HashMap::new();
        map.insert("A".to_string(), "Alpha".to_string());
        map.insert("B".to_string(), "Beta".to_string());
        map.insert("C".to_string(), "Gamma".to_string());

        g.change_labels(&map, true);

        assert_eq!(g.node_label(a), Some("Alpha"));
        assert_eq!(g.node_label(b), Some("B")); // unchanged, not a leaf
        assert_eq!(g.node_label(c), Some("Gamma"));
    }

    #[test]
    fn change_labels_partial_map() {
        let (mut g, a, _b, c, _, _) = make_abc();
        let mut map = HashMap::new();
        map.insert("A".to_string(), "X".to_string());
        // B and C not in map

        g.change_labels(&map, false);

        assert_eq!(g.node_label(a), Some("X"));
        // B and C unchanged
        assert_eq!(g.node_label(c), Some("C"));
    }

    #[test]
    fn change_labels_empty_map_is_noop() {
        let (mut g, a, b, c, _, _) = make_abc();
        let map = HashMap::new();
        g.change_labels(&map, false);

        assert_eq!(g.node_label(a), Some("A"));
        assert_eq!(g.node_label(b), Some("B"));
        assert_eq!(g.node_label(c), Some("C"));
    }

    // ==================== get_opposite ====================

    #[test]
    fn get_opposite_returns_other_endpoint() {
        let (g, a, b, c, e_ab, e_bc) = make_abc();
        assert_eq!(g.get_opposite(a, e_ab), b);
        assert_eq!(g.get_opposite(b, e_ab), a);
        assert_eq!(g.get_opposite(b, e_bc), c);
        assert_eq!(g.get_opposite(c, e_bc), b);
    }

    // ==================== clear_taxa_for_node ====================

    #[test]
    fn clear_taxa_for_node_removes_only_that_nodes_taxa() {
        let (mut g, a, b, _c, _, _) = make_abc();
        g.clear_taxa_for_node(a);

        assert_eq!(g.get_taxon_node(1), None); // taxon 1 was on a
        assert_eq!(g.get_node_taxon(a), None);
        // taxon 2 on b is unaffected
        assert_eq!(g.get_taxon_node(2), Some(b));
        assert_eq!(g.number_of_taxa(), 2);
    }

    // ==================== node2taxa ====================

    #[test]
    fn node2taxa_exposed() {
        let (g, a, b, c, _, _) = make_abc();
        let map = g.node2taxa().unwrap();
        assert_eq!(map[&a], vec![1]);
        assert_eq!(map[&b], vec![2]);
        assert_eq!(map[&c], vec![3]);
    }

    // ==================== Default ====================

    #[test]
    fn default_graph_is_empty() {
        let g = PhyloGraph::default();
        assert_eq!(g.graph.node_count(), 0);
        assert_eq!(g.graph.edge_count(), 0);
        assert_eq!(g.name(), None);
        assert!(!g.has_edge_weights());
        assert!(!g.has_edge_confidences());
        assert!(!g.has_edge_probabilities());
        assert_eq!(g.number_of_taxa(), 0);
    }
}

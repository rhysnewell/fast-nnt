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
        if self.edge_weights.is_none() {
            self.edge_weights = Some(HashMap::new());
        }
        self.edge_weights.as_mut().unwrap()
    }
    fn confidences_mut(&mut self) -> &mut HashMap<EdgeIndex, f64> {
        if self.edge_confidences.is_none() {
            self.edge_confidences = Some(HashMap::new());
        }
        self.edge_confidences.as_mut().unwrap()
    }
    fn probabilities_mut(&mut self) -> &mut HashMap<EdgeIndex, f64> {
        if self.edge_probabilities.is_none() {
            self.edge_probabilities = Some(HashMap::new());
        }
        self.edge_probabilities.as_mut().unwrap()
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
        if self.taxon2node.is_none() {
            self.taxon2node = Some(HashMap::new());
        }
        self.taxon2node.as_mut().unwrap()
    }
    fn node2taxa_mut(&mut self) -> &mut HashMap<NodeIndex, Vec<usize>> {
        if self.node2taxa.is_none() {
            self.node2taxa = Some(HashMap::new());
        }
        self.node2taxa.as_mut().unwrap()
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
            let (u_old, v_old) = src.graph.edge_endpoints(e).unwrap();
            let u_new = map[&u_old];
            let v_new = map[&v_old];
            let f = if let Some(lab) = src.edge_label(e).map(|s| s.to_string()) {
                self.new_edge_with_label(u_new, v_new, lab)
                    .expect("self-edge")
            } else {
                self.new_edge(u_new, v_new).expect("self-edge")
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
            let (u, v) = other.graph.edge_endpoints(e).unwrap();
            let u2 = old2new[&u];
            let v2 = old2new[&v];
            let new_e = if let Some(l) = other.edge_label(e) {
                self.new_edge_with_label(u2, v2, l.to_string())
                    .expect("self-edge")
            } else {
                self.new_edge(u2, v2).expect("self-edge")
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

use ndarray::Array2;
use anyhow::{Result, Context};

use crate::{data::splits_blocks::SplitsBlock, nexus::bitset_to_vec, phylo::phylo_splits_graph::PhyloSplitsGraph};


pub struct Nexus {
    pub labels: Vec<String>,
    pub distance_matrix: Array2<f64>,
    pub splits: SplitsBlock,
    pub graph: PhyloSplitsGraph,
}

impl Nexus {
    pub fn new(labels: Vec<String>, distance_matrix: Array2<f64>, splits: SplitsBlock, graph: PhyloSplitsGraph) -> Self {
        Nexus { labels, distance_matrix, splits, graph }
    }

    pub fn get_labels(&self) -> &[String] {
        &self.labels
    }

    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    pub fn distance_matrix(&self) -> &Array2<f64> {
        &self.distance_matrix
    }
    
    pub fn splits(&self) -> &SplitsBlock {
        &self.splits
    }
    
    pub fn num_splits(&self) -> usize {
        self.splits.nsplits()
    }

    pub fn graph(&self) -> &PhyloSplitsGraph {
        &self.graph
    }

    pub fn get_cycle(&self) -> Vec<usize> {
        self.graph.get_cycle()
    }

    pub fn get_splits_records(&self) -> Vec<(String, f64, f64, Vec<usize>, Vec<usize>)> {
        self.splits
            .get_splits()
            .iter()
            .map(|s| (s.get_label().unwrap_or("").to_string(), s.get_weight(), s.get_confidence(), bitset_to_vec(s.get_a()), bitset_to_vec(s.get_b())))
            .collect()
    }

    pub fn get_node_translations(&self) -> Result<Vec<(usize, String)>> {
        self.graph.get_node_translations(&self.labels).context("Failed to get node translations")
    }

    pub fn get_node_positions(&self) -> Result<Vec<(usize, f64, f64)>> {
        self.graph.get_node_positions().context("Failed to get node positions")
    }

    pub fn get_graph_edges(&self) -> Result<Vec<(usize, usize, usize, i32, f64)>> {
        self.graph.get_graph_edges().context("Failed to get graph edges")
    }    
}
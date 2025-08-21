use extendr_api::prelude::*;
use ndarray::Array2;
use fast_nnt_core::{run_fast_nnt_from_memory, cli::NeighbourNetArgs, nexus::nexus::Nexus};

#[extendr]
pub struct RNexus { inner: Nexus }

#[extendr] 
impl RNexus {     
    pub fn get_labels(&self) -> Vec<String> {
        self.inner.get_labels().to_vec()
    }

    /// Example: splits as list[tuple[list[int], list[int], float]]
    fn get_splits_records(&self) -> Vec<(String, f64, f64, Vec<usize>, Vec<usize>)> {
        self.inner
            .get_splits_records()
    }

    fn get_node_translations(&self) -> Vec<(usize, String)> {
        // need to reindex by -1
        self.inner.get_node_translations().expect("Failed to get node translations").into_iter()
            .map(|(id, label)| (id - 1, label)) // Convert to 0-based indexing
            .collect()
    }

    fn get_node_positions(&self) -> Vec<(usize, f64, f64)> {
        self.inner.get_node_positions().expect("Failed to get node positions").into_iter()
            .map(|(id, x, y)| (id - 1, x, y)) // Convert to 0-based indexing
            .collect()
    }

    /// Example: graph as list[(edge_id, u, v, split_id, weight)]
    fn get_graph_edges(&self) -> Vec<(usize, usize, usize, i32, f64)> {
        self.inner
            .get_graph_edges().expect("Failed to get graph edges").into_iter()
            .map(|(edge_id, u, v, split_id, weight)| (edge_id - 1, u - 1, v - 1, split_id, weight))
            .collect()
    }
}

fn to_numeric_matrix(x: Robj) -> extendr_api::Result<RMatrix<f64>> {
    if x.inherits("matrix") { x.try_into() } else { call!("as.matrix", x)?.try_into() }
}
fn rownames_or_default(x: Robj, n: usize) -> Vec<String> {
    if let Ok(rn) = call!("rownames", x) {
        if let Some(v) = rn.as_str_vector() { if v.len() == n { return v; } }
    }
    (0..n).map(|i| format!("row_{i}")).collect()
}

#[extendr]
fn run_neighbour_net(x: Robj, labels: Nullable<Robj>, out_dir: &str) -> extendr_api::Result<RNexus> {
    let mx = to_numeric_matrix(x.clone())?;
    let n = mx.nrows(); if n != mx.ncols() { return Err(Error::Other(format!("matrix must be square (got {}x{})", n, mx.ncols()))); }
    // Copy to row-major ndarray
    let mut arr = Array2::<f64>::zeros((n, n));
    for i in 0..n { for j in 0..n { arr[(i,j)] = mx[[i,j]]; } }
    let lbls = if let Some(l) = labels.into_option() {
        l.as_str_vector().ok_or_else(|| Error::Other("labels must be character vector".into()))?
    } else { rownames_or_default(x, n) };
    let args = NeighbourNetArgs::default(); // You can customize this if needed
    let nexus = run_fast_nnt_from_memory(arr, &lbls, &args)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(RNexus { inner: nexus })
}

extendr_module! {
    mod fastnnt;
    impl RNexus;
    fn run_neighbour_net;
}

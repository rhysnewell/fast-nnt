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

    // /// Example: splits as list[tuple[list[int], list[int], float]]
    // fn get_splits_records(&self) -> Vec<(String, f64, f64, Vec<usize>, Vec<usize>)> {
    //     self.inner
    //         .get_splits_records()
    // }

    /// (id, label) — IDs stay 1-based for R
    pub fn get_node_translations(&self) -> List {
        let data = self.inner
            .get_node_translations()
            .expect("Failed to get node translations");

        let mut ids: Vec<i32> = Vec::with_capacity(data.len());
        let mut labels: Vec<String> = Vec::with_capacity(data.len());
        for (id, label) in data {
            ids.push(id as i32);      // 1-based
            labels.push(label);
        }
        list!(id = r!(ids), label = r!(labels))
    }

    /// (id, x, y) — IDs stay 1-based for R
    pub fn get_node_positions(&self) -> List {
        let data = self.inner
            .get_node_positions()
            .expect("Failed to get node positions");

        let mut ids: Vec<i32> = Vec::with_capacity(data.len());
        let mut xs: Vec<f64> = Vec::with_capacity(data.len());
        let mut ys: Vec<f64> = Vec::with_capacity(data.len());
        for (id, x, y) in data {
            ids.push(id as i32);      // 1-based
            xs.push(x);
            ys.push(y);
        }
        list!(id = r!(ids), x = r!(xs), y = r!(ys))
    }

    /// (edge_id, u, v, split_id, weight) — all IDs stay 1-based for R
    pub fn get_graph_edges(&self) -> List {
        let data = self.inner
            .get_graph_edges()
            .expect("Failed to get graph edges");

        let mut eids: Vec<i32> = Vec::with_capacity(data.len());
        let mut us:   Vec<i32> = Vec::with_capacity(data.len());
        let mut vs:   Vec<i32> = Vec::with_capacity(data.len());
        let mut sids: Vec<i32> = Vec::with_capacity(data.len());
        let mut ws:   Vec<f64> = Vec::with_capacity(data.len());

        for (edge_id, u, v, split_id, weight) in data {
            eids.push(edge_id as i32);  // 1-based
            us.push(u as i32);          // 1-based
            vs.push(v as i32);          // 1-based
            sids.push(split_id);
            ws.push(weight);
        }
        list!(edge_id = r!(eids), u = r!(us), v = r!(vs), split_id = r!(sids), weight = r!(ws))
    }
}

fn to_numeric_matrix(x: &Robj) -> extendr_api::Result<RMatrix<f64>> {
    if x.inherits("matrix") {
        x.try_into()
    } else {
        let mx = call!("as.matrix", x)?;
        mx.try_into()
    }
}

fn colnames_or_default(x: &Robj, n: usize) -> Vec<String> {
    if let Ok(cn) = call!("colnames", x) {
        if let Some(v) = cn.as_str_vector() {
            if v.len() == n {
                return v.into_iter().map(|s| s.to_string()).collect();
            }
        }
    }
    (1..=n).map(|i| format!("col_{i}")).collect()
}

#[extendr]
fn run_neighbour_net(x: Robj, labels: Robj, _out_dir: &str) -> RNexus {
    // 1) coerce to numeric matrix
    let mx = to_numeric_matrix(&x).expect("Can't convert matrix");
    let n = mx.nrows();
    if n != mx.ncols() {
        // return Err(Error::Other(format!("matrix must be square (got {}x{})", n, mx.ncols())));
        panic!("matrix must be square (got {}x{})", n, mx.ncols());
    }

    // 2) copy column-major R -> row-major ndarray
    let mut arr = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            arr[(i, j)] = mx[[i, j]];
        }
    }

    // 3) labels
    let lbls: Vec<String> = if labels.is_null() {
        colnames_or_default(&x, n)
    } else {
        labels
            .as_str_vector()
            .ok_or_else(|| Error::Other("labels must be character vector".into())).expect("Failed to extract labels")
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    };

    // 4) args (adjust as needed)
    let args = NeighbourNetArgs::default();

    // 5) run core
    let nexus = run_fast_nnt_from_memory(arr, lbls, args)
        .map_err(|e| Error::Other(e.to_string())).expect("Failed to run NeighbourNet");

    RNexus { inner: nexus }
}

extendr_module! {
    mod fastnntr;
    impl RNexus;
    fn run_neighbour_net;
}

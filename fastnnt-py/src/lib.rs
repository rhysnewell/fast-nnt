use fast_nnt::cli::NeighbourNetArgs;
use fast_nnt::nexus::nexus::Nexus;
use fast_nnt::run_fast_nnt_from_memory;
use fast_nnt::ordering::OrderingMethod;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};


#[pyclass(name = "Nexus")]
pub struct PyNexus {
    inner: Nexus,
}

#[pymethods]
impl PyNexus {
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

// ---- helpers (Bound<'py, PyAny> everywhere) ----
fn coerce_to_numpy_2d<'py>(obj: Bound<'py, PyAny>) -> PyResult<PyReadonlyArray2<'py, f64>> {
    // Already a NumPy array?
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<f64>>() {
        return Ok(arr);
    }
    // pandas / polars DataFrame -> to_numpy()
    if obj.hasattr("to_numpy")? {
        let np = obj.call_method0("to_numpy")?;
        return np.extract::<PyReadonlyArray2<f64>>();
    }
    // array-like with __array__()
    if obj.hasattr("__array__")? {
        let np = obj.call_method0("__array__")?;
        return np.extract::<PyReadonlyArray2<f64>>();
    }
    Err(PyTypeError::new_err(
        "Expected a 2D numpy array or DataFrame-like (pandas/polars).",
    ))
}

fn labels_from<'py>(obj: Bound<'py, PyAny>, n: usize) -> PyResult<Vec<String>> {
    if let Ok(v) = obj.extract::<Vec<String>>() {
        if v.len() == n {
            return Ok(v);
        }
    }
    if let Ok(v) = obj.extract::<Vec<String>>() {
        let v: Vec<String> = v.into_iter().map(|s| s.to_string()).collect();
        if v.len() == n {
            return Ok(v);
        }
    }
    Err(PyTypeError::new_err(
        "labels must be a sequence of strings of length n",
    ))
}

fn infer_labels<'py>(obj: Bound<'py, PyAny>, n: usize) -> PyResult<Vec<String>> {
    // pandas: columns is an Index object with .tolist()
    if obj.hasattr("columns")? {
        let cols = obj.getattr("columns")?;
        if cols.hasattr("tolist")? {
            if let Ok(v) = cols.call_method0("tolist")?.extract::<Vec<String>>() {
                if v.len() == n { return Ok(v); }
            }
        } else if let Ok(v) = cols.extract::<Vec<String>>() {
            if v.len() == n { return Ok(v); }
        }
    }
    // polars: .columns is a plain list[str]
    if let Ok(v) = obj.getattr("columns")?.extract::<Vec<String>>() {
        if v.len() == n { return Ok(v); }
    }
    // fallback
    Ok((1..=n).map(|i| format!("col_{i}")).collect())
}
// ---- public API ----
#[pyfunction]
#[pyo3(signature = (x, max_iterations=5000, ordering_method=None, labels=None))]
fn run_neighbour_net<'py>(
    _py: Python<'py>,
    x: Bound<'py, PyAny>,
    max_iterations: usize,
    ordering_method: Option<String>,
    labels: Option<Bound<'py, PyAny>>,
) -> PyResult<PyNexus> {
    let arr = coerce_to_numpy_2d(x.clone())?;
    let view: Array2<f64> = arr.as_array().to_owned();
    let (n, m) = view.dim();
    if n != m {
        return Err(PyValueError::new_err(format!(
            "distance matrix must be square (got {n}x{m})"
        )));
    }

    let lbls = if let Some(l) = labels {
        labels_from(l, n)?
    } else {
        infer_labels(x, n)?
    };

    let mut args = NeighbourNetArgs::default(); // You can customize this if needed
    args.nnls_params.max_iterations = max_iterations;
    args.ordering = OrderingMethod::from_option(ordering_method);

    let nexus = run_fast_nnt_from_memory(view, lbls, args)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyNexus { inner: nexus })
}

#[pyfunction]
#[pyo3(signature = (x, max_iterations=5000, ordering_method=None, labels=None))]
fn run_neighbor_net<'py>(
    _py: Python<'py>,
    x: Bound<'py, PyAny>,
    max_iterations: usize,
    ordering_method: Option<String>,
    labels: Option<Bound<'py, PyAny>>,
) -> PyResult<PyNexus> {
    run_neighbour_net(_py, x, labels)
}

#[pymodule]
fn fastnntpy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNexus>()?;
    m.add_function(wrap_pyfunction!(run_neighbour_net, m)?)?;
    m.add_function(wrap_pyfunction!(run_neighbor_net, m)?)?;
    Ok(())
}

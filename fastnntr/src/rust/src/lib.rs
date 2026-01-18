use extendr_api::prelude::*;
use ndarray::Array2;
use std::collections::{HashMap, HashSet};


// From your core:
use fast_nnt::{run_fast_nnt_from_memory, cli::NeighbourNetArgs, nexus::nexus::Nexus, ordering::OrderingMethod};

/// Coerce to numeric matrix.
fn to_numeric_matrix(x: &Robj) -> extendr_api::Result<RMatrix<f64>> {
    if x.inherits("matrix") { x.try_into() } else { call!("as.matrix", x)?.try_into() }
}

/// Default to *column* names if provided; otherwise col_1..col_n
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

/// Build an integer matrix (n x 2) from two integer columns (column-major)
fn int_mat2(col1: &[i32], col2: &[i32]) -> extendr_api::Result<Robj> {
    let n = col1.len();
    let mut data = Vec::<i32>::with_capacity(n * 2);
    data.extend_from_slice(col1);
    data.extend_from_slice(col2);
    call!("matrix", r!(data), r!(n as i32), r!(2i32))
}

/// Build a numeric matrix (rows = max_id, cols = 2) from id->(x,y) map (column-major)
fn num_mat_vertices(max_id: usize, xy_by_id: &HashMap<usize, (f64, f64)>, flip_y: bool) -> Robj {
    let n = max_id;
    let mut xs = vec![f64::NAN; n];
    let mut ys = vec![f64::NAN; n];
    for (&id, &(x, y)) in xy_by_id {
        let idx = id - 1; // row index (1-based IDs → 0-based row)
        xs[idx] = x;
        ys[idx] = if flip_y { -y } else { y };
    }
    // column-major concat: all xs then all ys
    let mut data = Vec::<f64>::with_capacity(n * 2);
    data.extend_from_slice(&xs);
    data.extend_from_slice(&ys);
    let mut robj = r!(data);
    robj.set_attrib("dim", r!( [ n as i32, 2i32 ] )).unwrap();
    robj
}

fn vecvec_usize_rows_to_rmatrix_i32(rows: &[Vec<usize>]) -> Robj {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, |v| v.len());
    assert!(rows.iter().all(|v| v.len() == ncols), "ragged input");

    let m: RMatrix<i32> = RMatrix::new_matrix(nrows, ncols, |r, c| {
        rows[r][c] as i32
    });
    r!(m)
}

fn vecvec_usize_cols_to_rmatrix_i32(cols: &[Vec<usize>]) -> Robj {
    let ncols = cols.len();
    let nrows = cols.first().map_or(0, |v| v.len());
    assert!(cols.iter().all(|v| v.len() == nrows), "ragged input");

    // new_matrix expects a value for (row=r, col=c)
    let m: RMatrix<i32> = RMatrix::new_matrix(nrows, ncols, |r, c| {
        // If you worry about overflow, use i32::try_from(...) and handle Err.
        cols[c][r] as i32
    });
    r!(m)
}

/// Convert your in-memory `Nexus` into a networkx-like R list.
/// Returns a list with fields: edge, tip.label, edge.length, Nnode, splitIndex, splits, translate, and .plot$vertices.
/// `flip_y` matches the example (`vert[,2] <- -vert[,2]`).
fn nexus_to_networkx(nexus: &Nexus, flip_y: bool) -> extendr_api::Result<List> {
    // 1) labels / ntaxa
    let tip_labels: Vec<String> = nexus.get_labels().to_vec();
    let ntaxa = tip_labels.len();

    // 2) translations (node -> label) as data.frame
    let translations = nexus.get_node_translations()
        .map_err(|e| Error::Other(e.to_string()))?;
    let label_to_tip: HashMap<&str, usize> = tip_labels
        .iter()
        .enumerate()
        .map(|(i, lbl)| (lbl.as_str(), i + 1))
        .collect();
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    let mut trans_ids: Vec<i32> = Vec::with_capacity(translations.len());
    let mut trans_lbl: Vec<String> = Vec::with_capacity(translations.len());
    for (old_id, lbl) in translations {
        let tip_idx = *label_to_tip
            .get(lbl.as_str())
            .ok_or_else(|| Error::Other(format!("unknown label in translations: {lbl}")))?;
        id_map.insert(old_id, tip_idx);
        trans_ids.push(tip_idx as i32);
        trans_lbl.push(lbl);
    }
    // data.frame(node=..., label=..., stringsAsFactors=FALSE)
    let translate_df = data_frame!(node = r!(trans_ids.clone()), label = r!(trans_lbl.clone()));

    // 3) positions → vertices matrix, and id->(x,y) map
    let positions = nexus.get_node_positions()
        .map_err(|e| Error::Other(e.to_string()))?;
    let mut all_ids: HashSet<usize> = HashSet::with_capacity(positions.len());
    for (id, _x, _y) in &positions {
        all_ids.insert(*id);
    }

    // 4) edges (edge_id, u, v, split_id, weight) → edge matrix, splitIndex, edge.length
    let edges = nexus.get_graph_edges()
        .map_err(|e| Error::Other(e.to_string()))?;
    for (_eid, u, v, _sid, _w) in &edges {
        all_ids.insert(*u);
        all_ids.insert(*v);
    }

    let mut internal_ids: Vec<usize> = all_ids
        .iter()
        .copied()
        .filter(|id| !id_map.contains_key(id))
        .collect();
    internal_ids.sort_unstable();
    let mut next_internal = ntaxa + 1;
    for old_id in internal_ids {
        id_map.insert(old_id, next_internal);
        next_internal += 1;
    }

    let mut xy_by_id: HashMap<usize, (f64, f64)> = HashMap::with_capacity(positions.len());
    let mut max_id: usize = 0;
    for (old_id, x, y) in positions {
        let new_id = *id_map
            .get(&old_id)
            .ok_or_else(|| Error::Other(format!("missing node id mapping for {old_id}")))?;
        xy_by_id.insert(new_id, (x, y));
        if new_id > max_id { max_id = new_id; }
    }

    let mut us: Vec<i32> = Vec::with_capacity(edges.len());
    let mut vs: Vec<i32> = Vec::with_capacity(edges.len());
    let mut split_index: Vec<i32> = Vec::with_capacity(edges.len());
    let mut edge_length: Vec<f64> = Vec::with_capacity(edges.len());
    let mut max_node_in_edges: usize = 0;

    for (_eid, u, v, sid, _w) in edges {
        let u_new = *id_map
            .get(&u)
            .ok_or_else(|| Error::Other(format!("missing node id mapping for edge node {u}")))?;
        let v_new = *id_map
            .get(&v)
            .ok_or_else(|| Error::Other(format!("missing node id mapping for edge node {v}")))?;
        us.push(u_new as i32);
        vs.push(v_new as i32);
        split_index.push(sid);
        // edge length from vertices
        let (xu, yu) = *xy_by_id.get(&u_new).unwrap_or(&(f64::NAN, f64::NAN));
        let (xv, yv) = *xy_by_id.get(&v_new).unwrap_or(&(f64::NAN, f64::NAN));
        let yu2 = if flip_y { -yu } else { yu };
        let yv2 = if flip_y { -yv } else { yv };
        edge_length.push(((xu - xv).powi(2) + (yu2 - yv2).powi(2)).sqrt());

        if u_new > max_node_in_edges { max_node_in_edges = u_new; }
        if v_new > max_node_in_edges { max_node_in_edges = v_new; }
    }

    let edge_mat = int_mat2(&us, &vs)?;
    let vertices_mat = num_mat_vertices(max_node_in_edges.max(max_id), &xy_by_id, flip_y);

    // 5) splits: convert to list<int> (indices of taxa on one side).
    let mut splits_list = vec![vec![0; ntaxa]; nexus.splits.nsplits()];
    for (i, split) in nexus.splits.get_splits().iter().enumerate() {
        let split_vec = split.get_a().as_slice().to_vec();
        splits_list[i] = split_vec;
    }
    // assert they are all the same length
    let first_len = splits_list[0].len();

    let splits_mat = vecvec_usize_rows_to_rmatrix_i32(&splits_list);

    // 6) Nnode (as in your example): max(edge) - ntaxa
    let max_edge_node = max_node_in_edges as i32;
    let nnode = (max_edge_node as isize - ntaxa as isize).max(0) as i32;

    // 7) assemble .plot sublist
    let plot_list = list!(
        vertices   = vertices_mat,
        edge_color = r!("black"),
        edge_width = r!(3i32),
        edge_lty   = r!(1i32)
    );

    // 8) assemble main list (use safe names, then set R names with dots)
    let mut list = list!(
        edge        = edge_mat,
        tip_label   = r!(tip_labels),
        edge_length = r!(edge_length),
        Nnode       = r!(nnode),
        splitIndex  = r!(split_index),
        splits      = splits_mat,
        translate   = translate_df,
        plot        = r!(plot_list)
    );
    let names = vec![
        "edge",
        "tip.label",
        "edge.length",
        "Nnode",
        "splitIndex",
        "splits",
        "translate",
        ".plot",
    ];
    list.set_attrib("names", r!(names))?;
    // Match tanggle's S3 method naming so ggplot2 dispatches fortify.networx.
    list.set_class(&["networx", "phylo"])?;
    Ok(list)
}


#[extendr]
fn run_neighbornet_networkx(x: Robj, #[default = "TRUE"] flip_y: Robj, #[default = "NULL"] labels: Robj, #[default = "5000"] max_iterations: Robj, #[default = "NULL"] ordering_method: Robj) -> extendr_api::Result<List> {
    // Coerce & build Array2
    let mx = to_numeric_matrix(&x)?; let n = mx.nrows();
    if n != mx.ncols() {
        return Err(Error::Other(format!("matrix must be square (got {}x{})", n, mx.ncols())));
    }
    let mut arr = Array2::<f64>::zeros((n, n));
    for i in 0..n { for j in 0..n { arr[[i, j]] = mx[[i, j]]; } }
    // Labels: prefer explicit vector; else use column names
    let lbls: Vec<String> = if labels.is_null() {
        colnames_or_default(&x, n)
    } else {
        labels.as_str_vector()
            .ok_or_else(|| Error::Other("labels must be character vector".into()))?
            .into_iter().map(|s| s.to_string()).collect()
    };
    // Run core
    let mut args = NeighbourNetArgs::default();

    if !max_iterations.is_null() {
        if let Some(max_iter) = max_iterations.as_integer() {
            args.nnls_params.max_iterations = max_iter as usize;
        }
    }

    if !ordering_method.is_null() {
        args.ordering = OrderingMethod::from_option(ordering_method.as_str());
    }

    let nexus = run_fast_nnt_from_memory(arr, lbls, args)
        .map_err(|e| Error::Other(e.to_string()))?;
    // Convert to networkx-like list

    let mut flip = true;
    if !flip_y.is_null() {
        if let Some(val) = flip_y.as_bool() {
            flip = val;
        }
    }

    nexus_to_networkx(&nexus, flip)
}

#[extendr]
fn run_neighbournet_networkx(x: Robj, #[default = "TRUE"] flip_y: Robj, #[default = "NULL"] labels: Robj, #[default = "5000"] max_iterations: Robj, #[default = "NULL"] ordering_method: Robj) -> extendr_api::Result<List> {
    // Alias for backward compatibility
    run_neighbornet_networkx(x, flip_y, labels, max_iterations, ordering_method)
}

extendr_module! {
    mod fastnntr;
    fn run_neighbornet_networkx;
    fn run_neighbournet_networkx;
}

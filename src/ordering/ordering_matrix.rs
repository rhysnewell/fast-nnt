use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct NeighbourNetResult {
    pub ordering: Vec<usize>, // circular order in terms of original taxa indices (0-based)
}

type Matrix = Array2<f64>;

/// Update cluster distance matrix `dm` for row/col `j` given full `d` and cluster list `cl`.
fn update_dm(dm: &mut Matrix, cl: &Vec<Vec<usize>>, sums: &Vec<Vec<f64>>, j: usize) {
    let l = cl.len();
    // Compute all means to cluster j in parallel, then assign
    let col_vals: Vec<f64> = (0..l)
        .into_par_iter()
        .map(|i| mean_between_clusters_cached(cl, sums, i, j))
        .collect();

    // Assign row j
    for i in 0..l {
        dm[[i, j]] = col_vals[i];
    }
    // Mirror to column j
    for i in 0..l {
        dm[[j, i]] = col_vals[i];
    }
    dm[[j, j]] = 0.0;
}

/// Rx helper from the R code.
fn rx(d: &Matrix, x: &[usize], cl: &Vec<Vec<usize>>, sums: &Vec<Vec<f64>>) -> Vec<f64> {
    let lx = x.len();
    let mut res = vec![0.0; lx];
    for (i, &xi) in x.iter().enumerate() {
        let mut tmp = 0.0;
        // sum to other x's
        for (j, &xj) in x.iter().enumerate() {
            if j != i {
                tmp += unsafe { *d.uget((xi, xj)) };
            }
        }
        // plus mean to each other cluster
        for (c_idx, _c) in cl.iter().enumerate() {
            tmp += mean_single_to_cluster_cached(cl, sums, c_idx, xi);
        }
        res[i] = tmp;
    }
    res
}

/// Formula (1) reduction step; mutates `d` in place.
fn reduc(d: &mut Matrix, x: usize, y: usize, z: usize) {
    let n = d.nrows();
    // capture rows before overwriting
    let row_x = d.row(x).to_owned();
    let row_y = d.row(y).to_owned();
    let row_z = d.row(z).to_owned();

    // u = 2/3 * row_x + 1/3 * row_y
    // v = 2/3 * row_z + 1/3 * row_y
    let u: Array1<f64> = &(&row_x * (2.0 / 3.0)) + &(&row_y * (1.0 / 3.0));
    let v: Array1<f64> = &(&row_z * (2.0 / 3.0)) + &(&row_y * (1.0 / 3.0));

    let uv = (row_x[y] + row_x[z] + row_y[z]) / 3.0;

    // write back rows
    d.row_mut(x).assign(&u);
    d.row_mut(z).assign(&v);
    d.row_mut(y).fill(0.0);

    // symmetric columns
    for j in 0..n {
        d[[j, x]] = u[j];
        d[[j, z]] = v[j];
        d[[j, y]] = 0.0;
    }

    d[[x, z]] = uv;
    d[[z, x]] = uv;
    d[[x, x]] = 0.0;
    d[[z, z]] = 0.0;
}

/// Remove row/col `idx` from a square matrix, returning a new (n-1)x(n-1) array.
fn remove_row_col(m: &Matrix, idx: usize) -> Matrix {
    let n = m.nrows();
    debug_assert_eq!(n, m.ncols());
    if n == 1 {
        return Array2::zeros((0, 0));
    }
    let mut out = Array2::<f64>::zeros((n - 1, n - 1));
    // Top-left block
    if idx > 0 {
        out.slice_mut(s![0..idx, 0..idx])
            .assign(&m.slice(s![0..idx, 0..idx]));
    }
    // Top-right block
    if idx + 1 < n {
        out.slice_mut(s![0..idx, idx..])
            .assign(&m.slice(s![0..idx, (idx + 1)..]));
    }
    // Bottom-left block
    if idx + 1 < n {
        out.slice_mut(s![idx.., 0..idx])
            .assign(&m.slice(s![(idx + 1).., 0..idx]));
    }
    // Bottom-right block
    if idx + 1 < n {
        out.slice_mut(s![idx.., idx..])
            .assign(&m.slice(s![(idx + 1).., (idx + 1)..]));
    }
    out
}

/// Choose (e1,e2) minimizing DM[i,j] - r[i] - r[j] (i<j), parallelized.
fn choose_pair(dm: &Matrix, r: &[f64]) -> (usize, usize) {
    let l = dm.nrows();
    if l <= 1 {
        return (0, 0);
    }
    // Each worker scans a band of i and returns (best_val, i, j),
    // then we reduce to the global best.
    let per_i = (0..l).into_par_iter().map(|i| {
        let mut best = f64::INFINITY;
        let mut best_j = i + 1;
        for j in (i + 1)..l {
            let q = unsafe { *dm.uget((i, j)) } - r[i] - r[j];
            if q < best {
                best = q;
                best_j = j;
            }
        }
        (best, i, best_j)
    });

    let (.., bi, bj) = per_i.reduce(
        || (f64::INFINITY, 0usize, 1usize),
        |a, b| if a.0 <= b.0 { a } else { b },
    );
    (bi, bj)
}

fn remove_e2(
    cl: &mut Vec<Vec<usize>>,
    ord: &mut Vec<Vec<usize>>,
    dm: &mut Matrix,
    sums: &mut Vec<Vec<f64>>,
    e2: usize,
) {
    cl.remove(e2);
    ord.remove(e2);
    *dm = remove_row_col(dm, e2);
    sums.remove(e2);
}

/// The main ordering routine.
pub fn get_ordering_nn(x: &Matrix) -> Vec<usize> {
    assert_eq!(x.nrows(), x.ncols(), "Distance matrix must be square");
    let n = x.nrows();

    // Mutable working copy of D (modified by `reduc`)
    let mut d = x.clone();

    // Clusters & per-cluster linear orders
    let mut cl: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut ord: Vec<Vec<usize>> = cl.clone();
    let mut sums: Vec<Vec<f64>> = (0..n).map(|i| d.row(i).to_vec()).collect();

    // Cluster distance matrix (start with singleton distances)
    let mut dm = d.clone();

    while cl.len() > 1 {
        let l = dm.nrows();
        let (e1, e2) = if l > 2 {
            // r = rowSums(DM) / (l - 2), parallel row sums
            let denom = (l as f64) - 2.0;
            let r: Vec<f64> = dm
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| row.sum() / denom)
                .collect();

            choose_pair(&dm, &r)
        } else {
            (0, 1)
        };

        let n1 = cl[e1].len();
        let n2 = cl[e2].len();

        if n1 == 1 && n2 == 1 {
            // Simple merge of two leaves
            let mut new_cl = Vec::with_capacity(2);
            new_cl.extend_from_slice(&cl[e1]);
            new_cl.extend_from_slice(&cl[e2]);
            let new_ord = new_cl.clone();

            cl[e1] = new_cl;
            ord[e1] = new_ord;

            merge_cluster_sums(&mut sums, e1, e2);
            update_dm(&mut dm, &cl, &sums, e1);
            remove_e2(&mut cl, &mut ord, &mut dm, &mut sums, e2);
        } else {
            // Build "others" (all clusters except e1,e2)
            let mut others: Vec<Vec<usize>> = Vec::with_capacity(cl.len().saturating_sub(2));
            let mut others_sums: Vec<Vec<f64>> = Vec::with_capacity(cl.len().saturating_sub(2));
            for (idx, c) in cl.iter().enumerate() {
                if idx != e1 && idx != e2 {
                    others.push(c.clone());
                    others_sums.push(sums[idx].clone());
                }
            }

            // cltmp2 = elements of CL[e1] followed by CL[e2]
            let mut cltmp2: Vec<usize> = Vec::with_capacity(n1 + n2);
            cltmp2.extend_from_slice(&cl[e1]);
            cltmp2.extend_from_slice(&cl[e2]);

            let mut rtmp2 = rx(&d, &cltmp2, &others, &others_sums);
            let ltmp = cl[e1].len() + cl[e2].len() + others.len();
            if ltmp > 2 {
                let scale = 1.0 / ((ltmp as f64) - 2.0);
                for v in rtmp2.iter_mut() {
                    *v *= scale;
                }
            }

            // DM3 = d[cltmp2, cltmp2] - (rtmp2[i] + rtmp2[j])
            // We only need the cross block rows 0..n1-1, cols n1..n1+n2-1
            let mut best_val = f64::INFINITY;
            let mut best_row = 0usize; // 0..n1-1
            let mut best_col = 0usize; // 0..n2-1
            for col in 0..n2 {
                for row in 0..n1 {
                    let i = cltmp2[row];
                    let j = cltmp2[n1 + col];
                    let v = unsafe { *d.uget((i, j)) } - (rtmp2[row] + rtmp2[n1 + col]);
                    if v < best_val {
                        best_val = v;
                        best_row = row;
                        best_col = col;
                    }
                }
            }

            // Cases with cluster sizes from {1,2}
            let (new_cl, new_ord) = match (n1, n2) {
                (2, 1) => {
                    if best_row == 1 {
                        // blub == 2
                        reduc(&mut d, cl[e1][0], cl[e1][1], cl[e2][0]);
                        let nc = vec![cl[e1][0], cl[e2][0]];
                        let mut no = ord[e1].clone();
                        no.extend_from_slice(&ord[e2]);
                        (nc, no)
                    } else {
                        // else
                        reduc(&mut d, cl[e2][0], cl[e1][0], cl[e1][1]);
                        let nc = vec![cl[e2][0], cl[e1][1]];
                        let mut no = ord[e2].clone();
                        no.extend_from_slice(&ord[e1]);
                        (nc, no)
                    }
                }
                (1, 2) => {
                    if best_col == 0 {
                        // blub == 1
                        reduc(&mut d, cl[e1][0], cl[e2][0], cl[e2][1]);
                        let nc = vec![cl[e1][0], cl[e2][1]];
                        let mut no = ord[e1].clone();
                        no.extend_from_slice(&ord[e2]);
                        (nc, no)
                    } else {
                        // else
                        reduc(&mut d, cl[e2][0], cl[e2][1], cl[e1][0]);
                        let nc = vec![cl[e2][0], cl[e1][0]];
                        let mut no = ord[e2].clone();
                        no.extend_from_slice(&ord[e1]);
                        (nc, no)
                    }
                }
                (2, 2) => match (best_row, best_col) {
                    (0, 0) => {
                        // blub == 1
                        reduc(&mut d, cl[e1][1], cl[e1][0], cl[e2][0]);
                        reduc(&mut d, cl[e1][1], cl[e2][0], cl[e2][1]);
                        let nc = vec![cl[e1][1], cl[e2][1]];
                        let mut no = ord[e1].clone();
                        no.reverse();
                        no.extend_from_slice(&ord[e2]);
                        (nc, no)
                    }
                    (1, 0) => {
                        // blub == 2
                        reduc(&mut d, cl[e1][0], cl[e1][1], cl[e2][0]);
                        reduc(&mut d, cl[e1][0], cl[e2][0], cl[e2][1]);
                        let nc = vec![cl[e1][0], cl[e2][1]];
                        let mut no = ord[e1].clone();
                        no.extend_from_slice(&ord[e2]);
                        (nc, no)
                    }
                    (0, 1) => {
                        // blub == 3
                        reduc(&mut d, cl[e1][1], cl[e1][0], cl[e2][1]);
                        reduc(&mut d, cl[e1][1], cl[e2][1], cl[e2][0]);
                        let nc = vec![cl[e1][1], cl[e2][0]];
                        let mut no = ord[e1].clone();
                        no.reverse();
                        let mut oe2 = ord[e2].clone();
                        oe2.reverse();
                        no.extend_from_slice(&oe2);
                        (nc, no)
                    }
                    (1, 1) => {
                        reduc(&mut d, cl[e1][0], cl[e1][1], cl[e2][1]);
                        reduc(&mut d, cl[e1][0], cl[e2][1], cl[e2][0]);
                        let nc = vec![cl[e1][0], cl[e2][0]];
                        let mut no = ord[e1].clone();
                        let mut oe2 = ord[e2].clone();
                        oe2.reverse();
                        no.extend_from_slice(&oe2);
                        (nc, no)
                    }
                    _ => unreachable!(),
                },
                _ => panic!(
                    "Unhandled cluster sizes in NeighborNet step: n1={}, n2={}",
                    n1, n2
                ),
            };

            ord[e1] = new_ord;
            cl[e1] = new_cl;

            recompute_cluster_sum(&d, &cl, &mut sums, e1);
            update_dm(&mut dm, &cl, &sums, e1);
            remove_e2(&mut cl, &mut ord, &mut dm, &mut sums, e2);
        }
    }

    ord.into_iter().next().unwrap_or_default()
}

fn mean_between_clusters_cached(
    cl: &Vec<Vec<usize>>,
    sums: &Vec<Vec<f64>>,
    i: usize,
    j: usize,
) -> f64 {
    let ai = cl[i].len();
    let bj = cl[j].len();
    if ai == 0 || bj == 0 {
        return 0.0;
    }
    let (small, large) = if ai <= bj { (i, j) } else { (j, i) };
    let sum: f64 = cl[small].iter().map(|&t| sums[large][t]).sum();
    let denom = (ai * bj) as f64;
    if denom == 0.0 { 0.0 } else { sum / denom }
}

fn mean_single_to_cluster_cached(cl: &Vec<Vec<usize>>, sums: &Vec<Vec<f64>>, c_idx: usize, xi: usize) -> f64 {
    let size = cl[c_idx].len();
    if size == 0 {
        0.0
    } else {
        sums[c_idx][xi] / (size as f64)
    }
}

fn merge_cluster_sums(sums: &mut Vec<Vec<f64>>, e1: usize, e2: usize) {
    let add = sums[e2].clone();
    for (v, a) in sums[e1].iter_mut().zip(add.iter()) {
        *v += a;
    }
}

fn recompute_cluster_sum(d: &Matrix, cl: &Vec<Vec<usize>>, sums: &mut Vec<Vec<f64>>, idx: usize) {
    let mut out = vec![0.0f64; d.ncols()];
    for &t in cl[idx].iter() {
        let row = d.row(t);
        for (k, v) in row.iter().enumerate() {
            out[k] += *v;
        }
    }
    sums[idx] = out;
}

pub fn neighbor_net_ordering(x: &Matrix) -> NeighbourNetResult {
    NeighbourNetResult {
        ordering: get_ordering_nn(x),
    }
}

// --- Example usage & quick test ---
#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn smoke() {
        let d = array![
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ];
        let res = neighbor_net_ordering(&d);
        assert_eq!(res.ordering.len(), 5);
    }
}

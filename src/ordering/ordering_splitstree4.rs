use anyhow::{Context, Result, anyhow, ensure};
use ndarray::Array2;
use rayon::prelude::*;
use std::time::Instant;

const EPS: f64 = 1e-12;

/// Compute the NeighborNet circular ordering (Bryant & Huson 2005).
/// - `dist` is 0-based, shape n×n, symmetric with 0 on the diagonal.
/// - Returns a 1-based cycle with a leading 0 sentinel: `[0, t1, t2, ..., tn]`.
pub fn compute_order_splits_tree4(
    dist: &Array2<f64>,
    canonical_presort: bool,
) -> Result<Vec<usize>> {
    let n_tax = dist.nrows();
    let sx_mode = default_sx_mode(n_tax);
    compute_order_splits_tree4_with_sx(dist, sx_mode, canonical_presort)
}

#[derive(Clone, Copy, Debug)]
pub enum SxMode {
    Serial,
    Parallel,
}

/// Same as `compute_order_splits_tree4`, but allows selecting the Sx computation mode.
pub fn compute_order_splits_tree4_with_sx(
    dist: &Array2<f64>,
    sx_mode: SxMode,
    canonical_presort: bool,
) -> Result<Vec<usize>> {
    let n_tax = dist.nrows();
    ensure!(dist.ncols() == n_tax, "Distance matrix must be square");

    if n_tax <= 3 {
        let mut cycle = vec![0usize];
        cycle.extend(1..=n_tax);
        return Ok(cycle);
    }

    let perm = if canonical_presort {
        Some(canonical_permutation(dist))
    } else {
        None
    };

    let max_nodes = 3 * n_tax - 5;
    let stride = max_nodes;
    let mut mat = vec![0.0_f64; max_nodes * max_nodes];

    for i in 1..=n_tax {
        for j in 1..=n_tax {
            mat[i * stride + j] = if let Some(ref p) = perm {
                dist[(p[i - 1], p[j - 1])]
            } else {
                dist[(i - 1, j - 1)]
            };
        }
    }

    let mut nodes = vec![NetNode::default(); max_nodes];
    nodes[0] = NetNode::new(0);

    // Insert singletons in reverse so final list is 1..=n in ascending order
    for id in (1..=n_tax).rev() {
        nodes[id] = NetNode::new(id);
        nodes[id].next = nodes[0].next;
        nodes[0].next = Some(id);
    }
    // set prev pointers (first active's prev = header 0)
    {
        let mut t = 0usize;
        while let Some(nxt) = nodes[t].next {
            nodes[nxt].prev = Some(t);
            t = nxt;
        }
    }

    debug!("Working matrix allocated: {}x{}", stride, stride);
    // 3) Agglomerate
    let joins = join_nodes(&mut mat, stride, &mut nodes, 0, n_tax, sx_mode)?;

    // 4) Expand joins to a circular ordering of leaves
    let mut cycle = expand_nodes(n_tax, &mut nodes, 0, joins)?;

    // Map sorted indices back to original indices through the permutation
    if let Some(ref p) = perm {
        for i in 1..cycle.len() {
            cycle[i] = p[cycle[i] - 1] + 1; // sorted 1-based → original 1-based
        }
    }
    Ok(cycle)
}

/// Compute a canonical permutation of taxa for deterministic ordering.
/// Returns `perm` where `perm[i]` is the original 0-based index of the taxon
/// that should appear at sorted position `i`.
///
/// Sort criteria:
///   1. Row sum (ascending)
///   2. Sorted row values (lexicographic ascending) — for tiebreaking
fn canonical_permutation(dist: &Array2<f64>) -> Vec<usize> {
    let n = dist.nrows();
    let row_sums: Vec<f64> = (0..n).map(|i| dist.row(i).sum()).collect();

    // Pre-sort each row's values for deterministic tiebreaking
    let sorted_rows: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = dist.row(i).to_vec();
            row.sort_unstable_by(|a, b| a.total_cmp(b));
            row
        })
        .collect();

    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&a, &b| {
        row_sums[a]
            .total_cmp(&row_sums[b])
            .then_with(|| {
                sorted_rows[a]
                    .iter()
                    .zip(sorted_rows[b].iter())
                    .find_map(|(va, vb)| {
                        let ord = va.total_cmp(vb);
                        if ord.is_eq() {
                            None
                        } else {
                            Some(ord)
                        }
                    })
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    perm
}

fn default_sx_mode(n_tax: usize) -> SxMode {
    if n_tax < 300 || rayon::current_num_threads() <= 1 {
        SxMode::Serial
    } else {
        SxMode::Parallel
    }
}

/* ---------------------------- core structures ---------------------------- */

#[derive(Clone, Debug)]
struct NetNode {
    id: usize,
    // active list links
    next: Option<usize>,
    prev: Option<usize>,
    // cluster partner (None if isolated)
    nbr: Option<usize>,
    // children for expansions (set on join3/4)
    ch1: Option<usize>,
    ch2: Option<usize>,
    // work accumulators
    sx: f64,
    rx: f64,
}
impl Default for NetNode {
    fn default() -> Self {
        Self {
            id: 0,
            next: None,
            prev: None,
            nbr: None,
            ch1: None,
            ch2: None,
            sx: 0.0,
            rx: 0.0,
        }
    }
}
impl NetNode {
    fn new(id: usize) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }
}

/* ---------------------------- agglomeration ----------------------------- */

fn join_nodes(
    d: &mut [f64],
    stride: usize,
    nodes: &mut [NetNode],
    head: usize,
    n_tax: usize,
    sx_mode: SxMode,
) -> Result<Vec<usize>> {
    let mut joins: Vec<usize> = Vec::new();

    let mut num_nodes = n_tax;
    let mut num_active = n_tax;
    let mut num_clusters = n_tax;
    let ordering_started = Instant::now();
    let mut iters = 0usize;
    let mut next_progress_report = n_tax.saturating_sub(250);

    // Compute Sx once, then maintain it incrementally as clusters are agglomerated.
    match sx_mode {
        SxMode::Serial => compute_sx_serial(d, stride, nodes, head),
        SxMode::Parallel => compute_sx_parallel(d, stride, nodes, head),
    }

    // Reusable buffers across loop iterations
    let mut reps_buf = Vec::with_capacity(n_tax);
    let mut updates_buf: Vec<(usize, f64)> = Vec::with_capacity(n_tax);

    while num_active > 3 {
        iters += 1;
        // Special case: 4 active and 2 clusters
        if num_active == 4 && num_clusters == 2 {
            let actives = snapshot_active(nodes, head);
            let p = actives[0];
            let q = if Some(actives[1]) != nodes[p].nbr {
                actives[1]
            } else {
                actives[2]
            };
            let pn = nodes[p].nbr.context("expected partner (pn)")?;
            let qn = nodes[q].nbr.context("expected partner (qn)")?;
            let lhs =
                d[nodes[p].id * stride + nodes[q].id] + d[nodes[pn].id * stride + nodes[qn].id];
            let rhs =
                d[nodes[p].id * stride + nodes[qn].id] + d[nodes[pn].id * stride + nodes[q].id];
            if lhs < rhs {
                join3way(p, q, qn, &mut joins, d, stride, nodes, head, &mut num_nodes)?;
            } else {
                join3way(p, qn, q, &mut joins, d, stride, nodes, head, &mut num_nodes)?;
            }
            break;
        }

        fill_cluster_representatives(nodes, head, &mut reps_buf);

        // Precompute max Sx for pruning bounds
        let max_sx = reps_buf
            .iter()
            .map(|&r| nodes[r].sx)
            .fold(f64::NEG_INFINITY, f64::max);

        // --- Choose representatives ---
        let (cx, cy, _bestq) = {
            let mut c_x: Option<usize> = None;
            let mut c_y: Option<usize> = None;
            let mut best = 0.0_f64; // Java seeds to 0. First candidate sets it.
            let mut best_leafs: u8 = 0;

            for i in 0..reps_buf.len() {
                let p = reps_buf[i];
                // Outer bound: if even pairing with the highest-Sx rep can't beat best, skip
                if c_x.is_some() {
                    let bound = -(nodes[p].sx + max_sx);
                    if bound > best + EPS {
                        continue;
                    }
                }
                for &q in &reps_buf[..i] {
                    // Inner bound: check if this pair can possibly beat best
                    if c_x.is_some() {
                        let bound = -(nodes[p].sx + nodes[q].sx);
                        if bound > best + EPS {
                            continue;
                        }
                    }
                    let dpq = avg_cluster_dist(d, stride, nodes, p, q);
                    let qpq = (num_clusters as f64 - 2.0) * dpq - nodes[p].sx - nodes[q].sx;
                    // System.out.debug("\t"+"["+p.id+","+q.id+"] \t = \t "+Qpq);
                    if num_clusters <= 40 {
                        debug!("\t[{},{}] \t = \t {}", nodes[p].id, nodes[q].id, qpq);
                    }

                    if c_x.is_none() || fuzzy_lt(qpq, best) {
                        c_x = Some(p);
                        c_y = Some(q);
                        best = qpq;
                        best_leafs = leaf_count_pair(p, q, nodes, n_tax);
                    } else if fuzzy_eq(qpq, best)
                        && better_tie_pair(p, q, c_x.unwrap(), c_y.unwrap(), nodes)
                    {
                        let leafs = leaf_count_pair(p, q, nodes, n_tax);
                        if leafs > best_leafs {
                            c_x = Some(p);
                            c_y = Some(q);
                            best_leafs = leafs;
                        }
                    }
                }
            }

            (
                c_x.context("failed selecting Cx")?,
                c_y.context("failed selecting Cy")?,
                best,
            )
        };

        debug!("Cx {} Cy {} bestq {}", cx, cy, _bestq);

        // --- Rx for candidates (if needed) ---
        if nodes[cx].nbr.is_some() || nodes[cy].nbr.is_some() {
            compute_rx_candidates(cx, cy, d, stride, nodes, head);
        } else {
            nodes[cx].rx = 0.0;
            nodes[cy].rx = 0.0;
        }

        // --- Pick x,y among candidates ---
        let mut x = cx;
        let mut y = cy;

        let mut m = num_clusters;
        if nodes[cx].nbr.is_some() {
            m += 1;
        }
        if nodes[cy].nbr.is_some() {
            m += 1;
        }

        let mut best_q = (m as f64 - 2.0) * d[nodes[cx].id * stride + nodes[cy].id]
            - nodes[cx].rx
            - nodes[cy].rx;

        if let Some(cxb) = nodes[cx].nbr {
            let qv = (m as f64 - 2.0) * d[nodes[cxb].id * stride + nodes[cy].id]
                - nodes[cxb].rx
                - nodes[cy].rx;
            if fuzzy_lt(qv, best_q) {
                best_q = qv;
                x = cxb;
                y = cy;
            }
        }
        if let Some(cyb) = nodes[cy].nbr {
            let qv = (m as f64 - 2.0) * d[nodes[cx].id * stride + nodes[cyb].id]
                - nodes[cx].rx
                - nodes[cyb].rx;
            if fuzzy_lt(qv, best_q) {
                best_q = qv;
                x = cx;
                y = cyb;
            }
        }
        if let (Some(cxb), Some(cyb)) = (nodes[cx].nbr, nodes[cy].nbr) {
            let qv = (m as f64 - 2.0) * d[nodes[cxb].id * stride + nodes[cyb].id]
                - nodes[cxb].rx
                - nodes[cyb].rx;
            if fuzzy_lt(qv, best_q) {
                x = cxb;
                y = cyb;
            }
        }

        // --- Agglomeration ---
        match (nodes[x].nbr, nodes[y].nbr, num_active) {
            (None, None, _) => {
                // 2-way
                debug!("Join 2 way: x {} y {} num_active {}", x, y, num_active);
                join2way(nodes, x, y);
                let new_rep = cluster_rep(x, nodes);
                update_sx_after_join2(d, stride, nodes, &reps_buf, x, y, new_rep, &mut updates_buf);
                num_clusters -= 1;
            }
            (None, Some(_), _) => {
                // 3-way (x isolated)
                let rem_a = cluster_rep(x, nodes);
                let rem_b = cluster_rep(y, nodes);
                let y_nbr = nodes[y]
                    .nbr
                    .context("expected y.nbr for 3-way agglomeration")?;
                debug!("Join 3(1) way: x {} y {} num_active {}", x, y, num_active);
                let new_rep = join3way(
                    x,
                    y,
                    y_nbr,
                    &mut joins,
                    d,
                    stride,
                    nodes,
                    head,
                    &mut num_nodes,
                )?;
                update_sx_after_merge(
                    d,
                    stride,
                    nodes,
                    &reps_buf,
                    rem_a,
                    rem_b,
                    new_rep,
                    &mut updates_buf,
                );
                num_nodes += 2;
                num_active -= 1;
                num_clusters -= 1;
            }
            (Some(_), None, _) | (_, _, 4) => {
                // 3-way (y isolated) OR last 4 active
                let x2 = y;
                let y2 = x;
                let rem_a = cluster_rep(x2, nodes);
                let rem_b = cluster_rep(y2, nodes);
                let y2_nbr = nodes[y2]
                    .nbr
                    .context("expected y2.nbr for 3-way agglomeration")?;
                debug!(
                    "Join 3(2) way: x2 {} y2 {} num_active {}",
                    x2, y2, num_active
                );
                let new_rep = join3way(
                    x2,
                    y2,
                    y2_nbr,
                    &mut joins,
                    d,
                    stride,
                    nodes,
                    head,
                    &mut num_nodes,
                )?;
                update_sx_after_merge(
                    d,
                    stride,
                    nodes,
                    &reps_buf,
                    rem_a,
                    rem_b,
                    new_rep,
                    &mut updates_buf,
                );
                num_nodes += 2;
                num_active -= 1;
                num_clusters -= 1;
            }
            (Some(xb), Some(_yb), _) => {
                // 4-way
                let rem_a = cluster_rep(x, nodes);
                let rem_b = cluster_rep(y, nodes);
                let yb = nodes[y]
                    .nbr
                    .context("expected yb.nbr for 4-way agglomeration")?;
                debug!(
                    "Join 4-way: xb {} x {} y {} yb {} num_active {}",
                    xb, x, y, yb, num_active
                );
                let new_rep = join4way(
                    xb,
                    x,
                    y,
                    yb,
                    &mut joins,
                    d,
                    stride,
                    nodes,
                    head,
                    &mut num_nodes,
                )?;
                update_sx_after_merge(
                    d,
                    stride,
                    nodes,
                    &reps_buf,
                    rem_a,
                    rem_b,
                    new_rep,
                    &mut updates_buf,
                );
                num_active -= 2;
                num_clusters -= 1;
            }
        }

        if n_tax >= 1_000 && num_clusters <= next_progress_report {
            info!(
                "SplitsTree4 ordering progress: {} clusters remaining (active {}, iteration {}, elapsed {:.1}s)",
                num_clusters,
                num_active,
                iters,
                ordering_started.elapsed().as_secs_f64()
            );
            next_progress_report = num_clusters.saturating_sub(250);
        }
    }

    info!(
        "SplitsTree4 ordering finished in {:.3}s over {} iterations (sx_mode={:?})",
        ordering_started.elapsed().as_secs_f64(),
        iters,
        sx_mode
    );

    Ok(joins)
}

fn fuzzy_lt(a: f64, b: f64) -> bool {
    (a - b) < -EPS
}
fn fuzzy_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS
}

#[inline]
fn leaf_count_pair(p: usize, q: usize, nodes: &[NetNode], n_tax: usize) -> u8 {
    let a = (nodes[p].id <= n_tax) as u8;
    let b = (nodes[q].id <= n_tax) as u8;
    a + b
}

/// Return true if (p,q) should replace (cx,cy) when qpq ~ best.
/// Use lexicographic taxon-id order as a deterministic tiebreaker.
#[inline]
fn better_tie_pair(p: usize, q: usize, cx: usize, cy: usize, nodes: &[NetNode]) -> bool {
    pair_key(p, q, nodes) < pair_key(cx, cy, nodes)
}

#[inline]
fn pair_key(a: usize, b: usize, nodes: &[NetNode]) -> (usize, usize) {
    let (ia, ib) = (nodes[a].id, nodes[b].id);
    if ia <= ib { (ia, ib) } else { (ib, ia) }
}

#[inline]
fn fill_cluster_representatives(nodes: &[NetNode], head: usize, buf: &mut Vec<usize>) {
    buf.clear();
    let mut p_opt = nodes[head].next;
    while let Some(p) = p_opt {
        if nodes[p].nbr.map_or(true, |nb| nodes[nb].id > nodes[p].id) {
            buf.push(p);
        }
        p_opt = nodes[p].next;
    }
}

#[inline]
fn cluster_rep(p: usize, nodes: &[NetNode]) -> usize {
    match nodes[p].nbr {
        Some(nb) if nodes[nb].id < nodes[p].id => nb,
        _ => p,
    }
}

#[inline]
fn set_cluster_sx(nodes: &mut [NetNode], rep: usize, sx: f64) {
    nodes[rep].sx = sx;
    if let Some(nb) = nodes[rep].nbr {
        nodes[nb].sx = sx;
    }
}

#[inline]
fn avg_cluster_dist_to_singleton(
    mat: &[f64],
    stride: usize,
    nodes: &[NetNode],
    p: usize,
    s: usize,
) -> f64 {
    match nodes[p].nbr {
        None => mat[nodes[p].id * stride + nodes[s].id],
        Some(pb) => {
            0.5 * (mat[nodes[p].id * stride + nodes[s].id]
                + mat[nodes[pb].id * stride + nodes[s].id])
        }
    }
}

fn update_sx_after_join2(
    mat: &[f64],
    stride: usize,
    nodes: &mut [NetNode],
    reps_before: &[usize],
    x: usize,
    y: usize,
    new_rep: usize,
    updates: &mut Vec<(usize, f64)>,
) {
    updates.clear();
    let mut new_sx = 0.0;

    for &k in reps_before {
        if k == x || k == y {
            continue;
        }
        let old = nodes[k].sx;
        let d_kx = avg_cluster_dist_to_singleton(mat, stride, nodes, k, x);
        let d_ky = avg_cluster_dist_to_singleton(mat, stride, nodes, k, y);
        let d_knew = 0.5 * (d_kx + d_ky);
        updates.push((k, old - d_kx - d_ky + d_knew));
        new_sx += d_knew;
    }

    for &(k, sx) in updates.iter() {
        set_cluster_sx(nodes, k, sx);
    }
    set_cluster_sx(nodes, new_rep, new_sx);
}

fn update_sx_after_merge(
    mat: &[f64],
    stride: usize,
    nodes: &mut [NetNode],
    reps_before: &[usize],
    rem_a: usize,
    rem_b: usize,
    new_rep: usize,
    updates: &mut Vec<(usize, f64)>,
) {
    updates.clear();
    let mut new_sx = 0.0;

    for &k in reps_before {
        if k == rem_a || k == rem_b {
            continue;
        }
        let old = nodes[k].sx;
        let d_ka = avg_cluster_dist(mat, stride, nodes, k, rem_a);
        let d_kb = avg_cluster_dist(mat, stride, nodes, k, rem_b);
        let d_knew = avg_cluster_dist(mat, stride, nodes, k, new_rep);
        updates.push((k, old - d_ka - d_kb + d_knew));
        new_sx += d_knew;
    }

    for &(k, sx) in updates.iter() {
        set_cluster_sx(nodes, k, sx);
    }
    set_cluster_sx(nodes, new_rep, new_sx);
}

/* ---------------------------- join primitives --------------------------- */

fn join2way(nodes: &mut [NetNode], x: usize, y: usize) {
    nodes[x].nbr = Some(y);
    nodes[y].nbr = Some(x);
}

/// Returns the new node `u` (and pushes it onto `joins`)
fn join3way(
    x: usize,
    y: usize,
    z: usize,
    joins: &mut Vec<usize>,
    mat: &mut [f64],
    stride: usize,
    nodes: &mut [NetNode],
    head: usize,
    num_nodes: &mut usize, // current max id; this function *does not* increment it (call-site does)
) -> Result<usize> {
    let u = *num_nodes + 1;
    let v = *num_nodes + 2;

    ensure!(u < nodes.len() && v < nodes.len(), "node capacity exceeded");

    nodes[u] = NetNode {
        id: u,
        ..Default::default()
    };
    nodes[v] = NetNode {
        id: v,
        ..Default::default()
    };

    nodes[u].ch1 = Some(x);
    nodes[u].ch2 = Some(y);
    nodes[v].ch1 = Some(y);
    nodes[v].ch2 = Some(z);
    nodes[u].nbr = Some(v);
    nodes[v].nbr = Some(u);

    // Replace x by u in the linked list
    nodes[u].next = nodes[x].next;
    nodes[u].prev = nodes[x].prev;
    if let Some(nx) = nodes[u].next {
        nodes[nx].prev = Some(u);
    }
    if let Some(px) = nodes[u].prev {
        nodes[px].next = Some(u);
    }

    // Replace z by v in the linked list
    nodes[v].next = nodes[z].next;
    nodes[v].prev = nodes[z].prev;
    if let Some(nz) = nodes[v].next {
        nodes[nz].prev = Some(v);
    }
    if let Some(pz) = nodes[v].prev {
        nodes[pz].next = Some(v);
    }

    // Remove y from the linked list
    if let Some(ny) = nodes[y].next {
        nodes[ny].prev = nodes[y].prev;
    }
    if let Some(py) = nodes[y].prev {
        nodes[py].next = nodes[y].next;
    }

    // --- Update distances exactly like the Java code ---
    {
        let xid = nodes[x].id;
        let yid = nodes[y].id;
        let zid = nodes[z].id;

        let mut p_opt = nodes[head].next;
        while let Some(p) = p_opt {
            let pid = nodes[p].id;

            mat[u * stride + pid] =
                (2.0 / 3.0) * mat[xid * stride + pid] + (1.0 / 3.0) * mat[yid * stride + pid];
            mat[pid * stride + u] = mat[u * stride + pid];

            mat[v * stride + pid] =
                (2.0 / 3.0) * mat[zid * stride + pid] + (1.0 / 3.0) * mat[yid * stride + pid];
            mat[pid * stride + v] = mat[v * stride + pid];

            p_opt = nodes[p].next;
        }
        mat[u * stride + u] = 0.0;
        mat[v * stride + v] = 0.0;
    }

    joins.push(u);
    Ok(u)
}

fn join4way(
    x2: usize,
    x: usize,
    y: usize,
    y2: usize,
    joins: &mut Vec<usize>,
    mat: &mut [f64],
    stride: usize,
    nodes: &mut [NetNode],
    head: usize,
    num_nodes: &mut usize,
) -> Result<usize> {
    // First 3-way
    let u = join3way(x2, x, y, joins, mat, stride, nodes, head, num_nodes)?;
    *num_nodes += 2;
    // Second 3-way
    let final_u = join3way(
        u,
        nodes[u].nbr.context("u.nbr")?,
        y2,
        joins,
        mat,
        stride,
        nodes,
        head,
        num_nodes,
    )?;
    *num_nodes += 2;
    Ok(final_u)
}

/* ---------------------------- scoring helpers --------------------------- */

fn avg_cluster_dist(mat: &[f64], stride: usize, nodes: &[NetNode], p: usize, q: usize) -> f64 {
    match (nodes[p].nbr, nodes[q].nbr) {
        (None, None) => mat[nodes[p].id * stride + nodes[q].id],
        (Some(pb), None) => {
            0.5 * (mat[nodes[p].id * stride + nodes[q].id]
                + mat[nodes[pb].id * stride + nodes[q].id])
        }
        (None, Some(qb)) => {
            0.5 * (mat[nodes[p].id * stride + nodes[q].id]
                + mat[nodes[p].id * stride + nodes[qb].id])
        }
        (Some(pb), Some(qb)) => {
            0.25 * (mat[nodes[p].id * stride + nodes[q].id]
                + mat[nodes[p].id * stride + nodes[qb].id]
                + mat[nodes[pb].id * stride + nodes[q].id]
                + mat[nodes[pb].id * stride + nodes[qb].id])
        }
    }
}

fn compute_sx_serial(d: &[f64], stride: usize, nodes: &mut [NetNode], head: usize) {
    // zero Sx for all actives
    let mut p_opt = nodes[head].next;
    while let Some(p) = p_opt {
        nodes[p].sx = 0.0;
        p_opt = nodes[p].next;
    }

    // p walks active list
    let mut p_opt = nodes[head].next;
    while let Some(p) = p_opt {
        // evaluate only one per cluster: (p.nbr == null) || (p.nbr.id > p.id)
        let eval_p = nodes[p].nbr.map_or(true, |nb| nodes[nb].id > nodes[p].id);
        if eval_p {
            // q walks from p.next forward
            let mut q_opt = nodes[p].next;
            while let Some(q) = q_opt {
                // Java: if (q.nbr == null || ((q.nbr.id > q.id) && (q.nbr != p)))
                let eval_q = match nodes[q].nbr {
                    None => true,
                    Some(nb) => nodes[nb].id > nodes[q].id && nodes[q].nbr != Some(p),
                };
                if eval_q {
                    let dpq = avg_cluster_dist(d, stride, nodes, p, q);
                    nodes[p].sx += dpq;
                    if let Some(pb) = nodes[p].nbr {
                        nodes[pb].sx += dpq;
                    }
                    nodes[q].sx += dpq;
                    if let Some(qb) = nodes[q].nbr {
                        nodes[qb].sx += dpq;
                    }
                }
                q_opt = nodes[q].next;
            }
        }
        p_opt = nodes[p].next;
    }
}

fn compute_sx_parallel(d: &[f64], stride: usize, nodes: &mut [NetNode], head: usize) {
    let mut actives = Vec::new();
    let mut p_opt = nodes[head].next;
    while let Some(p) = p_opt {
        nodes[p].sx = 0.0;
        actives.push(p);
        p_opt = nodes[p].next;
    }

    let nodes_ro = &*nodes;
    let eligible: Vec<bool> = actives
        .iter()
        .map(|&p| {
            nodes_ro[p]
                .nbr
                .map_or(true, |nb| nodes_ro[nb].id > nodes_ro[p].id)
        })
        .collect();

    let deltas = (0..actives.len())
        .into_par_iter()
        .fold(
            || vec![0.0f64; nodes_ro.len()],
            |mut acc, p_idx| {
                if !eligible[p_idx] {
                    return acc;
                }
                let p = actives[p_idx];
                for q_idx in (p_idx + 1)..actives.len() {
                    let q = actives[q_idx];
                    let eval_q = match nodes_ro[q].nbr {
                        None => true,
                        Some(nb) => nodes_ro[nb].id > nodes_ro[q].id && nodes_ro[q].nbr != Some(p),
                    };
                    if !eval_q {
                        continue;
                    }
                    let dpq = avg_cluster_dist(d, stride, nodes_ro, p, q);
                    acc[p] += dpq;
                    if let Some(pb) = nodes_ro[p].nbr {
                        acc[pb] += dpq;
                    }
                    acc[q] += dpq;
                    if let Some(qb) = nodes_ro[q].nbr {
                        acc[qb] += dpq;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; nodes_ro.len()],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );

    for p in actives {
        nodes[p].sx = deltas[p];
    }
}

fn compute_rx_candidates(
    cx: usize,
    cy: usize,
    mat: &[f64],
    stride: usize,
    nodes: &mut [NetNode],
    head: usize,
) {
    let mut targets = Vec::with_capacity(4);
    targets.push(cx);
    if let Some(cxb) = nodes[cx].nbr {
        if !targets.contains(&cxb) {
            targets.push(cxb);
        }
    }
    if !targets.contains(&cy) {
        targets.push(cy);
    }
    if let Some(cyb) = nodes[cy].nbr {
        if !targets.contains(&cyb) {
            targets.push(cyb);
        }
    }

    let mut sums = vec![0.0f64; targets.len()];
    let mut p_opt = nodes[head].next;
    while let Some(p) = p_opt {
        let cond = p == cx
            || nodes[cx].nbr == Some(p)
            || p == cy
            || nodes[cy].nbr == Some(p)
            || nodes[p].nbr.is_none();
        for (i, &z) in targets.iter().enumerate() {
            let term = mat[nodes[z].id * stride + nodes[p].id];
            sums[i] += if cond { term } else { term / 2.0 };
        }
        p_opt = nodes[p].next;
    }
    for (i, &z) in targets.iter().enumerate() {
        nodes[z].rx = sums[i];
    }
}

/* ---------------------------- expansion phase --------------------------- */

fn are_adjacent(nodes: &[NetNode], u: usize, v: usize) -> Option<bool> {
    if nodes[u].next == Some(v) {
        Some(true)
    } else if nodes[v].next == Some(u) {
        Some(false)
    } else {
        None
    }
}

fn next_leaf_in_dir(nodes: &[NetNode], start: usize, forward: bool, n_tax: usize) -> Result<usize> {
    let mut a = start;
    loop {
        a = if forward {
            nodes[a]
                .next
                .context("ring broken while seeking next leaf (forward)")?
        } else {
            nodes[a]
                .prev
                .context("ring broken while seeking next leaf (backward)")?
        };
        let id = nodes[a].id;
        if (1..=n_tax).contains(&id) {
            return Ok(id);
        }
        // guard against infinite loop on malformed rings
        ensure!(a != start, "looped around without finding a leaf");
    }
}

fn expand_nodes(
    n_tax: usize,
    nodes: &mut [NetNode],
    head: usize,
    mut joins: Vec<usize>,
) -> Result<Vec<usize>> {
    ensure!(head == 0, "head must be 0");
    ensure!(n_tax >= 3, "need at least 3 taxa");
    ensure!(!joins.is_empty(), "joins stack is empty");

    // Seed 3-cycle from first three actives (like Java)
    let x0 = nodes[head].next.context("need ≥3 actives (x0)")?;
    let y0 = nodes[x0].next.context("need ≥3 actives (y0)")?;
    let z0 = nodes[y0].next.context("need ≥3 actives (z0)")?;

    nodes[z0].next = Some(x0);
    nodes[x0].prev = Some(z0);

    // Expand joins, LIFO
    while let Some(mut u) = joins.pop() {
        let mut v = nodes[u].nbr.context("u.nbr missing")?;
        let mut x1 = nodes[u].ch1.context("u.ch1 missing")?;
        let y1 = nodes[u].ch2.context("u.ch2 missing")?;
        let mut z1 = nodes[v].ch2.context("v.ch2 missing")?;

        // Make sure (u,v) are consecutive in the *current* ring
        match are_adjacent(nodes, u, v) {
            Some(true) => Ok(()),
            Some(false) => {
                // v -> u: swap roles and outer ends (x ↔ z)
                std::mem::swap(&mut u, &mut v);
                std::mem::swap(&mut x1, &mut z1);
                Ok(())
            }
            None => Err(anyhow!(
                "Join expansion invariant broken: u={} and v={} are not adjacent in the ring",
                u,
                v
            )),
        }?;

        // Anchors must exist
        let uprev = nodes[u].prev.context("u.prev missing")?;
        let vnext = nodes[v].next.context("v.next missing")?;

        // splice: uprev -> x1 -> y1 -> z1 -> vnext
        nodes[x1].prev = Some(uprev);
        nodes[uprev].next = Some(x1);

        nodes[x1].next = Some(y1);
        nodes[y1].prev = Some(x1);

        nodes[y1].next = Some(z1);
        nodes[z1].prev = Some(y1);

        nodes[z1].next = Some(vnext);
        nodes[vnext].prev = Some(z1);
    }

    // Find leaf 1
    let start = {
        let head_next = nodes[head].next.context("no active nodes")?;
        let mut cur = head_next;
        loop {
            if nodes[cur].id == 1 {
                break cur;
            }
            cur = nodes[cur]
                .next
                .context("broken ring while seeking leaf 1")?;
            ensure!(cur != head_next, "leaf 1 not found in ring");
        }
    };

    // Canonicalize orientation: pick the direction from 1 whose *next leaf* is smaller
    let next_leaf_fwd = next_leaf_in_dir(nodes, start, true, n_tax)?;
    let next_leaf_bwd = next_leaf_in_dir(nodes, start, false, n_tax)?;
    let forward = next_leaf_fwd <= next_leaf_bwd;

    // Extract n_tax leaves walking in chosen direction
    let mut cycle = vec![0usize];
    let mut a = start;
    let mut seen = 0usize;
    loop {
        let id = nodes[a].id;
        if (1..=n_tax).contains(&id) {
            cycle.push(id);
            seen += 1;
            if seen == n_tax {
                break;
            }
        }
        a = if forward {
            nodes[a]
                .next
                .context("ring broken during extraction (forward)")?
        } else {
            nodes[a]
                .prev
                .context("ring broken during extraction (backward)")?
        };
    }
    Ok(cycle)
}

/* ------------------------------ utilities ------------------------------- */

/// Take a one-time snapshot of the current active list (in list order).
fn snapshot_active(nodes: &[NetNode], head: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut cur = nodes[head].next;
    while let Some(p) = cur {
        out.push(p);
        cur = nodes[p].next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn small_triangle() {
        // 3 taxa — returns [0,1,2,3]
        let d = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);
        let ord = compute_order_splits_tree4(&d, false)
            .context("computing order for small triangle")
            .unwrap();
        assert_eq!(ord, vec![0, 1, 2, 3]);
    }

    #[test]
    fn small_square() {
        // 4 taxa — returns [0,1,2,3,4]
        // a, b, c, d
        // 0.0,  1.0,  2.0,  3.0
        // 1.0,  0.0,  1.5,  2.5
        // 2.0,  1.5,  0.0,  1.5
        // 3.0,  2.5,  1.5,  0.0
        let d = arr2(&[
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.5],
            [2.0, 1.5, 0.0, 1.5],
            [3.0, 2.5, 1.5, 0.0],
        ]);
        let ord = compute_order_splits_tree4(&d, false)
            .context("computing order for small square")
            .unwrap();
        assert_eq!(ord, vec![0, 1, 2, 4, 3]);
    }

    #[test]
    fn smoke_5_1() {
        // 5 taxa — returns [0,1,2,5,4,3]
        // a, b, c, d, e
        // 0.0,  5.0,  9.0,  9.0,  8.0
        // 5.0,  0.0,  10.0, 10.0, 9.0
        // 9.0,  10.0, 0.0,  8.0,  7.0
        // 9.0,  10.0, 8.0,  0.0,  3.0
        // 8.0,  9.0,  7.0,  3.0,  0.0

        let d = arr2(&[
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ]);
        let ord = compute_order_splits_tree4(&d, false)
            .context("computing order for smoke 5_1")
            .unwrap();
        assert_eq!(ord, vec![0, 1, 2, 5, 4, 3]);
    }

    #[test]
    fn smoke_5_2() {
        // 5 taxa — returns [0,1,2,4,5,3]
        // a,b,c,d,e
        // 0.0,2.0,3.0,4.0,5.0
        // 2.0,0.0,6.0,7.0,8.0
        // 3.0,6.0,0.0,9.0,1.0
        // 4.0,7.0,9.0,0.0,2.0
        // 5.0,8.0,1.0,2.0,0.0

        let d = arr2(&[
            [0.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 0.0, 6.0, 7.0, 8.0],
            [3.0, 6.0, 0.0, 9.0, 1.0],
            [4.0, 7.0, 9.0, 0.0, 2.0],
            [5.0, 8.0, 1.0, 2.0, 0.0],
        ]);
        let ord = compute_order_splits_tree4(&d, false)
            .context("computing order for smoke 5_2")
            .unwrap();
        assert_eq!(ord, vec![0, 1, 2, 4, 5, 3]);
    }

    #[test]
    fn smoke_10_1() {
        // 10 taxa - returns [0, 1, 2, 10, 6, 4, 8, 3, 9, 7, 5]
        // a,b,c,d,e,f,g,h,i,j
        // 0.0,5.0,12.0,7.0,3.0,9.0,11.0,6.0,4.0,10.0
        // 5.0,0.0,8.0,2.0,14.0,5.0,13.0,7.0,12.0,1.0
        // 12.0,8.0,0.0,4.0,9.0,3.0,8.0,2.0,5.0,6.0
        // 7.0,2.0,4.0,0.0,11.0,7.0,10.0,4.0,6.0,9.0
        // 3.0,14.0,9.0,11.0,0.0,8.0,1.0,13.0,2.0,7.0
        // 9.0,5.0,3.0,7.0,8.0,0.0,12.0,5.0,3.0,4.0
        // 11.0,13.0,8.0,10.0,1.0,12.0,0.0,6.0,2.0,8.0
        // 6.0,7.0,2.0,4.0,13.0,5.0,6.0,0.0,9.0,7.0
        // 4.0,12.0,5.0,6.0,2.0,3.0,2.0,9.0,0.0,5.0
        // 10.0,1.0,6.0,9.0,7.0,4.0,8.0,7.0,5.0,0.0

        let d = arr2(&[
            [0.0, 5.0, 12.0, 7.0, 3.0, 9.0, 11.0, 6.0, 4.0, 10.0],
            [5.0, 0.0, 8.0, 2.0, 14.0, 5.0, 13.0, 7.0, 12.0, 1.0],
            [12.0, 8.0, 0.0, 4.0, 9.0, 3.0, 8.0, 2.0, 5.0, 6.0],
            [7.0, 2.0, 4.0, 0.0, 11.0, 7.0, 10.0, 4.0, 6.0, 9.0],
            [3.0, 14.0, 9.0, 11.0, 0.0, 8.0, 1.0, 13.0, 2.0, 7.0],
            [9.0, 5.0, 3.0, 7.0, 8.0, 0.0, 12.0, 5.0, 3.0, 4.0],
            [11.0, 13.0, 8.0, 10.0, 1.0, 12.0, 0.0, 6.0, 2.0, 8.0],
            [6.0, 7.0, 2.0, 4.0, 13.0, 5.0, 6.0, 0.0, 9.0, 7.0],
            [4.0, 12.0, 5.0, 6.0, 2.0, 3.0, 2.0, 9.0, 0.0, 5.0],
            [10.0, 1.0, 6.0, 9.0, 7.0, 4.0, 8.0, 7.0, 5.0, 0.0],
        ]);

        let ord = compute_order_splits_tree4(&d, false)
            .context("computing order for smoke 10_1")
            .unwrap();
        assert_eq!(ord, vec![0, 1, 2, 10, 6, 4, 8, 3, 9, 7, 5]);
    }

    #[test]
    fn smoke_10_2() {
        // 10 taxa - [1 2 4 8 3 7 9 5 6 10]
        // a,b,c,d,e,f,g,h,i,j
        // 0.0,3.0,7.0,4.0,5.0,9.0,2.0,8.0,6.0,1.0
        // 3.0,0.0,5.0,2.0,10.0,4.0,11.0,7.0,9.0,8.0
        // 7.0,5.0,0.0,6.0,3.0,8.0,4.0,2.0,5.0,9.0
        // 4.0,2.0,6.0,0.0,7.0,5.0,9.0,3.0,4.0,6.0
        // 5.0,10.0,3.0,7.0,0.0,2.0,6.0,12.0,1.0,5.0
        // 9.0,4.0,8.0,5.0,2.0,0.0,7.0,4.0,3.0,2.0
        // 2.0,11.0,4.0,9.0,6.0,7.0,0.0,5.0,2.0,8.0
        // 8.0,7.0,2.0,3.0,12.0,4.0,5.0,0.0,6.0,7.0
        // 6.0,9.0,5.0,4.0,1.0,3.0,2.0,6.0,0.0,4.0
        // 1.0,8.0,9.0,6.0,5.0,2.0,8.0,7.0,4.0,0.0

        let d = arr2(&[
            [0.0, 3.0, 7.0, 4.0, 5.0, 9.0, 2.0, 8.0, 6.0, 1.0],
            [3.0, 0.0, 5.0, 2.0, 10.0, 4.0, 11.0, 7.0, 9.0, 8.0],
            [7.0, 5.0, 0.0, 6.0, 3.0, 8.0, 4.0, 2.0, 5.0, 9.0],
            [4.0, 2.0, 6.0, 0.0, 7.0, 5.0, 9.0, 3.0, 4.0, 6.0],
            [5.0, 10.0, 3.0, 7.0, 0.0, 2.0, 6.0, 12.0, 1.0, 5.0],
            [9.0, 4.0, 8.0, 5.0, 2.0, 0.0, 7.0, 4.0, 3.0, 2.0],
            [2.0, 11.0, 4.0, 9.0, 6.0, 7.0, 0.0, 5.0, 2.0, 8.0],
            [8.0, 7.0, 2.0, 3.0, 12.0, 4.0, 5.0, 0.0, 6.0, 7.0],
            [6.0, 9.0, 5.0, 4.0, 1.0, 3.0, 2.0, 6.0, 0.0, 4.0],
            [1.0, 8.0, 9.0, 6.0, 5.0, 2.0, 8.0, 7.0, 4.0, 0.0],
        ]);

        let ord = compute_order_splits_tree4(&d, false)
            .context("computing order for smoke 10_2")
            .unwrap();
        assert_eq!(ord, vec![0, 1, 2, 4, 8, 3, 7, 9, 5, 6, 10]);
    }

    #[test]
    fn smoke_15_1() {
        // 15 taxa - [0, 1 4 15 11 10 12 14 2 3 7 8 13 5 9 6]
        // a,b,c,d,e,f,g,h,i,j,k,l,m,n,o
        // 0.0,3.0,9.0,2.0,8.0,1.0,7.0,13.0,6.0,12.0,5.0,11.0,4.0,10.0,3.0
        // 3.0,0.0,2.0,9.0,3.0,10.0,4.0,11.0,5.0,12.0,6.0,13.0,7.0,1.0,8.0
        // 9.0,2.0,0.0,3.0,11.0,6.0,1.0,9.0,4.0,12.0,7.0,2.0,10.0,5.0,13.0
        // 2.0,9.0,3.0,0.0,6.0,2.0,11.0,7.0,3.0,12.0,8.0,4.0,13.0,9.0,5.0
        // 8.0,3.0,11.0,6.0,0.0,11.0,8.0,5.0,2.0,12.0,9.0,6.0,3.0,13.0,10.0
        // 1.0,10.0,6.0,2.0,11.0,0.0,5.0,3.0,1.0,12.0,10.0,8.0,6.0,4.0,2.0
        // 7.0,4.0,1.0,11.0,8.0,5.0,0.0,1.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0
        // 13.0,11.0,9.0,7.0,5.0,3.0,1.0,0.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0
        // 6.0,5.0,4.0,3.0,2.0,1.0,13.0,12.0,0.0,12.0,13.0,1.0,2.0,3.0,4.0
        // 12.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0,12.0,0.0,1.0,3.0,5.0,7.0,9.0
        // 5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,1.0,0.0,5.0,8.0,11.0,1.0
        // 11.0,13.0,2.0,4.0,6.0,8.0,10.0,12.0,1.0,3.0,5.0,0.0,11.0,2.0,6.0
        // 4.0,7.0,10.0,13.0,3.0,6.0,9.0,12.0,2.0,5.0,8.0,11.0,0.0,6.0,11.0
        // 10.0,1.0,5.0,9.0,13.0,4.0,8.0,12.0,3.0,7.0,11.0,2.0,6.0,0.0,3.0
        // 3.0,8.0,13.0,5.0,10.0,2.0,7.0,12.0,4.0,9.0,1.0,6.0,11.0,3.0,0.0

        let d = arr2(&[
            [
                0.0, 3.0, 9.0, 2.0, 8.0, 1.0, 7.0, 13.0, 6.0, 12.0, 5.0, 11.0, 4.0, 10.0, 3.0,
            ],
            [
                3.0, 0.0, 2.0, 9.0, 3.0, 10.0, 4.0, 11.0, 5.0, 12.0, 6.0, 13.0, 7.0, 1.0, 8.0,
            ],
            [
                9.0, 2.0, 0.0, 3.0, 11.0, 6.0, 1.0, 9.0, 4.0, 12.0, 7.0, 2.0, 10.0, 5.0, 13.0,
            ],
            [
                2.0, 9.0, 3.0, 0.0, 6.0, 2.0, 11.0, 7.0, 3.0, 12.0, 8.0, 4.0, 13.0, 9.0, 5.0,
            ],
            [
                8.0, 3.0, 11.0, 6.0, 0.0, 11.0, 8.0, 5.0, 2.0, 12.0, 9.0, 6.0, 3.0, 13.0, 10.0,
            ],
            [
                1.0, 10.0, 6.0, 2.0, 11.0, 0.0, 5.0, 3.0, 1.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0,
            ],
            [
                7.0, 4.0, 1.0, 11.0, 8.0, 5.0, 0.0, 1.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0,
            ],
            [
                13.0, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0, 0.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0,
            ],
            [
                6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 13.0, 12.0, 0.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0,
            ],
            [
                12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0,
            ],
            [
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 1.0, 0.0, 5.0, 8.0, 11.0, 1.0,
            ],
            [
                11.0, 13.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 1.0, 3.0, 5.0, 0.0, 11.0, 2.0, 6.0,
            ],
            [
                4.0, 7.0, 10.0, 13.0, 3.0, 6.0, 9.0, 12.0, 2.0, 5.0, 8.0, 11.0, 0.0, 6.0, 11.0,
            ],
            [
                10.0, 1.0, 5.0, 9.0, 13.0, 4.0, 8.0, 12.0, 3.0, 7.0, 11.0, 2.0, 6.0, 0.0, 3.0,
            ],
            [
                3.0, 8.0, 13.0, 5.0, 10.0, 2.0, 7.0, 12.0, 4.0, 9.0, 1.0, 6.0, 11.0, 3.0, 0.0,
            ],
        ]);

        let ord = compute_order_splits_tree4(&d, false)
            .context("computing order for smoke 15_1")
            .unwrap();
        assert_eq!(
            ord,
            vec![0, 1, 4, 15, 11, 10, 12, 14, 2, 3, 7, 8, 13, 5, 9, 6]
        );
    }

    #[test]
    fn permutation_invariance() {
        // The smoke_10_1 matrix
        let d = arr2(&[
            [0.0, 5.0, 12.0, 7.0, 3.0, 9.0, 11.0, 6.0, 4.0, 10.0],
            [5.0, 0.0, 8.0, 2.0, 14.0, 5.0, 13.0, 7.0, 12.0, 1.0],
            [12.0, 8.0, 0.0, 4.0, 9.0, 3.0, 8.0, 2.0, 5.0, 6.0],
            [7.0, 2.0, 4.0, 0.0, 11.0, 7.0, 10.0, 4.0, 6.0, 9.0],
            [3.0, 14.0, 9.0, 11.0, 0.0, 8.0, 1.0, 13.0, 2.0, 7.0],
            [9.0, 5.0, 3.0, 7.0, 8.0, 0.0, 12.0, 5.0, 3.0, 4.0],
            [11.0, 13.0, 8.0, 10.0, 1.0, 12.0, 0.0, 6.0, 2.0, 8.0],
            [6.0, 7.0, 2.0, 4.0, 13.0, 5.0, 6.0, 0.0, 9.0, 7.0],
            [4.0, 12.0, 5.0, 6.0, 2.0, 3.0, 2.0, 9.0, 0.0, 5.0],
            [10.0, 1.0, 6.0, 9.0, 7.0, 4.0, 8.0, 7.0, 5.0, 0.0],
        ]);
        let baseline = compute_order_splits_tree4(&d, true).unwrap();

        // Apply an arbitrary permutation
        let perm = [3, 7, 0, 5, 9, 1, 4, 8, 2, 6];
        let n = d.nrows();
        let mut d_perm = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d_perm[(i, j)] = d[(perm[i], perm[j])];
            }
        }

        let result_perm = compute_order_splits_tree4(&d_perm, true).unwrap();
        // Map permuted result back to original indices
        let result_mapped: Vec<usize> = result_perm
            .iter()
            .map(|&idx| if idx == 0 { 0 } else { perm[idx - 1] + 1 })
            .collect();

        assert_eq!(baseline, result_mapped);
    }
}

use anyhow::{Context, Result, ensure, anyhow};
use ndarray::Array2;

const EPS: f64 = 1e-12;

/// Compute the NeighborNet circular ordering (Bryant & Huson 2005).
/// - `dist` is 0-based, shape n×n, symmetric with 0 on the diagonal.
/// - Returns a 1-based cycle with a leading 0 sentinel: `[0, t1, t2, ..., tn]`.
pub fn compute_order_splits_tree4(dist: &Array2<f64>) -> Result<Vec<usize>> {
    let n_tax = dist.nrows();
    ensure!(dist.ncols() == n_tax, "Distance matrix must be square");

    if n_tax <= 3 {
        let mut cycle = vec![0usize];
        cycle.extend(1..=n_tax);
        return Ok(cycle);
    }

    let max_nodes = 3 * n_tax - 5;
    let mut mat = vec![vec![0.0_f64; max_nodes]; max_nodes];

    for i in 1..=n_tax {
        for j in 1..=n_tax {
            mat[i][j] = dist[(i - 1, j - 1)];
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
    // set prev pointers (first active’s prev = header 0)
    {
        let mut t = 0usize;
        while let Some(nxt) = nodes[t].next {
            nodes[nxt].prev = Some(t);
            t = nxt;
        }
    }

    // 3) Agglomerate
    let joins = join_nodes(&mut mat, &mut nodes, 0, n_tax)?;

    // 4) Expand joins to a circular ordering of leaves
    expand_nodes(n_tax, &mut nodes, 0, joins)
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
        Self { id, ..Default::default() }
    }
}

/* ---------------------------- agglomeration ----------------------------- */

fn join_nodes(
    d: &mut [Vec<f64>],
    nodes: &mut [NetNode],
    head: usize,
    n_tax: usize,
) -> Result<Vec<usize>> {
    let mut joins: Vec<usize> = Vec::new();

    let mut num_nodes = n_tax;
    let mut num_active = n_tax;
    let mut num_clusters = n_tax;

    while num_active > 3 {
        // Special case: 4 active and 2 clusters
        if num_active == 4 && num_clusters == 2 {
            let actives = snapshot_active(nodes, head);
            let p = actives[0];
            let q = if Some(actives[1]) != nodes[p].nbr { actives[1] } else { actives[2] };
            let pn = nodes[p].nbr.context("expected partner (pn)")?;
            let qn = nodes[q].nbr.context("expected partner (qn)")?;
            let lhs = d[nodes[p].id][nodes[q].id] + d[nodes[pn].id][nodes[qn].id];
            let rhs = d[nodes[p].id][nodes[qn].id] + d[nodes[pn].id][nodes[q].id];
            if lhs < rhs {
                join3way(p, q, qn, &mut joins, d, nodes, head, &mut num_nodes)?;
            } else {
                join3way(p, qn, q, &mut joins, d, nodes, head, &mut num_nodes)?;
            }
            break;
        }

        // --- Compute Sx ---
        {
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
                            let dpq = avg_cluster_dist(d, nodes, p, q);
                            nodes[p].sx += dpq;
                            if let Some(pb) = nodes[p].nbr { nodes[pb].sx += dpq; }
                            nodes[q].sx += dpq;
                            if let Some(qb) = nodes[q].nbr { nodes[qb].sx += dpq; }
                        }
                        q_opt = nodes[q].next;
                    }
                }
                p_opt = nodes[p].next;
            }
        }


        // --- Choose representatives ---
        let (cx, cy, _bestq) = {
            let mut c_x: Option<usize> = None;
            let mut c_y: Option<usize> = None;
            let mut best = 0.0_f64; // Java seeds to 0. First candidate sets it.

            let mut p_opt = nodes[head].next;
            while let Some(p) = p_opt {
                // evaluate only one per cluster (keep smaller id)
                if nodes[p].nbr.map_or(false, |nb| nodes[nb].id < nodes[p].id) {
                    p_opt = nodes[p].next;
                    continue;
                }

                let mut q_opt = nodes[head].next;
                while let Some(q) = q_opt {
                    if q == p { break; }
                    if nodes[q].nbr.map_or(false, |nb| nodes[nb].id < nodes[q].id) {
                        q_opt = nodes[q].next; continue;
                    }
                    if nodes[q].nbr == Some(p) {
                        q_opt = nodes[q].next; continue;
                    }

                    let dpq = avg_cluster_dist(d, nodes, p, q);
                    let qpq = (num_clusters as f64 - 2.0) * dpq - nodes[p].sx - nodes[q].sx;

                    if c_x.is_none()
                        || fuzzy_lt(qpq, best)
                        || (fuzzy_eq(qpq, best) && better_tie_pair(p, q, c_x.unwrap(), c_y.unwrap(), nodes))
                    {
                        c_x = Some(p);
                        c_y = Some(q);
                        best = qpq;
                    }

                    q_opt = nodes[q].next;
                }
                p_opt = nodes[p].next;
            }

            (c_x.context("failed selecting Cx")?, c_y.context("failed selecting Cy")?, best)
        };


        // --- Rx for candidates (if needed) ---
        if nodes[cx].nbr.is_some() || nodes[cy].nbr.is_some() {
            nodes[cx].rx = compute_rx(cx, cx, cy, d, nodes, head);
            if let Some(cxb) = nodes[cx].nbr {
                nodes[cxb].rx = compute_rx(cxb, cx, cy, d, nodes, head);
            }
            nodes[cy].rx = compute_rx(cy, cx, cy, d, nodes, head);
            if let Some(cyb) = nodes[cy].nbr {
                nodes[cyb].rx = compute_rx(cyb, cx, cy, d, nodes, head);
            }

        }

        // --- Pick x,y among candidates ---
        let mut x = cx;
        let mut y = cy;

        let mut m = num_clusters;
        if nodes[cx].nbr.is_some() { m += 1; }
        if nodes[cy].nbr.is_some() { m += 1; }

        let mut best_q = (m as f64 - 2.0) * d[nodes[cx].id][nodes[cy].id] - nodes[cx].rx - nodes[cy].rx;

        if let Some(cxb) = nodes[cx].nbr {
            let qv = (m as f64 - 2.0) * d[nodes[cxb].id][nodes[cy].id] - nodes[cxb].rx - nodes[cy].rx;
            if fuzzy_lt(qv, best_q) { best_q = qv; x = cxb; y = cy; }
        }
        if let Some(cyb) = nodes[cy].nbr {
            let qv = (m as f64 - 2.0) * d[nodes[cx].id][nodes[cyb].id] - nodes[cx].rx - nodes[cyb].rx;
            if fuzzy_lt(qv, best_q) { best_q = qv; x = cx; y = cyb; }
        }
        if let (Some(cxb), Some(cyb)) = (nodes[cx].nbr, nodes[cy].nbr) {
            let qv = (m as f64 - 2.0) * d[nodes[cxb].id][nodes[cyb].id] - nodes[cxb].rx - nodes[cyb].rx;
            if fuzzy_lt(qv, best_q) { x = cxb; y = cyb; }
        }

        // --- Agglomeration ---
        match (nodes[x].nbr, nodes[y].nbr, num_active) {
            (None, None, _) => {               // 2-way
                join2way(nodes, x, y);
                num_clusters -= 1;
            }
            (None, Some(_), _) => {            // 3-way (x isolated)
                join3way(x, y, nodes[y].nbr.context("expected y.nbr for 3-way agglomeration")?, &mut joins, d, nodes, head, &mut num_nodes)?;
                num_nodes += 2;
                num_active -= 1;
                num_clusters -= 1;
            }
            (Some(_), None, _) | (_, _, 4) => { // 3-way (y isolated) OR last 4 active
                let x2 = y;
                let y2 = x;
                let y2_nbr = nodes[y2].nbr.context("expected y2.nbr for 3-way agglomeration")?;
                join3way(x2, y2, y2_nbr, &mut joins, d, nodes, head, &mut num_nodes)?;
                num_nodes += 2;
                num_active -= 1;
                num_clusters -= 1;
            }
            (Some(xb), Some(_yb), _) => {       // 4-way
                let yb = nodes[y].nbr.context("expected yb.nbr for 4-way agglomeration")?;
                join4way(xb, x, y, yb, &mut joins, d, nodes, head, &mut num_nodes)?;
                num_active -= 2;
                num_clusters -= 1;
            }
        }
    }

    Ok(joins)
}

fn fuzzy_lt(a: f64, b: f64) -> bool { (a - b) < -EPS }
fn fuzzy_eq(a: f64, b: f64) -> bool { (a - b).abs() <= EPS }

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

/* ---------------------------- join primitives --------------------------- */

fn join2way(nodes: &mut [NetNode], x: usize, y: usize) {
    nodes[x].nbr = Some(y);
    nodes[y].nbr = Some(x);
}

/// Returns the new node `u` (and pushes it onto `joins`)
fn join3way(
    x: usize, y: usize, z: usize,
    joins: &mut Vec<usize>,
    mat: &mut [Vec<f64>],
    nodes: &mut [NetNode],
    head: usize,
    num_nodes: &mut usize, // current max id; this function *does not* increment it (call-site does)
) -> Result<usize> {
    let u = *num_nodes + 1;
    let v = *num_nodes + 2;

    ensure!(u < nodes.len() && v < nodes.len(), "node capacity exceeded");

    nodes[u] = NetNode { id: u, ..Default::default() };
    nodes[v] = NetNode { id: v, ..Default::default() };

    nodes[u].ch1 = Some(x); nodes[u].ch2 = Some(y);
    nodes[v].ch1 = Some(y); nodes[v].ch2 = Some(z);
    nodes[u].nbr = Some(v); nodes[v].nbr = Some(u);

    // Replace x by u in the linked list
    nodes[u].next = nodes[x].next;
    nodes[u].prev = nodes[x].prev;
    if let Some(nx) = nodes[u].next { nodes[nx].prev = Some(u); }
    if let Some(px) = nodes[u].prev { nodes[px].next = Some(u); }

    // Replace z by v in the linked list
    nodes[v].next = nodes[z].next;
    nodes[v].prev = nodes[z].prev;
    if let Some(nz) = nodes[v].next { nodes[nz].prev = Some(v); }
    if let Some(pz) = nodes[v].prev { nodes[pz].next = Some(v); }

    // Remove y from the linked list
    if let Some(ny) = nodes[y].next { nodes[ny].prev = nodes[y].prev; }
    if let Some(py) = nodes[y].prev { nodes[py].next = nodes[y].next; }

    // --- Update distances exactly like the Java code ---
    // let actives = snapshot_active(nodes, head);
    {
        let xid = nodes[x].id;
        let yid = nodes[y].id;
        let zid = nodes[z].id;

        let mut p_opt = nodes[head].next;
        while let Some(p) = p_opt {
            let pid = nodes[p].id;

            mat[u][pid] = (2.0/3.0) * mat[xid][pid] + (1.0/3.0) * mat[yid][pid];
            mat[pid][u] = mat[u][pid];

            mat[v][pid] = (2.0/3.0) * mat[zid][pid] + (1.0/3.0) * mat[yid][pid];
            mat[pid][v] = mat[v][pid];

            p_opt = nodes[p].next;
        }
        mat[u][u] = 0.0;
        mat[v][v] = 0.0;
    }

    joins.push(u);
    Ok(u)
}

fn join4way(
    x2: usize, x: usize, y: usize, y2: usize,
    joins: &mut Vec<usize>,
    mat: &mut [Vec<f64>],
    nodes: &mut [NetNode],
    head: usize,
    num_nodes: &mut usize
) -> Result<()> {
    // First 3-way
    let u = join3way(x2, x, y, joins, mat, nodes, head, num_nodes)?;
    *num_nodes += 2;
    // Second 3-way
    let _ = join3way(u, nodes[u].nbr.context("u.nbr")?, y2, joins, mat, nodes, head, num_nodes)?;
    *num_nodes += 2;
    Ok(())
}


/* ---------------------------- scoring helpers --------------------------- */

fn avg_cluster_dist(mat: &[Vec<f64>], nodes: &[NetNode], p: usize, q: usize) -> f64 {
    match (nodes[p].nbr, nodes[q].nbr) {
        (None, None) => mat[nodes[p].id][nodes[q].id],
        (Some(pb), None) => 0.5 * (mat[nodes[p].id][nodes[q].id] + mat[nodes[pb].id][nodes[q].id]),
        (None, Some(qb)) => 0.5 * (mat[nodes[p].id][nodes[q].id] + mat[nodes[p].id][nodes[qb].id]),
        (Some(pb), Some(qb)) => 0.25
            * (mat[nodes[p].id][nodes[q].id]
                + mat[nodes[p].id][nodes[qb].id]
                + mat[nodes[pb].id][nodes[q].id]
                + mat[nodes[pb].id][nodes[qb].id]),
    }
}

fn compute_rx(z: usize, cx: usize, cy: usize, mat: &[Vec<f64>], nodes: &[NetNode], head: usize) -> f64 {
    let mut rx = 0.0;
    let mut p_opt = nodes[head].next;
    while let Some(p) = p_opt {
        let cond = p == cx
            || nodes[cx].nbr == Some(p)
            || p == cy
            || nodes[cy].nbr == Some(p)
            || nodes[p].nbr.is_none();
        let term = mat[nodes[z].id][nodes[p].id];
        rx += if cond { term } else { term / 2.0 };
        p_opt = nodes[p].next;
    }
    rx
}

/* ---------------------------- expansion phase --------------------------- */


fn are_adjacent(nodes: &[NetNode], u: usize, v: usize) -> Option<bool> {
    if nodes[u].next == Some(v) { Some(true) }
    else if nodes[v].next == Some(u) { Some(false) }
    else { None }
}

fn next_leaf_in_dir(nodes: &[NetNode], start: usize, forward: bool, n_tax: usize) -> Result<usize> {
    let mut a = start;
    loop {
        a = if forward {
            nodes[a].next.context("ring broken while seeking next leaf (forward)")?
        } else {
            nodes[a].prev.context("ring broken while seeking next leaf (backward)")?
        };
        let id = nodes[a].id;
        if (1..=n_tax).contains(&id) { return Ok(id); }
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
        let mut v  = nodes[u].nbr.context("u.nbr missing")?;
        let mut x1 = nodes[u].ch1.context("u.ch1 missing")?;
        let y1 = nodes[u].ch2.context("u.ch2 missing")?;
        let mut z1 = nodes[v].ch2.context("v.ch2 missing")?;

        // Make sure (u,v) are consecutive in the *current* ring
        match are_adjacent(nodes, u, v) {
            Some(true) => Ok(()),
            Some(false) => {
                // v -> u: swap roles and outer ends (x ↔ z)
                std::mem::swap(&mut u,  &mut v);
                std::mem::swap(&mut x1, &mut z1);
                Ok(())
            }
            None => Err(anyhow!("Join expansion invariant broken: u={} and v={} are not adjacent in the ring", u, v)),
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
            if nodes[cur].id == 1 { break cur; }
            cur = nodes[cur].next.context("broken ring while seeking leaf 1")?;
            ensure!(cur != head_next, "leaf 1 not found in ring");
        }
    };

    // Canonicalize orientation: pick the direction from 1 whose *next leaf* is smaller
    let next_leaf_fwd = next_leaf_in_dir(nodes, start, true,  n_tax)?;
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
            if seen == n_tax { break; }
        }
        a = if forward {
            nodes[a].next.context("ring broken during extraction (forward)")?
        } else {
            nodes[a].prev.context("ring broken during extraction (backward)")?
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
        let ord = compute_order_splits_tree4(&d).context("computing order for small triangle").unwrap();
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
        let ord = compute_order_splits_tree4(&d).context("computing order for small square").unwrap();
        let exp = vec![0, 1, 2, 4, 3];
        let rev_exp = vec![0, 1, 3, 4, 2];
        assert!(ord == exp || ord == rev_exp, "Expected order to be either {:?} or {:?}", exp, rev_exp);
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
        let ord = compute_order_splits_tree4(&d).context("computing order for smoke 5_1").unwrap();
        let exp = vec![0, 1, 2, 5, 4, 3];
        let rev_exp = vec![0, 1, 3, 4, 5, 2];
        assert!(ord == exp || ord == rev_exp, "Expected order to be either {:?} or {:?}", exp, rev_exp);
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
        let ord = compute_order_splits_tree4(&d).context("computing order for smoke 5_2").unwrap();
        let exp = vec![0, 1, 2, 4, 5, 3];
        assert_eq!(ord, exp);
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

        let ord = compute_order_splits_tree4(&d).context("computing order for smoke 10_1").unwrap();
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

        let ord = compute_order_splits_tree4(&d).context("computing order for smoke 10_2").unwrap();
        let exp_ord = vec![0, 1, 2, 4, 8, 3, 7, 9, 5, 6, 10];
        let rev_ex_order = vec![0, 1, 10, 6, 5, 9, 7, 3, 8, 4, 2];
        assert!(ord == exp_ord || ord == rev_ex_order, "Unexpected order {:?} | {:?} OR {:?}", ord, exp_ord, rev_ex_order);
    }

}
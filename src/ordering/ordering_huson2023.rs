use ndarray::Array2;
use petgraph::Undirected;
use petgraph::graph::{Graph, NodeIndex};
use rayon::prelude::*;
use std::cmp::min;

// --- components: always singleton or pair ---
#[derive(Clone, Copy, Debug)]
struct Component {
    first: usize,
    second: Option<usize>,
}

impl Component {
    fn singleton(first: usize) -> Self {
        Self {
            first,
            second: None,
        }
    }

    fn pair(a: usize, b: usize) -> Self {
        Self {
            first: a,
            second: Some(b),
        }
    }

    fn size(&self) -> usize {
        if self.second.is_some() { 2 } else { 1 }
    }

    fn values(&self) -> [Option<usize>; 2] {
        [Some(self.first), self.second]
    }

    fn other(&self, p: usize) -> usize {
        match self.second {
            Some(b) => {
                if p == self.first {
                    b
                } else {
                    self.first
                }
            }
            None => panic!("other() called on singleton"),
        }
    }

    fn first(&self) -> usize {
        self.first
    }

    fn second(&self) -> usize {
        self.second.expect("not a pair")
    }

    fn _is_singleton(&self) -> bool {
        self.second.is_none()
    }
}

/// Computes the circular ordering using the 2023 NeighborNet cycle algorithm.
/// Input distances are 0-based (shape n x n), symmetric, with zeros on the diagonal.
/// Output matches Java version: a vector of length n+1 with a leading 0 sentinel,
/// followed by a 1-based circular order of taxa.
pub fn compute_order_huson_2023(dist: &Array2<f64>) -> Vec<usize> {
    let n_tax = dist.nrows();
    assert_eq!(
        n_tax,
        dist.ncols(),
        "Distance matrix must be square (n x n)"
    );
    if n_tax <= 3 {
        return create_array_upward_count(n_tax);
    }

    // --- Graph & node map (1-based labels like the Java code) ---
    let mut graph: Graph<usize, (), Undirected> = Graph::default();
    // node_map[0] unused to keep 1-based indexing
    let mut node_map: Vec<NodeIndex> = vec![NodeIndex::end(); n_tax + 1];
    let mut components: Vec<Component> = (1..=n_tax).map(Component::singleton).collect();
    for t in 1..=n_tax {
        node_map[t] = graph.add_node(t);
    }

    // --- 1-based padded, mutable distance matrix D (mirrors Java) ---
    // D has shape (n_tax+1, n_tax+1), index 0 row/col unused.
    let mut d = Array2::<f64>::zeros((n_tax + 1, n_tax + 1));
    for i in 1..=n_tax {
        for j in 1..=n_tax {
            d[[i, j]] = dist[[i - 1, j - 1]];
        }
    }

    // ---- Core loop ----
    while components.len() >= 2 {
        debug!("Updated distance matrix D:\n{:?}", d);

        // Select closest pair of components (P,Q) per adjusted distance
        let (ip, iq) = select_closest_pair(&components, &d);

        let (p_comp, q_comp) = (components[ip], components[iq]);
        debug!("Selected: P={} Q={}", p_comp.first(), q_comp.first());
        debug!("Sizes: P={} Q={}", p_comp.size(), q_comp.size());
        if p_comp.size() == 1 && q_comp.size() == 1 {
            // --- Case 1: 1 vs 1 ---
            let p = p_comp.first();
            let q = q_comp.first();
            graph.add_edge(node_map[p], node_map[q], ());

            let new_component = Component::pair(p, q);
            components[ip] = new_component;
            debug!(
                "First case: P={} Q={} NewComponent={:?}",
                p, q, components[ip]
            );
            components.remove(iq);
        } else if p_comp.size() == 1 && q_comp.size() == 2 {
            // --- Case 2: 1 vs 2 ---
            let p = p_comp.first();
            let q = select_closest_1_vs_2(ip, iq, &d, &components);
            let qb = q_comp.other(q);

            // Update distances
            // D[p][qb] = D[qb][p] = (D[p][qb] + D[q][qb] + D[p][q]) / 3.0;
            d[[p, qb]] = (d[[p, qb]] + d[[q, qb]] + d[[p, q]]) / 3.0;
            d[[qb, p]] = d[[p, qb]];

            for i in 0..components.len() {
                if i == ip || i == iq {
                    continue; // skip P and Q
                }

                for &r_opt in components[i].values().iter() {
                    if let Some(r) = r_opt {
                        if r != p && r != q && r != qb {
                            // D[p][r] = D[r][p] = (2.0 * D[p][r] + D[q][r]) / 3.0;
                            // D[qb][r] = D[r][qb] = (2.0 * D[qb][r] + D[q][r]) / 3.0;
                            d[[p, r]] = (2.0 * d[[p, r]] + d[[q, r]]) / 3.0;
                            d[[r, p]] = d[[p, r]];
                            d[[qb, r]] = (2.0 * d[[qb, r]] + d[[q, r]]) / 3.0;
                            d[[r, qb]] = d[[qb, r]];
                        }
                    }
                }
            }

            // Update graph & components
            graph.add_edge(node_map[p], node_map[q], ());
            let new_component = Component::pair(p, qb);
            components[ip] = new_component;
            debug!(
                "Second case: P={} Q={} NewComponent={:?}",
                p, q, components[ip]
            );
            components.remove(iq);
        } else if p_comp.size() == 2 && q_comp.size() == 2 {
            // --- Case 3: 2 vs 2 ---
            let (p, q) = select_closest_2_vs_2(ip, iq, &d, &components);
            let pb = p_comp.other(p);
            let qb = q_comp.other(q);

            // D[pb][qb] = D[qb][pb] = (D[pb][p] + D[pb][q] + D[pb][qb] + D[p][q] + D[p][qb] + D[q][qb]) / 6.0;
            d[[pb, qb]] =
                (d[[pb, p]] + d[[pb, q]] + d[[pb, qb]] + d[[p, q]] + d[[p, qb]] + d[[q, qb]]) / 6.0;
            d[[qb, pb]] = d[[pb, qb]];

            for i in 0..components.len() {
                if i == ip || i == iq {
                    continue; // skip P and Q
                }

                let other = components[i];
                for &r_opt in other.values().iter() {
                    if let Some(r) = r_opt {
                        if r != p && r != q && r != pb && r != qb {
                            // D[pb][r] = D[r][pb] = D[pb][r] / 2.0 + D[p][r] / 3.0 + D[q][r] / 6.0;
                            // D[qb][r] = D[r][qb] = D[p][r] / 6.0 + D[q][r] / 3.0 + D[qb][r] / 2.0;
                            let pb_r = d[[pb, r]] / 2.0 + d[[p, r]] / 3.0 + d[[q, r]] / 6.0;
                            let qb_r = d[[p, r]] / 6.0 + d[[q, r]] / 3.0 + d[[qb, r]] / 2.0;
                            d[[pb, r]] = pb_r;
                            d[[r, pb]] = pb_r;
                            d[[qb, r]] = qb_r;
                            d[[r, qb]] = qb_r;
                        }
                    }
                }
            }

            graph.add_edge(node_map[p], node_map[q], ());
            let new_component = Component::pair(pb, qb);
            debug!(
                "Third case: P={} Q={} NewComponent={:?}",
                p, q, new_component
            );
            components[ip] = new_component;
            components.remove(iq);
        } else {
            panic!(
                "Internal error: |P|={} and |Q|={}",
                p_comp.size(),
                q_comp.size()
            );
        }
    }

    // Close cycle with the last remaining pair
    let p = components[0].first();
    let q = components[0].second();
    graph.add_edge(node_map[p], node_map[q], ());
    debug!("Final edge: P={} Q={}", p, q);
    debug!("Graph: {:?}", graph);

    extract_ordering(&graph, &node_map)
}

// ---------- helpers ----------

fn create_array_upward_count(n: usize) -> Vec<usize> {
    // Java produces an array of length n+1 with a leading 0 sentinel
    // followed by 1..n for small n:
    let mut v = Vec::with_capacity(n + 1);
    v.push(0);
    for t in 1..=n {
        v.push(t);
    }
    v
}

#[inline]
fn avg_d_comp_comp(d: &Array2<f64>, p: &Component, q: &Component) -> f64 {
    match (p.second, q.second) {
        (None, None) => {
            // 1x1
            d[[p.first, q.first]]
        }
        (None, Some(qb)) => {
            // 1x2
            (d[[p.first, q.first]] + d[[p.first, qb]]) / 2.0
        }
        (Some(pb), None) => {
            // 2x1
            (d[[p.first, q.first]] + d[[pb, q.first]]) / 2.0
        }
        (Some(pb), Some(qb)) => {
            // 2x2
            (d[[p.first, q.first]] + d[[p.first, qb]] + d[[pb, q.first]] + d[[pb, qb]]) / 4.0
        }
    }
}

#[inline]
fn avg_d_p_comp(d: &Array2<f64>, p: usize, q: &Component) -> f64 {
    match q.second {
        None => d[[p, q.first]],
        Some(qb) => (d[[p, q.first]] + d[[p, qb]]) / 2.0,
    }
}

/// Serial, deterministic closest-pair selection (matches Java)
fn select_closest_pair(components: &[Component], d: &Array2<f64>) -> (usize, usize) {
    let m = components.len();
    if m == 2 {
        // ensure |P| <= |Q|
        if components[0].size() < components[1].size() {
            return (0, 1);
        } else {
            return (1, 0);
        }
    }

    let (_best_val, best_ip, best_iq) = (0..m)
        .into_par_iter()
        .filter(|&ip| ip + 1 < m)
        .map(|ip| {
            let p = components[ip];
            let mut local_best_val = f64::INFINITY;
            let mut local_best_iq = ip + 1;

            for iq in (ip + 1)..m {
                let q = components[iq];

                // sums over all S != P,Q
                let mut sum_p = 0.0;
                let mut sum_q = 0.0;
                for is in 0..m {
                    if is == ip || is == iq {
                        continue;
                    }
                    let s = components[is];
                    sum_p += avg_d_comp_comp(d, &p, &s);
                    sum_q += avg_d_comp_comp(d, &q, &s);
                }

                let pq = avg_d_comp_comp(d, &p, &q);
                let adjusted = (m as f64 - 2.0) * pq - sum_p - sum_q;

                if adjusted < local_best_val {
                    local_best_val = adjusted;
                    local_best_iq = iq;
                }
            }

            (local_best_val, ip, local_best_iq)
        })
        .reduce(
            || (f64::INFINITY, 0usize, 1usize),
            |a, b| {
                let cmp = a.0.total_cmp(&b.0);
                if cmp == std::cmp::Ordering::Less {
                    a
                } else if cmp == std::cmp::Ordering::Greater {
                    b
                } else if a.1 < b.1 || (a.1 == b.1 && a.2 <= b.2) {
                    a
                } else {
                    b
                }
            },
        );

    // Ensure |P| <= |Q|
    if components[best_ip].size() > components[best_iq].size() {
        return (best_iq, best_ip);
    }

    (best_ip, best_iq)
}

fn select_closest_1_vs_2(ip: usize, iq: usize, d: &Array2<f64>, components: &[Component]) -> usize {
    let m = components.len();
    let p = components[ip].first();
    let q1 = components[iq].first();
    let q2 = components[iq].second();

    let mut p_r = d[[q1, p]] + d[[q2, p]];
    let mut q1_r = d[[q1, q2]] + d[[q1, p]];
    let mut q2_r = d[[q1, q2]] + d[[q2, p]];

    for i in 0..m {
        if i == ip || i == iq {
            continue;
        }
        let other = components[i];
        p_r += avg_d_p_comp(d, p, &other);
        q1_r += avg_d_p_comp(d, q1, &other);
        q2_r += avg_d_p_comp(d, q2, &other);
    }

    let mm1 = m as f64 - 1.0;
    let q1p_adj = mm1 * d[[q1, p]] - q1_r - p_r;
    let q2p_adj = mm1 * d[[q2, p]] - q2_r - p_r;

    debug!(
        "1x2 @ P={} Q={} -> q1p_adj:{:.9} q2p_adj:{:.9}",
        p, q1, q1p_adj, q2p_adj
    );

    if q1p_adj <= q2p_adj { q1 } else { q2 }
}

fn select_closest_2_vs_2(
    ip: usize,
    iq: usize,
    d: &Array2<f64>,
    components: &[Component],
) -> (usize, usize) {
    let m = components.len();

    let p1 = components[ip].first();
    let p2 = components[ip].second();
    let q1 = components[iq].first();
    let q2 = components[iq].second();

    let mut p1_r = d[[p1, q1]] + d[[p1, q2]];
    let mut p2_r = d[[p2, q1]] + d[[p2, q2]];
    let mut q1_r = d[[p1, q1]] + d[[p2, q1]];
    let mut q2_r = d[[p1, q2]] + d[[p2, q2]];

    for i in 0..m {
        if i == ip || i == iq {
            continue;
        }
        let other = components[i];
        p1_r += avg_d_p_comp(d, p1, &other);
        p2_r += avg_d_p_comp(d, p2, &other);
        q1_r += avg_d_p_comp(d, q1, &other);
        q2_r += avg_d_p_comp(d, q2, &other);
    }

    // m * D[p][q] - pR - qR
    let m_f = m as f64;
    let p1q1 = m_f * d[[p1, q1]] - p1_r - q1_r;
    let p2q1 = m_f * d[[p2, q1]] - p2_r - q1_r;
    let p1q2 = m_f * d[[p1, q2]] - p1_r - q2_r;
    let p2q2 = m_f * d[[p2, q2]] - p2_r - q2_r;

    debug!(
        "2x2 inputs P={{{}, {}}} Q={{{}, {}}} | \
        D: p1q1={:.12}, p2q1={:.12}, p1q2={:.12}, p2q2={:.12} | \
        R: p1R={:.12}, p2R={:.12}, q1R={:.12}, q2R={:.12}",
        p1,
        p2,
        q1,
        q2,
        d[[p1, q1]],
        d[[p2, q1]],
        d[[p1, q2]],
        d[[p2, q2]],
        p1_r,
        p2_r,
        q1_r,
        q2_r
    );

    match rank_of_min(&[p1q1, p2q1, p1q2, p2q2]) {
        0 => (p1, q1),
        1 => (p2, q1),
        2 => (p1, q2),
        _ => (p2, q2),
    }
}

#[inline]
fn rank_of_min(vals: &[f64]) -> usize {
    let mut idx = 0usize;
    for i in 1..vals.len() {
        if vals[i] < vals[idx] {
            idx = i;
        }
    }
    idx
}

/// Cycle extraction:
/// - start at taxon 1
/// - first step: pick the smaller-labeled of its two neighbors (stable direction)
/// - then always go to the neighbor != previous (degree 2 on a cycle)
/// Returns 1-based cycle with a leading 0 sentinel.
fn extract_ordering(graph: &Graph<usize, (), Undirected>, node_map: &[NodeIndex]) -> Vec<usize> {
    let n = node_map.len() - 1;
    let mut order: Vec<usize> = Vec::with_capacity(n + 1);
    order.push(0);
    debug!("Node Map: {:?}", node_map);
    if n == 0 {
        return order;
    }
    if n <= 3 {
        for t in 1..=n {
            order.push(t);
        }
        return order;
    }

    let v1 = node_map[1];
    // neighbors of 1: choose direction by smaller taxon label
    let neigh: Vec<NodeIndex> = graph.neighbors(v1).collect();
    assert!(
        neigh.len() == 2,
        "ordering graph must be a simple cycle: node 1 should have degree 2"
    );
    // neigh.sort_by_key(|&v| graph[v]); // stable direction
    let mut prev = v1;
    let mut cur = min(neigh[0], neigh[1]);
    debug!(
        "Starting at v1={} cur={}: neighbours {:?}",
        graph[v1], graph[cur], neigh
    );

    order.push(graph[v1]); // taxon 1
    while order.len() - 1 < n {
        order.push(graph[cur]);
        // advance: choose neighbor of `cur` that isn't `prev`
        let it = graph.neighbors(cur).collect::<Vec<_>>();
        debug!("prev {:?} -> cur {:?}: neighbours {:?}", prev, cur, it);
        let a = it[0];
        let b = it[1];
        let nxt = if a == prev { b } else { a };
        prev = cur;
        cur = nxt;
    }
    order
}

// ---------- tests ----------
fn _debug_pair(components: &[Component], d: &Array2<f64>, a: usize, b: usize) {
    let ip = components
        .iter()
        .position(|c| c.first == a || c.second == Some(a))
        .unwrap();
    let iq = components
        .iter()
        .position(|c| c.first == b || c.second == Some(b))
        .unwrap();

    let m = components.len();
    let p = &components[ip];
    let q = &components[iq];
    let mut sum_p = 0.0;
    let mut sum_q = 0.0;

    debug!("-- adj breakdown for P={:?} Q={:?} (m={})", p, q, m);
    for (is, s) in components.iter().enumerate() {
        if is == ip || is == iq {
            continue;
        }
        let aps = avg_d_comp_comp(d, p, s);
        let aqs = avg_d_comp_comp(d, q, s);
        debug!("  S={:?}: avg(P,S)={:.9} avg(Q,S)={:.9}", s, aps, aqs);
        sum_p += aps;
        sum_q += aqs;
    }
    let pq = avg_d_comp_comp(d, p, q);
    let adjusted = (m as f64 - 2.0) * pq - sum_p - sum_q;
    debug!(
        "  avg(P,Q)={:.9} sumP={:.9} sumQ={:.9} -> adjusted={:.9}",
        pq, sum_p, sum_q, adjusted
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn small_triangle() {
        // 3 taxa — returns [0,1,2,3]
        let d = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]]);
        let ord = compute_order_huson_2023(&d);
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
        let ord = compute_order_huson_2023(&d);
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
        let ord = compute_order_huson_2023(&d);
        let exp = vec![0, 1, 2, 5, 4, 3];
        assert_eq!(ord, exp);
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
        let ord = compute_order_huson_2023(&d);
        let exp = vec![0, 1, 2, 4, 5, 3];
        assert_eq!(ord, exp);
    }

    #[test]
    fn smoke_10_1() {
        // 10 taxa - returns [0, 1, 5, 7, 9, 3, 8, 4, 2, 10, 6]
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

        let ord = compute_order_huson_2023(&d);
        assert_eq!(ord, vec![0, 1, 5, 7, 9, 3, 8, 4, 2, 10, 6]);
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

        let ord = compute_order_huson_2023(&d);
        assert_eq!(ord, vec![0, 1, 2, 4, 8, 3, 7, 9, 5, 6, 10]);
    }

    #[test]
    fn smoke_15_1() {
        // 15 taxa - [0, 1, 4, 9, 5, 13, 8, 7, 3, 2, 14, 12, 10, 11, 15, 6]
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
                0.0, 14.0, 9.0, 4.0, 16.0, 11.0, 17.0, 12.0, 7.0, 19.0, 14.0, 9.0, 15.0, 10.0, 5.0,
            ],
            [
                14.0, 0.0, 17.0, 12.0, 7.0, 13.0, 8.0, 3.0, 15.0, 10.0, 22.0, 11.0, 6.0, 18.0, 13.0,
            ],
            [
                9.0, 17.0, 0.0, 20.0, 9.0, 4.0, 16.0, 11.0, 6.0, 18.0, 7.0, 2.0, 14.0, 9.0, 21.0,
            ],
            [
                4.0, 12.0, 20.0, 0.0, 17.0, 12.0, 7.0, 19.0, 14.0, 3.0, 15.0, 10.0, 5.0, 17.0, 12.0,
            ],
            [
                16.0, 7.0, 9.0, 17.0, 0.0, 20.0, 15.0, 10.0, 16.0, 11.0, 6.0, 18.0, 13.0, 8.0, 14.0,
            ],
            [
                11.0, 13.0, 4.0, 12.0, 20.0, 0.0, 6.0, 12.0, 7.0, 19.0, 14.0, 9.0, 21.0, 10.0, 5.0,
            ],
            [
                17.0, 8.0, 16.0, 7.0, 15.0, 6.0, 0.0, 3.0, 15.0, 10.0, 5.0, 17.0, 6.0, 18.0, 13.0,
            ],
            [
                12.0, 3.0, 11.0, 19.0, 10.0, 12.0, 3.0, 0.0, 6.0, 18.0, 13.0, 2.0, 14.0, 9.0, 4.0,
            ],
            [
                7.0, 15.0, 6.0, 14.0, 16.0, 7.0, 15.0, 6.0, 0.0, 9.0, 15.0, 10.0, 5.0, 17.0, 12.0,
            ],
            [
                19.0, 10.0, 18.0, 3.0, 11.0, 19.0, 10.0, 18.0, 9.0, 0.0, 6.0, 18.0, 13.0, 8.0, 20.0,
            ],
            [
                14.0, 22.0, 7.0, 15.0, 6.0, 14.0, 5.0, 13.0, 15.0, 6.0, 0.0, 9.0, 21.0, 16.0, 5.0,
            ],
            [
                9.0, 11.0, 2.0, 10.0, 18.0, 9.0, 17.0, 2.0, 10.0, 18.0, 9.0, 0.0, 12.0, 1.0, 13.0,
            ],
            [
                15.0, 6.0, 14.0, 5.0, 13.0, 21.0, 6.0, 14.0, 5.0, 13.0, 21.0, 12.0, 0.0, 9.0, 4.0,
            ],
            [
                10.0, 18.0, 9.0, 17.0, 8.0, 10.0, 18.0, 9.0, 17.0, 8.0, 16.0, 1.0, 9.0, 0.0, 12.0,
            ],
            [
                5.0, 13.0, 21.0, 12.0, 14.0, 5.0, 13.0, 4.0, 12.0, 20.0, 5.0, 13.0, 4.0, 12.0, 0.0,
            ],
        ]);

        let ord = compute_order_huson_2023(&d);
        assert_eq!(
            ord,
            vec![0, 1, 9, 3, 6, 12, 14, 5, 11, 10, 4, 7, 8, 2, 13, 15]
        );
    }
}

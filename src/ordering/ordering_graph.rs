use ndarray::Array2;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use rayon::prelude::*;
use std::collections::HashSet;

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
        if self.second.is_some() {
            2
        } else {
            1
        }
    }


    fn values(&self) -> [usize; 2] {
        match self.second {
            Some(b) => [self.first, b],
            None => [self.first, 0], // second unused
        }
    }


    fn other(&self, p: usize) -> usize {
        match self.second {
            Some(b) => {
                if p == self.first { b } else { self.first }
            }
            None => panic!("other() called on singleton"),
        }
    }

    fn first(&self) -> usize { self.first }
    fn second(&self) -> usize { self.second.expect("not a pair") }
    fn is_singleton(&self) -> bool { self.second.is_none() }
}

/// Computes the circular ordering using the 2023 NeighborNet cycle algorithm.
/// Input distances are 0-based (shape n x n), symmetric, with zeros on the diagonal.
/// Output matches Java version: a vector of length n+1 with a leading 0 sentinel,
/// followed by a 1-based circular order of taxa.
pub fn compute_ordering(dist: &Array2<f64>) -> Vec<usize> {
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



    let mut components: Vec<Component> = (1..=n_tax).map(Component::singleton).collect();

    // ---- Core loop ----
    while components.len() >= 2 {
        // Select closest pair of components (P,Q) per adjusted distance
        let (mut ip, mut iq) = select_closest_pair(&components, &d);

        // Ensure |P| <= |Q| like Java does
        if components[ip].size() > components[iq].size() {
            std::mem::swap(&mut ip, &mut iq);
        }

        let (p_comp, q_comp) = (components[ip], components[iq]);

        if p_comp.size() == 1 && q_comp.size() == 1 {
            // --- Case 1: 1 vs 1 ---
            let p = p_comp.first();
            let q = q_comp.first();
            graph.add_edge(node_map[p], node_map[q], ());

            let new_component = Component::pair(p, q);
            components[ip] = new_component;
            components.remove(iq);
        } else if p_comp.size() == 1 && q_comp.size() == 2 {
            // --- Case 2: 1 vs 2 ---
            let p = p_comp.first();
            let q = select_closest_1_vs_2(ip, iq, &d, &components);
            let qb = q_comp.other(q);

            // Update distances
            d[[p, qb]] = (d[[p, qb]] + d[[q, qb]] + d[[p, q]]) / 3.0;
            d[[qb, p]] = d[[p, qb]];

            for i in 0..components.len() {
                if i != ip && i != iq {
                    let other = components[i];
                    for &r in other.values().iter() {
                        if r == 0 { continue; } // skip unused
                        if r != p && r != q && r != qb {
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
            components.remove(iq);
        } else if p_comp.size() == 2 && q_comp.size() == 2 {
            // --- Case 3: 2 vs 2 ---
            let (p, q) = select_closest_2_vs_2(ip, iq, &d, &components);
            let pb = p_comp.other(p);
            let qb = q_comp.other(q);

            d[[pb, qb]] = (d[[pb, p]] + d[[pb, q]] + d[[pb, qb]] + d[[p, q]] + d[[p, qb]] + d[[q, qb]]) / 6.0;
            d[[qb, pb]] = d[[pb, qb]];

            for i in 0..components.len() {
                if i != ip && i != iq {
                    let other = components[i];
                    for &r in other.values().iter() {
                        if r == 0 { continue; }
                        if r != p && r != q && r != pb && r != qb {
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

#[derive(Clone, Copy)]
struct Pair(usize, usize);

#[inline]
fn avg_d_comp_comp(d: &Array2<f64>, p: (usize, Option<usize>), q: (usize, Option<usize>)) -> f64 {
    let (p1, p2) = (p.0, p.1);
    let (q1, q2) = (q.0, q.1);
    match (p2, q2) {
        (None, None) => d[[p1, q1]], // 1x1
        (None, Some(qb)) => (d[[p1, q1]] + d[[p1, qb]]) / 2.0, // 1x2
        (Some(pb), None) => (d[[p1, q1]] + d[[pb, q1]]) / 2.0, // 2x1
        (Some(pb), Some(qb)) => {
            (d[[p1, q1]] + d[[p1, qb]] + d[[pb, q1]] + d[[pb, qb]]) / 4.0 // 2x2
        }
    }
}

#[inline]
fn avg_d_p_comp(d: &Array2<f64>, p: usize, q: (usize, Option<usize>)) -> f64 {
    match q.1 {
        None => d[[p, q.0]],
        Some(qb) => (d[[p, q.0]] + d[[p, qb]]) / 2.0,
    }
}

/// Parallel selection of the closest pair (ip, iq) with ip < iq.
/// Uses: adjustedD = (m-2)*avgD(P,Q) - sum_{S≠P,Q} avgD(P,S) - sum_{S≠P,Q} avgD(Q,S)
fn select_closest_pair(components: &[Component], d: &Array2<f64>) -> (usize, usize) {
    let m = components.len();
    if m == 2 {
        return (0, 1);
    }

    // Prebuild all (ip, iq) pairs where ip < iq
    let pairs: Vec<(usize, usize)> = (0..m)
        .flat_map(|ip| (ip + 1..m).map(move |iq| (ip, iq)))
        .collect();

    // Compute adjusted distances in parallel
    let best = pairs
        .par_iter()
        .map(|&(ip, iq)| {
            let p = components[ip];
            let q = components[iq];

            // sums over others
            let mut sum_p = 0.0;
            let mut sum_q = 0.0;
            for is in 0..m {
                if is == ip || is == iq {
                    continue;
                }
                let s = components[is];
                sum_p += avg_d_comp_comp(d, (p.first, p.second), (s.first, s.second));
                sum_q += avg_d_comp_comp(d, (q.first, q.second), (s.first, s.second));
            }

            let pq = avg_d_comp_comp(d, (p.first, p.second), (q.first, q.second));
            let adjusted = (m as f64 - 2.0) * pq - sum_p - sum_q;
            (adjusted, ip, iq)
        })
        .reduce(
            || (f64::INFINITY, 0usize, 1usize),
            |a, b| if a.0 <= b.0 { a } else { b },
        );

    let (_, mut ip, mut iq) = best;
    if ip > iq {
        std::mem::swap(&mut ip, &mut iq);
    }
    (ip, iq)
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
        p_r += avg_d_p_comp(d, p, (other.first, other.second));
        q1_r += avg_d_p_comp(d, q1, (other.first, other.second));
        q2_r += avg_d_p_comp(d, q2, (other.first, other.second));
    }

    let mm1 = m as f64 - 1.0;
    let q1p_adj = mm1 * d[[q1, p]] - q1_r - p_r;
    let q2p_adj = mm1 * d[[q2, p]] - q2_r - p_r;

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
        p1_r += avg_d_p_comp(d, p1, (other.first, other.second));
        p2_r += avg_d_p_comp(d, p2, (other.first, other.second));
        q1_r += avg_d_p_comp(d, q1, (other.first, other.second));
        q2_r += avg_d_p_comp(d, q2, (other.first, other.second));
    }

    // m * D[p][q] - pR - qR (Java)
    let m_f = m as f64;
    let p1q1 = m_f * d[[p1, q1]] - p1_r - q1_r;
    let p2q1 = m_f * d[[p2, q1]] - p2_r - q1_r;
    let p1q2 = m_f * d[[p1, q2]] - p1_r - q2_r;
    let p2q2 = m_f * d[[p2, q2]] - p2_r - q2_r;

    match rank_of_min([p1q1, p2q1, p1q2, p2q2]) {
        0 => (p1, q1),
        1 => (p2, q1),
        2 => (p1, q2),
        _ => (p2, q2),
    }
}

#[inline]
fn rank_of_min(vals: [f64; 4]) -> usize {
    let mut idx = 0usize;
    for i in 1..4 {
        if vals[i] < vals[idx] {
            idx = i;
        }
    }
    idx
}

/// Follow the cycle starting from taxon 1, visiting the unseen neighbor each time.
/// Prepend 0 as a sentinel (to match the Java output shape).
fn extract_ordering(graph: &Graph<usize, (), Undirected>, node_map: &[NodeIndex]) -> Vec<usize> {
    let n = node_map.len() - 1;
    let mut order: Vec<usize> = Vec::with_capacity(n + 1);
    order.push(0); // "cycle is 1-based" sentinel

    let mut seen: HashSet<NodeIndex> = HashSet::with_capacity(n);
    let mut v = node_map[1];

    loop {
        order.push(graph[v]); // node weight is the taxon label (1-based)
        seen.insert(v);
        if seen.len() == n {
            break;
        }
        // choose the neighbor not yet seen
        let mut next_opt = None;
        for w in graph.neighbors(v) {
            if !seen.contains(&w) {
                next_opt = Some(w);
                break;
            }
        }
        v = next_opt.expect("graph should form a single cycle");
    }

    order
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn small_triangle() {
        // 3 taxa — returns [0,1,2,3]
        let d = arr2(&[
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ]);
        let ord = compute_ordering(&d);
        assert_eq!(ord, vec![0, 1, 2, 3]);
    }

    #[test]
    fn smoke_5() {
        let d = arr2(&[
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ]);
        let ord = compute_ordering(&d);
        // Shape check: n+1, starts with 0, contains 1..=5
        assert_eq!(ord.len(), 6);
        assert_eq!(ord[0], 0);
        let mut seen = ord[1..].to_vec();
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 2, 3, 4, 5]);
    }
}

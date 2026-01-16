use anyhow::Result;
use clap::Args;
use fixedbitset::FixedBitSet;
use ndarray::Array2;
use rayon::prelude::*;
use std::{
    cmp::max,
    time::{Duration, Instant},
};

use crate::splits::asplit::ASplit; // <-- adjust path if needed

const EPS: f64 = 1e-12;

#[derive(Debug)]
struct Scratch {
    p: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    w: Vec<f64>,
}

impl Scratch {
    fn new(len: usize) -> Self {
        Self {
            p: vec![0.0; len],
            r: vec![0.0; len],
            z: vec![0.0; len],
            w: vec![0.0; len],
        }
    }
}

pub trait Progress {
    fn check_for_cancel(&self) -> Result<()>;
}

#[derive(Args, Clone, Debug)]
pub struct NNLSParams {
    /// Include only split weights > cutoff (trivial splits always included).
    #[arg(short, long, default_value = "1e-4")]
    pub cutoff: f64,
    /// Stop if squared projected gradient < this (reset from ‖Aᵀ d‖ at runtime).
    #[arg(short, long, default_value = "1e-5")]
    pub proj_grad_bound: f64,
    /// Hard iteration cap (outer loops).
    #[arg(short, long, default_value = "5000")]
    pub max_iterations: usize,
    /// Wall-clock cap in ms (use `u64::MAX` to disable).
    #[arg(short, long, default_value_t = u64::MAX)]
    pub max_time_ms: u64,
    // CGNR
    #[arg(short, long, default_value = "50")]
    pub cgnr_iterations: usize,
    #[arg(short, long, default_value = "5e-6")]
    pub cgnr_tolerance: f64,
    /// Fraction of offending (negative) coords to push into the active set per correction.
    #[arg(short, long, default_value = "0.4")]
    pub active_set_rho: f64,
}

impl Default for NNLSParams {
    fn default() -> Self {
        Self {
            cutoff: 1e-4,
            proj_grad_bound: 1e-5, // will be reset from ‖Aᵀ d‖ below
            max_iterations: 5000,
            max_time_ms: u64::MAX,
            cgnr_iterations: 50,
            cgnr_tolerance: 1e-5 / 2.0,
            active_set_rho: 0.4,
        }
    }
}

/// Flattened weights (mainly for printlnging/validation).
#[derive(Clone, Debug)]
pub struct SplitWeights {
    /// x[(i,j)] for 1 ≤ i < j ≤ n, flattened by blocks (1,2..n), (2,3..n), ...
    pub x: Vec<f64>,
}

impl PartialEq for SplitWeights {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
    }
}

/// Public API: return **ASplit**s directly (wired).
///
/// - `cycle`: NeighborNet circular order (1-based with leading `0` sentinel)
/// - `distances`: symmetric `n×n` matrix in original taxon indices (0-based)
pub fn compute_asplits(
    cycle: &[usize],
    distances: &Array2<f64>,
    params: &mut NNLSParams,
    progress: Option<&dyn Progress>,
) -> Result<Vec<ASplit>> {
    let (_weights, pairs) = compute_use_1d(cycle, distances, params, progress)?;
    let n = cycle.len() - 1;

    // Convert bitset + weight pairs to ASplit
    let splits = pairs
        .into_iter()
        .map(|(a, w)| ASplit::from_a_ntax_with_weight(a, n, w))
        .collect();

    Ok(splits)
}

/// Core solver (vector form), also returns `(A_bitset, weight)` for building ASplit.
pub fn compute_use_1d(
    cycle: &[usize],
    distances: &Array2<f64>,
    params: &mut NNLSParams,
    progress: Option<&dyn Progress>,
) -> Result<(SplitWeights, Vec<(FixedBitSet, f64)>)> {
    let n = cycle.len() - 1;
    if n == 0 || n == 1 {
        return Ok((SplitWeights { x: vec![] }, vec![]));
    }
    if n == 2 {
        let d_12 = distances[[cycle[1] - 1, cycle[2] - 1]];
        let mut a = FixedBitSet::with_capacity(n + 1);
        a.grow(n + 1);
        if d_12 > 0.0 {
            a.set(cycle[1], true);
        }
        return Ok((SplitWeights { x: vec![d_12] }, vec![(a, d_12.max(0.0))]));
    }

    // Flatten distances d[(i,j)], 1<=i<j<=n, in cycle order (1-based).
    let npairs = n * (n - 1) / 2;
    let mut d = vec![0.0; npairs];
    {
        let mut idx = 0usize;
        for i in 1..=n {
            for j in (i + 1)..=n {
                d[idx] = distances[[cycle[i] - 1, cycle[j] - 1]];
                idx += 1;
            }
        }
    }

    // Threshold from ||Aᵀ d||.
    let mut a_t_d = vec![0.0; npairs];
    calc_atx(&d, &mut a_t_d, n);
    let norm_atd = sum_array_squared(&a_t_d, n).sqrt();
    params.proj_grad_bound = (1e-4 * norm_atd).powi(2);

    // var x = new double[npairs]; //array of split weights
    // calcAinv_y(d, x, n); //Compute unconstrained solution
    // var minVal = minArray(x);
    let mut x = vec![0.0; npairs];
    calc_ainv_y(&d, &mut x, n);
    let min_val = x.iter().copied().fold(f64::INFINITY, f64::min);
    if min_val < 0.0 {
        let start = Instant::now();
        zero_negative_entries(&mut x);
        let mut active = vec![false; npairs];
        get_active_entries(&x, &mut active);

        let mut scratch = Scratch::new(npairs); // p, r, z, w
        let mut splits_idx = vec![0usize; npairs];
        let mut order_idx: Vec<usize> = (0..npairs).collect();
        let mut vals = vec![0.0; npairs];

        params.cgnr_tolerance = params.proj_grad_bound / 2.0;
        params.cgnr_iterations = max(params.cgnr_iterations, n * (n - 1) / 2);
        params.active_set_rho = 0.4;

        active_set_method(
            &mut x,
            &d,
            n,
            params,
            &mut active,
            &mut scratch,
            &mut splits_idx,
            &mut order_idx,
            &mut vals,
            progress,
            start,
        )?;
    }

    // Build ASplit-compatible pairs: (bitset A, weight)
    let mut out_pairs = Vec::new();
    let mut idx = 0usize;
    for i in 1..=n {
        let mut a = FixedBitSet::with_capacity(n + 1);
        a.grow(n + 1);
        for j in (i + 1)..=n {
            a.set(cycle[j - 1], true);
            let w = x[idx].max(0.0);

            // Include if w>cutoff OR trivial (|A|==1 or n-1)
            let size_a = a.ones().filter(|&t| t != 0).count();
            if w > params.cutoff || size_a == 1 || size_a == n - 1 {
                out_pairs.push((a.clone(), w));
            }
            idx += 1;
        }
    }

    Ok((SplitWeights { x }, out_pairs))
}

/* ===================== Active-Set + CGNR ===================== */

fn active_set_method(
    x: &mut [f64], // feasible (x >= 0)
    d: &[f64],
    n: usize,
    params: &mut NNLSParams,
    active: &mut [bool],
    scratch: &mut Scratch, // p, r, z, w
    tmp_splits: &mut [usize],
    idx_sorted: &mut [usize],
    vals: &mut [f64],
    progress: Option<&dyn Progress>,
    started: Instant,
) -> Result<()> {
    let npairs = x.len();
    let mut xstar = vec![0.0; npairs];
    let mut k_outer = 0usize;
    debug!("active set {:?}", active);
    debug!("==============================");
    debug!("x: {:?}", x);
    let start = Instant::now();
    loop {
        debug!("Outer Loop {}", k_outer);
        loop {
            xstar.copy_from_slice(x);

            let iters = cgnr(
                &mut xstar,
                d,
                active,
                n,
                params.cgnr_iterations,
                params.cgnr_tolerance,
                scratch,
                progress,
                started,
                params.max_time_ms,
            )?;

            k_outer += 1;

            let ok = feasible_move_active_set(
                x, &xstar, active, tmp_splits, idx_sorted, vals, n, params,
            );

            debug!(
                "\t{}\t{}\t{}\t{}",
                k_outer,
                params.cgnr_iterations,
                params.active_set_rho,
                (Instant::now() - start).as_millis()
            );

            if ok && iters < params.cgnr_iterations {
                break;
            }
            if k_outer > params.max_iterations {
                return Ok(());
            }
        }

        x.copy_from_slice(&xstar);
        // projected gradient check at current x
        let (p, r) = (&mut scratch.p, &mut scratch.r);

        // print sum of x, d, p, r
        debug!("{} Sum of x: {:?}", k_outer, x.iter().sum::<f64>());
        debug!("{} Sum of d: {:?}", k_outer, d.iter().sum::<f64>());
        debug!("{} Sum of p: {:?}", k_outer, p.iter().sum::<f64>());
        debug!("{} Sum of r: {:?}", k_outer, r.iter().sum::<f64>());

        eval_gradient(x, d, p, r, n); // p := grad

        p.iter_mut().zip(x.iter()).for_each(|(gi, &xi)| {
            if xi == 0.0 {
                // Match Java: at the bound, only allow components pointing inside the feasible region.
                *gi = gi.min(0.0);
            }
        });

        debug!("Projected gradient sum {}: {:?}", n, p.iter().sum::<f64>());
        let pg = sum_array_squared(p, n);
        if (pg - params.proj_grad_bound) < -EPS {
            return Ok(());
        }

        debug!(
            "Projected gradient squared = {}\t target = {}\tNumber iterations = {}",
            pg, params.proj_grad_bound, k_outer
        );

        // Try to release worst active constraint
        let mut imin = 0usize;
        let mut pmin = 0.0f64;
        for i in 0..npairs {
            if active[i] && p[i] < pmin {
                pmin = p[i];
                imin = i;
            }
        }
        if pmin < 0.0 {
            active[imin] = false;
        }
    }
}

/// Rust port of Java:
/// feasibleMoveActiveSet(x, xstar, activeSet, splits, indices, vals, n, params)
///
/// Returns `true` if no negative entries in `xstar` (so we just copy `xstar -> x`);
/// otherwise marks some positions active and moves `x` toward `xstar` by `tmin`,
/// then returns `false`.
pub fn feasible_move_active_set(
    x: &mut [f64],
    xstar: &[f64],
    active_set: &mut [bool],
    splits: &mut [usize],
    indices: &mut [usize],
    vals: &mut [f64],
    n: usize,
    params: &NNLSParams,
) -> bool {
    let npairs = n * (n - 1) / 2;

    // Basic sanity checks to match Java's expectations
    assert_eq!(x.len(), npairs, "x len must be n*(n-1)/2");
    assert_eq!(xstar.len(), npairs, "xstar len must be n*(n-1)/2");
    assert_eq!(active_set.len(), npairs, "active_set len must be n*(n-1)/2");
    assert!(
        splits.len() >= npairs && indices.len() >= npairs && vals.len() >= npairs,
        "splits/indices/vals must have capacity >= n*(n-1)/2"
    );

    // Collect candidates where xstar[i] < 0
    let mut count = 0usize;
    for i in 0..npairs {
        if xstar[i] < 0.0 {
            // Java: vals[count] = x[i] / (x[i] - xstar[i]);
            vals[count] = x[i] / (x[i] - xstar[i]);
            splits[count] = i;
            indices[count] = count;
            count += 1;
        }
    }

    if count == 0 {
        // Java: copyArray(xstar, x); return true;
        x.copy_from_slice(xstar);
        return true;
    }

    // Sort indices[0..count) by vals[idx] ascending, like Java's Comparator.comparingDouble:
    // - NaNs sort AFTER numbers
    // - -0.0 < +0.0
    indices[..count].sort_by(|&a, &b| {
        let va = vals[a];
        let vb = vals[b];
        match (va.is_nan(), vb.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater, // NaN after numbers
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => va.total_cmp(&vb), // IEEE total order: -0.0 < +0.0
        }
    });

    let tmin = vals[indices[0]];

    // Java: max(1, ceil(count * activeSetRho))
    let num_to_make_active = max(
        1,
        (f64::from(count as u32) * params.active_set_rho).ceil() as usize,
    );

    // Mark chosen split positions active
    for i in 0..num_to_make_active {
        let pos = splits[indices[i]];
        active_set[pos] = true;
    }

    // Blend x toward xstar by tmin on non-active; zero-out active
    let mut active_count = 0usize;
    for i in 0..npairs {
        if active_set[i] {
            x[i] = 0.0;
            active_count += 1;
        } else {
            x[i] = (1.0 - tmin) * x[i] + tmin * xstar[i];
        }
    }

    debug!("Active count {}", active_count);

    false
}

/* ===================== Linear ops A and Aᵀ (vector form) ===================== */

fn calc_ax(x: &[f64], y: &mut [f64], n: usize) {
    debug_assert_eq!(y.len(), x.len());

    // y(i,i+1)
    for i in 1..=(n - 1) {
        let mut s = 0.0;
        for j in (i + 2)..=n {
            s += x[pair_idx(i + 1, j, n)];
        }
        for j in 1..=i {
            s += x[pair_idx(j, i + 1, n)];
        }
        y[pair_idx(i, i + 1, n)] = s;
    }

    // y(i,i+2)
    for i in 1..=(n - 2) {
        let a = y[pair_idx(i, i + 1, n)];
        let b = y[pair_idx(i + 1, i + 2, n)];
        let c = x[pair_idx(i + 1, i + 2, n)];
        y[pair_idx(i, i + 2, n)] = a + b - 2.0 * c;
    }

    // general recurrence
    for k in 3..=(n - 1) {
        for i in 1..=(n - k) {
            let j = i + k;
            let y_ijm1 = y[pair_idx(i, j - 1, n)];
            let y_ip1j = y[pair_idx(i + 1, j, n)];
            let y_ip1jm1 = y[pair_idx(i + 1, j - 1, n)];
            let x_ip1j = x[pair_idx(i + 1, j, n)];
            y[pair_idx(i, j, n)] = y_ijm1 + y_ip1j - y_ip1jm1 - 2.0 * x_ip1j;
        }
    }
}

fn calc_atx(x: &[f64], y: &mut [f64], n: usize) {
    let npairs = x.len();
    debug_assert_eq!(npairs, n * (n - 1) / 2);

    // pass 1
    let mut s_index = 0usize;
    for i in 1..n {
        let mut d_index: isize = (i as isize) - 2;
        let mut y_s = 0.0;
        for j in 1..i {
            y_s += x[d_index as usize];
            d_index += (n - j) as isize - 1;
        }
        d_index = s_index as isize;
        for _j in (i + 1)..=n {
            y_s += x[d_index as usize];
            d_index += 1;
        }
        y[s_index] = y_s;
        s_index += n - i;
    }

    // pass 2
    s_index = 1;
    for i in 1..=(n - 2) {
        y[s_index] = y[s_index - 1] + y[s_index + n - i - 1] - 2.0 * x[s_index - 1];
        s_index += n - i;
    }

    // pass 3
    for k in 3..=(n - 1) {
        s_index = k - 1;
        for i in 1..=(n - k) {
            y[s_index] = y[s_index - 1] + y[s_index + n - i - 1]
                - y[s_index + n - i - 2]
                - 2.0 * x[s_index - 1];
            s_index += n - i;
        }
    }
}

/// Compute x from y for size `n` using the same index arithmetic as the Java version.
/// - `y.len()` and `x.len()` must both be n*(n-1)/2 (upper triangle packed vector).
pub fn calc_ainv_y(y: &[f64], x: &mut [f64], n: usize) {
    assert!(n >= 2, "n must be >= 2");
    let m = n * (n - 1) / 2;
    assert_eq!(y.len(), m, "y must have length n*(n-1)/2");
    assert_eq!(x.len(), m, "x must have length n*(n-1)/2");

    // --- First row (i = 1 in the paper/code comments) ---
    // x[0] = (y[n-2] + y[0] - y[2n-4]) / 2
    {
        let t1 = y[n - 2] + y[0];
        let t2 = t1 - y[2 * n - 4];
        x[0] = t2 / 2.0;
    }

    // d_index starts at (2,n) in the original comments => 2n-4 in 0-based packed vector
    let mut d_index: isize = (2 * n - 4) as isize;

    // for j = 3..=n:
    // x[j-2] = (y[d_index] + y[j-2] - y[j-3] - y[d_index + n - j]) / 2
    for j in 3..=n {
        let j_i = j as isize;
        let n_i = n as isize;

        let t1 = y[usize::try_from(d_index).unwrap()] + y[j - 2];
        let t2 = t1 - y[j - 3];
        let t3 = t2 - y[usize::try_from(d_index + n_i - j_i).unwrap()];
        x[j - 2] = t3 / 2.0;

        d_index += n_i - j_i;
    }

    // x[n-2] = (y[n-2] + y[last] - y[n-3]) / 2
    {
        let t1 = y[n - 2] + y[y.len() - 1];
        let t2 = t1 - y[n - 3];
        x[n - 2] = t2 / 2.0;
    }

    // --- Remaining rows (i = 2..=n-1) ---
    // s_index = (2n - i) * (i - 1) / 2   (0-based packed start index for row i)
    for i in 2..=(n - 1) {
        let i_i = i as isize;
        let n_i = n as isize;

        let mut s_index: isize = ((2 * n_i - i_i) * (i_i - 1)) / 2;

        // x[i][i+1]
        // x[s_index] = (y[s_index + i - n - 1] + y[s_index] - y[s_index + i - n]) / 2
        {
            let a = y[usize::try_from(s_index + i_i - n_i - 1).unwrap()];
            let b = y[usize::try_from(s_index).unwrap()];
            let c = y[usize::try_from(s_index + i_i - n_i).unwrap()];
            let t1 = a + b;
            let t2 = t1 - c;
            x[usize::try_from(s_index).unwrap()] = t2 / 2.0;
        }
        s_index += 1;

        // for j = i+2..=n:
        // x[s_index] = (y[s_index + i - n - 1] + y[s_index] - y[s_index - 1] - y[s_index + i - n]) / 2
        for _j in (i + 2)..=n {
            let a = y[usize::try_from(s_index + i_i - n_i - 1).unwrap()];
            let b = y[usize::try_from(s_index).unwrap()];
            let c = y[usize::try_from(s_index - 1).unwrap()];
            let d = y[usize::try_from(s_index + i_i - n_i).unwrap()];
            let t1 = a + b;
            let t2 = t1 - c;
            let t3 = t2 - d;
            x[usize::try_from(s_index).unwrap()] = t3 / 2.0;

            s_index += 1;
        }
    }
}

/* ===================== Gradient & CGNR (rayon-accelerated) ===================== */

fn eval_gradient(x: &[f64], d: &[f64], gradient: &mut [f64], residual: &mut [f64], n: usize) {
    calc_ax(x, residual, n);
    residual
        .par_iter_mut()
        .zip(d.par_iter())
        .for_each(|(ri, &di)| *ri -= di);
    calc_atx(residual, gradient, n);
}

fn get_active_entries(x: &[f64], a: &mut [bool]) {
    a.par_iter_mut()
        .zip(x.par_iter())
        .for_each(|(ai, &xi)| *ai = xi <= 0.0);
}

fn zero_negative_entries(x: &mut [f64]) {
    x.par_iter_mut().for_each(|xi| {
        if *xi < 0.0 {
            *xi = 0.0
        }
    });
}

/// Sum of squares over a packed upper-triangular vector `x` (i<j) for an n×n matrix.
/// Mirrors the Java implementation's iteration and accumulation order.
pub fn sum_array_squared(x: &[f64], n: usize) -> f64 {
    let expected = n * (n - 1) / 2;
    assert!(
        x.len() == expected,
        "x must have length n*(n-1)/2 (got {}, expected {})",
        x.len(),
        expected
    );

    let mut total = 0.0f64;
    let mut index = 0usize;

    // Java loops i=1..=n, j=i+1..=n over the packed upper triangle.
    for i in 1..=n {
        let mut s_i = 0.0f64;
        for _j in (i + 1)..=n {
            let x_ij = x[index];
            s_i += x_ij * x_ij;
            index += 1;
        }
        // Sum each row separately, then add (matches Java's numeric stability choice)
        total += s_i;
    }

    total
}

fn cgnr(
    x: &mut [f64],
    d: &[f64],
    active_set: &[bool],
    n: usize,
    max_iters: usize,
    tol: f64,
    scratch: &mut Scratch,
    progress: Option<&dyn Progress>,
    started: Instant,
    max_time_ms: u64,
) -> Result<usize> {
    let (p, r, z, w) = (
        &mut scratch.p,
        &mut scratch.r,
        &mut scratch.z,
        &mut scratch.w,
    );

    debug!("Sum of z: {:?}", z.iter().sum::<f64>());
    debug!("Sum of w: {:?}", w.iter().sum::<f64>());

    zero_negative_entries(x);

    // r = d - A x
    calc_ax(x, r, n);
    r.iter_mut()
        .zip(d.iter())
        .for_each(|(ri, &di)| *ri = di - *ri);

    // z = Aᵀ r; mask actives
    calc_atx(r, z, n);
    mask_elements_branchless(z, active_set);
    p.copy_from_slice(z);

    let mut ztz = sum_array_squared(z, n);
    let mut k = 1usize;

    loop {
        // w = A p
        calc_ax(p, w, n);
        let denom = sum_array_squared(w, n);
        let alpha = ztz / denom;

        // x += alpha p; r -= alpha w
        for i in 0..x.len() {
            x[i] += alpha * p[i];
            r[i] -= alpha * w[i];
        }

        // z = Aᵀ r; mask
        calc_atx(r, z, n);
        mask_elements_branchless(z, active_set);

        let ztz_new = sum_array_squared(z, n);
        let beta = ztz_new / ztz;
        if ztz_new < tol || k >= max_iters {
            break;
        }
        // debug!("k={} ztz={} beta={} new_ztz={} sum x {} sum r {}", k, ztz, beta, ztz_new, x.iter().sum::<f64>(), r.iter().sum::<f64>());

        // for (var i = 0; i < p.length; i++) {
        //     p[i] = z[i] + beta * p[i];
        // }
        // ztz = ztz2;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i]; // correct: z + beta * old_p
        }
        // debug!("sum p {:?} sum z {:?}", p.iter().sum::<f64>(), z.iter().sum::<f64>());

        ztz = ztz_new;
        k += 1;

        if let Some(pl) = progress {
            if (k % n) == 0 {
                pl.check_for_cancel()?;
            }
        }
        if started.elapsed() >= Duration::from_millis(max_time_ms) {
            break;
        }
    }
    Ok(k)
}

/* ===================== Indexing ===================== */
pub fn mask_elements_branchless(r: &mut [f64], a: &[bool]) {
    assert_eq!(r.len(), a.len());
    for (x, &mask) in r.iter_mut().zip(a.iter()) {
        *x *= (!mask) as u8 as f64; // true -> 0, false -> 1
    }
}

#[inline]
fn pair_idx(i: usize, j: usize, n: usize) -> usize {
    // blocks: (1,2..n), (2,3..n), ...
    debug_assert!(1 <= i && i < j && j <= n);
    let offset = (i - 1) * n - (i - 1) * i / 2;
    offset + (j - i - 1)
}

/* ===================== Tests ===================== */

#[cfg(test)]
mod tests {
    use crate::ordering::ordering_huson2023::compute_order_huson_2023;

    use super::*;
    use ndarray::{Array2, arr2};

    fn build_pairs_from_fn(n: usize, f: impl Fn(usize, usize) -> f64) -> Vec<f64> {
        let mut x = vec![0.0; n * (n - 1) / 2];
        let mut idx = 0usize;
        for i in 1..=n {
            for j in (i + 1)..=n {
                x[idx] = f(i, j);
                idx += 1;
            }
        }
        x
    }

    fn ax_to_distances(n: usize, pairs: &[f64]) -> Array2<f64> {
        // Compute y = A x, then expand to full symmetric matrix in the identity cycle.
        let mut y = vec![0.0; pairs.len()];
        calc_ax(pairs, &mut y, n);
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 1..=n {
            for j in (i + 1)..=n {
                let v = y[pair_idx(i, j, n)];
                d[[i - 1, j - 1]] = v;
                d[[j - 1, i - 1]] = v;
            }
        }
        d
    }

    #[test]
    fn pair_idx_sanity_n5() {
        let n = 5;
        // Expected contiguous blocks: (1,2..5)=4, (2,3..5)=3, (3,4..5)=2, (4,5)=1
        assert_eq!(pair_idx(1, 2, n), 0);
        assert_eq!(pair_idx(1, 3, n), 1);
        assert_eq!(pair_idx(1, 4, n), 2);
        assert_eq!(pair_idx(1, 5, n), 3);
        assert_eq!(pair_idx(2, 3, n), 4);
        assert_eq!(pair_idx(2, 4, n), 5);
        assert_eq!(pair_idx(2, 5, n), 6);
        assert_eq!(pair_idx(3, 4, n), 7);
        assert_eq!(pair_idx(3, 5, n), 8);
        assert_eq!(pair_idx(4, 5, n), 9);
    }

    #[test]
    fn pair_idx_roundtrip() {
        // Reconstruct (i,j) by scanning; ensures block starts/lengths line up for general n
        fn inv(idx: usize, n: usize) -> (usize, usize) {
            let mut i = 1;
            let mut base = 0usize;
            while i < n {
                let len = n - i;
                if idx < base + len {
                    let j = i + 1 + (idx - base);
                    return (i, j);
                }
                base += len;
                i += 1;
            }
            unreachable!()
        }
        for n in 2..20 {
            let total = n * (n - 1) / 2;
            for idx in 0..total {
                let (i, j) = inv(idx, n);
                assert_eq!(pair_idx(i, j, n), idx);
            }
        }
    }

    #[test]
    fn reconstruct_uniform_weights_n5() {
        let n = 5;
        let cycle: Vec<usize> = (0..=n).collect(); // [0,1,2,3,4,5]
        let x_true = build_pairs_from_fn(n, |_i, _j| 1.0);
        let distances = ax_to_distances(n, &x_true);

        let mut params = NNLSParams::default();
        let (_w, asplits) = compute_use_1d(&cycle, &distances, &mut params, None).unwrap();

        // We expect exactly npairs interval splits (all positive, cutoff << 1.0)
        assert_eq!(asplits.len(), n * (n - 1) / 2);

        // Rebuild x_est back from returned ASplits by decoding the interval A
        // (cycle is identity so A = {i, i+1, ..., j-1} ⇒ first set bit = i, |A| = j-i).
        let mut x_est = vec![0.0; x_true.len()];
        for (a, w) in asplits {
            let i = a.ones().filter(|&t| t != 0).min().unwrap();
            let size = a.ones().filter(|&t| t != 0).count();
            let j = i + size;
            let idx = pair_idx(i, j, n);
            x_est[idx] = w;
        }

        // Compare
        for (et, tt) in x_est.iter().zip(x_true.iter()) {
            assert!((*et - *tt).abs() < 1e-8, "got {et}, wanted {tt}");
        }
    }

    #[test]
    fn optimality_check_pg_small() {
        let n = 6;
        let cycle: Vec<usize> = (0..=n).collect();
        // Make a non-uniform weight field
        let x_true = build_pairs_from_fn(n, |i, j| (i + j) as f64 * 0.1);
        let distances = ax_to_distances(n, &x_true);

        let mut params = NNLSParams::default();
        let (weights, _splits) = compute_use_1d(&cycle, &distances, &mut params, None).unwrap();

        // Check projected gradient at solution is small
        let npairs = n * (n - 1) / 2;
        let mut grad = vec![0.0; npairs];
        let mut resid = vec![0.0; npairs];
        eval_gradient(
            &weights.x,
            &y_pairs_from_matrix(&cycle, &distances),
            &mut grad,
            &mut resid,
            n,
        );

        // Project
        grad.iter_mut().zip(weights.x.iter()).for_each(|(g, &xi)| {
            if xi == 0.0 {
                *g = g.min(0.0)
            }
        });

        let pg = sum_array_squared(&grad, n);
        assert!(pg < params.proj_grad_bound * 10.0); // loose but indicative
    }

    fn y_pairs_from_matrix(cycle: &[usize], distances: &Array2<f64>) -> Vec<f64> {
        let n = cycle.len() - 1;
        let mut d = vec![0.0; n * (n - 1) / 2];
        let mut idx = 0usize;
        for i in 1..=n {
            for j in (i + 1)..=n {
                d[idx] = distances[[cycle[i] - 1, cycle[j] - 1]];
                idx += 1;
            }
        }
        d
    }

    #[test]
    fn test_calc_atx_smoke_10_1() {
        let n = 10;
        let n_pairs = n * (n - 1) / 2;
        let mut atx = vec![0.0; n_pairs];
        let cycle = vec![0, 1, 5, 7, 9, 3, 8, 4, 2, 10, 6];
        let distances = arr2(&[
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

        let mut d = vec![0.0; n_pairs];
        let mut index = 0;
        for i in 1..=n {
            for j in (i + 1)..=n {
                d[index] = distances[[cycle[i] - 1, cycle[j] - 1]];
                index += 1;
            }
        }

        calc_atx(&d, &mut atx, n);

        let exp = vec![
            67.0, 129.0, 176.0, 208.0, 197.0, 184.0, 160.0, 105.0, 56.0, 68.0, 137.0, 177.0, 190.0,
            189.0, 179.0, 134.0, 105.0, 71.0, 115.0, 146.0, 171.0, 183.0, 166.0, 151.0, 48.0, 95.0,
            132.0, 164.0, 173.0, 174.0, 57.0, 112.0, 156.0, 189.0, 200.0, 59.0, 111.0, 160.0,
            183.0, 60.0, 123.0, 160.0, 67.0, 122.0, 57.0,
        ];

        compare_float_array(&atx, &exp, 1e-8);
    }

    #[test]
    fn smoke_5_1() {
        let d = arr2(&[
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ]);
        let ord = vec![0, 1, 2, 5, 4, 3];
        let mut params = NNLSParams::default();
        let progress = None; // No progress tracking in this test
        let (_unused, pairs) = compute_use_1d(&ord, &d, &mut params, progress).expect("NNLS solve");

        assert_eq!(pairs.len(), 7);
        let weights = pairs.iter().map(|(_, w)| *w).collect::<Vec<f64>>();
        let expected_weights = vec![2.0, 3.0, 4.0, 3.0, 1.0, 2.0, 2.0];
        compare_float_array(&weights, &expected_weights, 1e-8);
    }

    #[test]
    fn smoke_10_1() {
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

        let ord = vec![0, 1, 5, 7, 9, 3, 8, 4, 2, 10, 6];
        let mut params = NNLSParams::default();
        let progress = None; // No progress tracking in this test

        let (_unused, pairs) = compute_use_1d(&ord, &d, &mut params, progress).expect("NNLS solve");

        assert_eq!(pairs.len(), 22);
        let weights = pairs.iter().map(|(_, w)| *w).collect::<Vec<f64>>();
        let expected_weights = vec![
            1.3724245430681956,
            1.3454752556293355,
            2.012589097140714,
            0.432648565932274,
            1.2912946058459376,
            0.0,
            1.7933838153171087,
            0.7711228988729982,
            1.0043842023776657,
            0.8997353824936709,
            0.6914161425268539,
            0.0,
            0.24973217871606335,
            0.9249564751107247,
            0.5594817784033412,
            0.6432103250072205,
            0.7511706061796792,
            0.06892851867483342,
            1.9992119883544948,
            0.0,
            1.889915657808208,
            0.0,
        ];
        compare_float_array(&weights, &expected_weights, 1e-8);
    }

    #[test]
    fn test_calc_ax() {
        let x = vec![
            0.08387815223462815,
            0.168552547193302,
            -0.0,
            0.2725865087474663,
            -0.0,
            -0.0,
            0.22820664558236292,
            -0.0,
            0.07308425784060892,
            -0.0,
            0.17730778700107405,
            0.27863338881613003,
            0.2468579134685847,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            0.0976830848827301,
            -0.0,
            -0.0,
            -0.0,
            0.2643530681213975,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            0.14757998083208332,
            0.10660005040749763,
            0.23764744206450752,
            0.26470104652828064,
            -0.0,
            0.1361962459593511,
            -0.0,
            0.0,
            -0.0,
            0.0,
            0.15848309888614887,
            -0.0,
            -0.0,
            0.15496911033502833,
            0.0,
        ];
        let mut y = vec![
            -0.0005469984750199043,
            -0.0008175078531115416,
            -0.00019075688926191675,
            -0.003992158520172918,
            -0.0022548516859499726,
            -0.0011927905553219325,
            -0.0010935633842476958,
            -0.0007804106343922194,
            -0.0005739648424036184,
            -0.00027050937809163726,
            -0.0009962639205133004,
            -0.0005944356671792588,
            -0.0034345964694969715,
            -0.002372535338868931,
            -0.0022733081677946943,
            -0.001960155417939218,
            -0.0017537096259506169,
            -0.000626750963849625,
            -0.00022492271051558345,
            -0.003065083512833296,
            -0.0020030223822052554,
            -0.0026737439059483312,
            -0.0023605911560928543,
            -0.0021541453641042534,
            0.0004018282533340416,
            -0.0024383325489836707,
            -0.0013762714183556297,
            -0.0020469929420987056,
            -0.0017338401922432289,
            -0.0015273944002546281,
            0.007061077971995901,
            -0.0025919828141187104,
            -0.0024577237764863004,
            -0.002978261677074054,
            -0.0027718158850854532,
            -0.0008546759798957635,
            -0.0007204169422633535,
            -0.0012409548428511067,
            -0.0010345090508625057,
            0.00013425903763241012,
            -0.0001788937122230671,
            2.7552079765533937e-5,
            -0.0003131527498554772,
            -0.0008470466652137695,
            -0.0005338939153582923,
        ];
        let n = 10;
        calc_ax(&x, &mut y, n);

        let y_exp = vec![
            0.7866772415204168,
            1.3172659417178465,
            1.0422750698340424,
            1.7927567095977475,
            1.534515061256431,
            1.4502018637757308,
            1.3313771095072173,
            0.9081929640927879,
            0.8263081115983688,
            0.5305887001974297,
            0.6102134023157736,
            1.917961819711739,
            2.1534359983075917,
            2.0691228008268916,
            1.9502980465583781,
            1.5271139011439487,
            1.4452290486495294,
            0.27499087188380417,
            1.5827392892797696,
            1.8182134678756223,
            1.7339002703949224,
            2.1437816523692037,
            1.720597506954774,
            1.6387126544603547,
            1.3077484173959655,
            1.543222595991818,
            1.458909398511118,
            1.868790780485399,
            1.4456066350709693,
            1.3637217825765497,
            0.5306341402600191,
            0.6595210435943145,
            1.5446973096976109,
            1.6509152573397423,
            1.569030404845323,
            0.4012793952529976,
            1.2864556613562939,
            1.3926736089984255,
            1.310788756504006,
            0.8851762661032964,
            1.3083604115177259,
            1.2264755590233065,
            0.4231841454144295,
            0.6512375135900668,
            0.22805336817563726,
        ];
        compare_float_array(&y, &y_exp, 1e-8);
    }

    #[test]
    fn smoke_20() {
        let d = arr2(&[
            [
                0.000000, 0.438878, 0.858598, 0.697368, 0.094177, 0.975622, 0.761140, 0.786064,
                0.128114, 0.450386, 0.370798, 0.926765, 0.643865, 0.822762, 0.443414, 0.227239,
                0.554585, 0.063817, 0.827631, 0.631664,
            ],
            [
                0.438878, 0.000000, 0.970698, 0.893121, 0.778383, 0.194639, 0.466721, 0.043804,
                0.154289, 0.683049, 0.744762, 0.967510, 0.325825, 0.370460, 0.469556, 0.189471,
                0.129922, 0.475705, 0.226909, 0.669814,
            ],
            [
                0.858598, 0.970698, 0.000000, 0.312367, 0.832260, 0.804764, 0.387478, 0.288328,
                0.682496, 0.139752, 0.199908, 0.007362, 0.786924, 0.664851, 0.705165, 0.780729,
                0.458916, 0.568741, 0.139797, 0.114530,
            ],
            [
                0.697368, 0.893121, 0.312367, 0.000000, 0.634718, 0.553579, 0.559207, 0.303950,
                0.030818, 0.436717, 0.214585, 0.408529, 0.853403, 0.233939, 0.058303, 0.281384,
                0.293594, 0.661917, 0.557032, 0.783898,
            ],
            [
                0.094177, 0.778383, 0.832260, 0.634718, 0.000000, 0.090048, 0.722359, 0.461877,
                0.161272, 0.501045, 0.152312, 0.696320, 0.446156, 0.381021, 0.301512, 0.630283,
                0.361813, 0.087650, 0.118006, 0.961898,
            ],
            [
                0.975622, 0.194639, 0.804764, 0.553579, 0.090048, 0.000000, 0.449362, 0.272242,
                0.096391, 0.902602, 0.455776, 0.202363, 0.305957, 0.579220, 0.176773, 0.856614,
                0.758520, 0.719463, 0.432093, 0.627309,
            ],
            [
                0.761140, 0.466721, 0.387478, 0.559207, 0.722359, 0.449362, 0.000000, 0.144524,
                0.103403, 0.587645, 0.170593, 0.925120, 0.581061, 0.346870, 0.590915, 0.022804,
                0.958559, 0.482303, 0.782735, 0.082730,
            ],
            [
                0.786064, 0.043804, 0.288328, 0.303950, 0.461877, 0.272242, 0.144524, 0.000000,
                0.438911, 0.021612, 0.826292, 0.896161, 0.140249, 0.554036, 0.108576, 0.672240,
                0.281234, 0.659423, 0.726995, 0.768647,
            ],
            [
                0.128114, 0.154289, 0.682496, 0.030818, 0.161272, 0.096391, 0.103403, 0.438911,
                0.000000, 0.952899, 0.290918, 0.515057, 0.255965, 0.936044, 0.164608, 0.044911,
                0.435097, 0.992376, 0.891677, 0.748608,
            ],
            [
                0.450386, 0.683049, 0.139752, 0.436717, 0.501045, 0.902602, 0.587645, 0.021612,
                0.952899, 0.000000, 0.936813, 0.240971, 0.122758, 0.831113, 0.153284, 0.179268,
                0.599383, 0.874562, 0.196435, 0.310324,
            ],
            [
                0.370798, 0.744762, 0.199908, 0.214585, 0.152312, 0.455776, 0.170593, 0.826292,
                0.290918, 0.936813, 0.000000, 0.581117, 0.199776, 0.804125, 0.715407, 0.738984,
                0.131058, 0.123754, 0.927563, 0.397578,
            ],
            [
                0.926765, 0.967510, 0.007362, 0.408529, 0.696320, 0.202363, 0.925120, 0.896161,
                0.515057, 0.240971, 0.581117, 0.000000, 0.966232, 0.596043, 0.933023, 0.804361,
                0.467382, 0.784763, 0.017837, 0.109144,
            ],
            [
                0.643865, 0.325825, 0.786924, 0.853403, 0.446156, 0.305957, 0.581061, 0.140249,
                0.255965, 0.122758, 0.199776, 0.966232, 0.000000, 0.247840, 0.236662, 0.746014,
                0.816569, 0.105278, 0.066559, 0.594434,
            ],
            [
                0.822762, 0.370460, 0.664851, 0.233939, 0.381021, 0.579220, 0.346870, 0.554036,
                0.936044, 0.831113, 0.804125, 0.596043, 0.247840, 0.000000, 0.680715, 0.393630,
                0.317991, 0.504526, 0.875005, 0.851132,
            ],
            [
                0.443414, 0.469556, 0.705165, 0.058303, 0.301512, 0.176773, 0.590915, 0.108576,
                0.164608, 0.153284, 0.715407, 0.933023, 0.236662, 0.680715, 0.000000, 0.902653,
                0.979571, 0.802026, 0.779478, 0.642483,
            ],
            [
                0.227239, 0.189471, 0.780729, 0.281384, 0.630283, 0.856614, 0.022804, 0.672240,
                0.044911, 0.179268, 0.738984, 0.804361, 0.746014, 0.393630, 0.902653, 0.000000,
                0.379464, 0.685743, 0.296876, 0.948858,
            ],
            [
                0.554585, 0.129922, 0.458916, 0.293594, 0.361813, 0.758520, 0.958559, 0.281234,
                0.435097, 0.599383, 0.131058, 0.467382, 0.816569, 0.317991, 0.979571, 0.379464,
                0.000000, 0.725716, 0.084493, 0.935940,
            ],
            [
                0.063817, 0.475705, 0.568741, 0.661917, 0.087650, 0.719463, 0.482303, 0.659423,
                0.992376, 0.874562, 0.123754, 0.784763, 0.105278, 0.504526, 0.802026, 0.685743,
                0.725716, 0.000000, 0.492153, 0.599593,
            ],
            [
                0.827631, 0.226909, 0.139797, 0.557032, 0.118006, 0.432093, 0.782735, 0.726995,
                0.891677, 0.196435, 0.927563, 0.017837, 0.066559, 0.875005, 0.779478, 0.296876,
                0.084493, 0.492153, 0.000000, 0.729686,
            ],
            [
                0.631664, 0.669814, 0.114530, 0.783898, 0.961898, 0.627309, 0.082730, 0.768647,
                0.748608, 0.310324, 0.397578, 0.109144, 0.594434, 0.851132, 0.642483, 0.948858,
                0.935940, 0.599593, 0.729686, 0.000000,
            ],
        ]);
        let ord = compute_order_huson_2023(&d);
        let mut params = NNLSParams::default();
        let progress = None; // No progress tracking in this test

        let (_unused, pairs) = compute_use_1d(&ord, &d, &mut params, progress).expect("NNLS solve");

        assert_eq!(pairs.len(), 50);
        let weights = pairs.iter().map(|(_, w)| *w).collect::<Vec<f64>>();

        let y_exp = vec![
            0.0,
            0.07443701124163404,
            0.008854418256011971,
            0.047052392535544,
            0.0,
            0.0,
            0.028437091426710127,
            0.1014819306894528,
            0.0,
            0.099735116497362,
            0.009076419105761136,
            0.12867528350288612,
            0.026822551594168173,
            0.0,
            0.1018766253178327,
            0.0,
            0.00536481969225152,
            0.019651481810187964,
            0.022204183936506056,
            0.13827460319422014,
            0.03379901971866573,
            0.008503974990191693,
            0.030615545661299648,
            0.016628364165308475,
            0.012105416623377177,
            0.0,
            0.019619614961265172,
            0.020919572578000537,
            0.0,
            0.09174727081827827,
            0.007696847303817906,
            0.0110261994938533,
            0.01993109263979137,
            0.15024929489868422,
            0.012309173790781782,
            0.17896205136338988,
            0.09754547819744813,
            0.016205102766173502,
            0.09095055150255353,
            0.0,
            0.09862840131875934,
            0.0,
            0.09160272102789793,
            0.0,
            0.041533890987266966,
            0.06736768518744002,
            0.0,
            0.06858387882037113,
            0.06034051869836362,
            0.0,
        ];

        compare_float_array(&weights, &y_exp, 1e-8);
    }

    #[test]
    fn smoke_30() {
        let d = arr2(&[
            [
                0.00, 4.09, 0.33, 3.16, 4.69, 0.09, 2.33, 0.16, 2.40, 3.37, 0.55, 0.92, 0.09, 0.89,
                1.54, 0.63, 2.04, 1.95, 1.51, 1.99, 3.01, 3.06, 1.28, 2.33, 4.40, 1.92, 1.68, 0.59,
                3.76, 0.62,
            ],
            [
                4.09, 0.00, 3.70, 0.91, 1.56, 1.04, 3.97, 2.83, 2.06, 2.50, 0.59, 2.45, 4.81, 4.86,
                1.71, 0.68, 2.78, 4.21, 3.53, 0.95, 2.38, 1.99, 4.57, 2.24, 3.31, 1.23, 3.66, 2.81,
                0.40, 0.97,
            ],
            [
                0.33, 3.70, 0.00, 3.24, 0.08, 3.13, 4.49, 2.38, 3.36, 1.40, 1.72, 2.36, 0.63, 3.13,
                3.31, 3.65, 0.86, 3.84, 0.46, 2.49, 3.19, 2.59, 2.41, 4.93, 4.24, 0.06, 0.41, 4.93,
                3.74, 3.64,
            ],
            [
                3.16, 0.91, 3.24, 0.00, 4.00, 1.35, 1.53, 4.81, 2.88, 2.74, 4.28, 1.47, 4.66, 3.14,
                2.23, 3.00, 3.28, 0.07, 4.61, 3.73, 0.62, 2.06, 2.85, 2.83, 1.92, 2.68, 0.51, 3.81,
                1.24, 4.40,
            ],
            [
                4.69, 1.56, 0.08, 4.00, 0.00, 0.33, 4.28, 0.55, 4.06, 3.71, 3.86, 0.21, 3.62, 0.62,
                0.71, 1.37, 3.97, 4.83, 1.38, 3.46, 1.24, 4.90, 0.47, 3.98, 3.63, 4.32, 0.77, 1.37,
                2.61, 1.22,
            ],
            [
                0.09, 1.04, 3.13, 1.35, 0.33, 0.00, 1.25, 1.49, 3.49, 2.22, 4.45, 3.44, 2.80, 0.81,
                2.67, 2.95, 1.64, 0.71, 3.28, 3.57, 0.31, 4.68, 3.58, 1.54, 4.11, 3.17, 2.40, 3.05,
                3.78, 3.92,
            ],
            [
                2.33, 3.97, 4.49, 1.53, 4.28, 1.25, 0.00, 2.14, 3.55, 2.94, 0.25, 0.51, 0.39, 0.45,
                1.69, 3.43, 1.11, 0.45, 3.53, 4.63, 3.68, 2.75, 2.72, 0.43, 2.35, 2.87, 2.45, 0.39,
                2.95, 0.06,
            ],
            [
                0.16, 2.83, 2.38, 4.81, 0.55, 1.49, 2.14, 0.00, 1.32, 4.21, 4.47, 1.86, 1.12, 4.92,
                3.44, 3.67, 4.36, 1.53, 0.31, 2.23, 0.41, 4.83, 1.03, 2.85, 4.94, 3.51, 0.03, 2.31,
                1.78, 4.99,
            ],
            [
                2.40, 2.06, 3.36, 2.88, 4.06, 3.49, 3.55, 1.32, 0.00, 4.35, 4.04, 2.02, 1.47, 4.28,
                3.72, 2.13, 0.03, 1.62, 3.74, 2.56, 3.43, 2.01, 0.24, 1.81, 4.47, 1.56, 4.30, 0.83,
                4.06, 0.11,
            ],
            [
                3.37, 2.50, 1.40, 2.74, 3.71, 2.22, 2.94, 4.21, 4.35, 0.00, 3.00, 1.72, 1.97, 0.25,
                1.24, 1.97, 3.24, 1.26, 0.52, 2.61, 0.70, 2.87, 2.60, 0.34, 2.66, 3.69, 0.39, 2.99,
                1.25, 0.23,
            ],
            [
                0.55, 0.59, 1.72, 4.28, 3.86, 4.45, 0.25, 4.47, 4.04, 3.00, 0.00, 1.73, 3.14, 4.79,
                0.70, 1.05, 0.04, 4.85, 1.13, 3.79, 3.10, 0.86, 3.13, 1.99, 4.86, 0.76, 4.61, 0.87,
                0.76, 3.25,
            ],
            [
                0.92, 2.45, 2.36, 1.47, 0.21, 3.44, 0.51, 1.86, 2.02, 1.72, 1.73, 0.00, 3.41, 2.24,
                0.20, 4.08, 3.17, 0.96, 3.96, 0.64, 3.33, 0.72, 4.03, 4.34, 3.68, 1.67, 3.91, 2.50,
                4.24, 3.76,
            ],
            [
                0.09, 4.81, 0.63, 4.66, 3.62, 2.80, 0.39, 1.12, 1.47, 1.97, 3.14, 3.41, 0.00, 4.35,
                4.14, 0.90, 2.32, 1.29, 3.29, 3.50, 1.03, 2.66, 2.15, 3.32, 3.55, 2.33, 2.39, 0.23,
                4.60, 4.48,
            ],
            [
                0.89, 4.86, 3.13, 3.14, 0.62, 0.81, 0.45, 4.92, 4.28, 0.25, 4.79, 2.24, 4.35, 0.00,
                1.22, 2.79, 4.40, 4.37, 4.04, 2.30, 1.14, 1.57, 2.34, 2.92, 3.91, 3.73, 1.16, 1.07,
                0.92, 0.42,
            ],
            [
                1.54, 1.71, 3.31, 2.23, 0.71, 2.67, 1.69, 3.44, 3.72, 1.24, 0.70, 0.20, 4.14, 1.22,
                0.00, 3.91, 0.29, 3.67, 0.62, 3.69, 1.51, 2.64, 4.53, 2.87, 0.63, 4.51, 0.26, 0.77,
                3.59, 3.25,
            ],
            [
                0.63, 0.68, 3.65, 3.00, 1.37, 2.95, 3.43, 3.67, 2.13, 1.97, 1.05, 4.08, 0.90, 2.79,
                3.91, 0.00, 0.35, 0.08, 2.78, 0.82, 1.87, 1.13, 2.79, 4.94, 3.20, 1.62, 1.28, 4.40,
                3.62, 0.67,
            ],
            [
                2.04, 2.78, 0.86, 3.28, 3.97, 1.64, 1.11, 4.36, 0.03, 3.24, 0.04, 3.17, 2.32, 4.40,
                0.29, 0.35, 0.00, 0.72, 3.74, 2.29, 2.35, 0.03, 4.84, 2.88, 1.52, 2.54, 3.54, 2.08,
                1.15, 2.90,
            ],
            [
                1.95, 4.21, 3.84, 0.07, 4.83, 0.71, 0.45, 1.53, 1.62, 1.26, 4.85, 0.96, 1.29, 4.37,
                3.67, 0.08, 0.72, 0.00, 1.15, 0.42, 3.81, 0.09, 4.51, 2.17, 4.44, 3.27, 1.80, 2.24,
                1.43, 0.07,
            ],
            [
                1.51, 3.53, 0.46, 4.61, 1.38, 3.28, 3.53, 0.31, 3.74, 0.52, 1.13, 3.96, 3.29, 4.04,
                0.62, 2.78, 3.74, 1.15, 0.00, 3.86, 2.27, 0.90, 2.64, 2.53, 3.87, 3.11, 1.73, 3.74,
                0.61, 3.64,
            ],
            [
                1.99, 0.95, 2.49, 3.73, 3.46, 3.57, 4.63, 2.23, 2.56, 2.61, 3.79, 0.64, 3.50, 2.30,
                3.69, 0.82, 2.29, 0.42, 3.86, 0.00, 4.33, 0.43, 4.52, 1.18, 4.96, 1.14, 3.76, 2.68,
                2.40, 1.48,
            ],
            [
                3.01, 2.38, 3.19, 0.62, 1.24, 0.31, 3.68, 0.41, 3.43, 0.70, 3.10, 3.33, 1.03, 1.14,
                1.51, 1.87, 2.35, 3.81, 2.27, 4.33, 0.00, 2.79, 0.55, 0.89, 2.57, 1.66, 3.83, 3.94,
                4.42, 4.16,
            ],
            [
                3.06, 1.99, 2.59, 2.06, 4.90, 4.68, 2.75, 4.83, 2.01, 2.87, 0.86, 0.72, 2.66, 1.57,
                2.64, 1.13, 0.03, 0.09, 0.90, 0.43, 2.79, 0.00, 1.10, 4.74, 0.55, 3.11, 2.70, 0.08,
                2.47, 2.10,
            ],
            [
                1.28, 4.57, 2.41, 2.85, 0.47, 3.58, 2.72, 1.03, 0.24, 2.60, 3.13, 4.03, 2.15, 2.34,
                4.53, 2.79, 4.84, 4.51, 2.64, 4.52, 0.55, 1.10, 0.00, 0.02, 0.04, 1.44, 4.21, 4.66,
                2.14, 1.87,
            ],
            [
                2.33, 2.24, 4.93, 2.83, 3.98, 1.54, 0.43, 2.85, 1.81, 0.34, 1.99, 4.34, 3.32, 2.92,
                2.87, 4.94, 2.88, 2.17, 2.53, 1.18, 0.89, 4.74, 0.02, 0.00, 3.68, 4.34, 3.15, 3.48,
                0.11, 3.72,
            ],
            [
                4.40, 3.31, 4.24, 1.92, 3.63, 4.11, 2.35, 4.94, 4.47, 2.66, 4.86, 3.68, 3.55, 3.91,
                0.63, 3.20, 1.52, 4.44, 3.87, 4.96, 2.57, 0.55, 0.04, 3.68, 0.00, 3.22, 2.47, 3.41,
                2.32, 4.86,
            ],
            [
                1.92, 1.23, 0.06, 2.68, 4.32, 3.17, 2.87, 3.51, 1.56, 3.69, 0.76, 1.67, 2.33, 3.73,
                4.51, 1.62, 2.54, 3.27, 3.11, 1.14, 1.66, 3.11, 1.44, 4.34, 3.22, 0.00, 1.57, 4.16,
                2.53, 2.38,
            ],
            [
                1.68, 3.66, 0.41, 0.51, 0.77, 2.40, 2.45, 0.03, 4.30, 0.39, 4.61, 3.91, 2.39, 1.16,
                0.26, 1.28, 3.54, 1.80, 1.73, 3.76, 3.83, 2.70, 4.21, 3.15, 2.47, 1.57, 0.00, 1.18,
                0.74, 0.12,
            ],
            [
                0.59, 2.81, 4.93, 3.81, 1.37, 3.05, 0.39, 2.31, 0.83, 2.99, 0.87, 2.50, 0.23, 1.07,
                0.77, 4.40, 2.08, 2.24, 3.74, 2.68, 3.94, 0.08, 4.66, 3.48, 3.41, 4.16, 1.18, 0.00,
                4.50, 2.11,
            ],
            [
                3.76, 0.40, 3.74, 1.24, 2.61, 3.78, 2.95, 1.78, 4.06, 1.25, 0.76, 4.24, 4.60, 0.92,
                3.59, 3.62, 1.15, 1.43, 0.61, 2.40, 4.42, 2.47, 2.14, 0.11, 2.32, 2.53, 0.74, 4.50,
                0.00, 3.06,
            ],
            [
                0.62, 0.97, 3.64, 4.40, 1.22, 3.92, 0.06, 4.99, 0.11, 0.23, 3.25, 3.76, 4.48, 0.42,
                3.25, 0.67, 2.90, 0.07, 3.64, 1.48, 4.16, 2.10, 1.87, 3.72, 4.86, 2.38, 0.12, 2.11,
                3.06, 0.00,
            ],
        ]);

        // Expected values from Java NeighborNetSplitWeightsClean.computeUse1D
        // with ACTIVESET method and the ordering from compute_order_huson_2023
        let y_exp = vec![
            0.0,
            0.33386944261543233,
            0.3388879350668599,
            0.34270578519337663,
            0.36004532168121506,
            0.19453441263285187,
            0.0383139723608989,
            0.23196817065268807,
            0.0,
            0.35370325885845616,
            0.3742063759667578,
            0.0,
            0.3247900488315818,
            0.02663901854944131,
            0.027685556753272243,
            0.3783462261828869,
            0.8045129549245084,
            0.15811390269633202,
            0.0,
            0.0,
            0.39149572300317187,
            0.45566513536557873,
            0.4749607477522994,
            0.055623495448245734,
            0.33697330076439175,
            0.060915026392566844,
            0.0,
            0.8483261951317439,
            0.07733895917394058,
            0.10458253668716265,
            0.14229905956495945,
            0.12453554187286958,
            0.0,
            0.23411912188430822,
            0.44371695880556306,
            0.05375501566480712,
            0.0,
            0.14270781644715672,
            0.0,
            0.39225683487013785,
            0.06854976748619288,
            0.046240468372369115,
            0.0,
            0.28207713779626364,
            0.13333670074066456,
            0.06231540987610938,
            0.13419545358220625,
            0.0,
            0.29770458994542837,
            0.1458875941098002,
            0.0,
            0.10062965535088406,
            0.05511074838255337,
            0.1622475924203761,
            0.5120970950596928,
            0.0,
            0.1341477048312368,
            0.0,
            0.47635040622977537,
            0.15371010578493824,
            0.0,
            0.04169140403152743,
            0.3348919120595291,
            0.3894732704931614,
            0.04982803207298111,
            0.18578280606056857,
            0.08087565158163217,
            0.0,
            0.09364913619773566,
            0.18494167352188504,
            0.0,
            0.6184984894530491,
            0.02475575701514745,
            0.4166482110425767,
            0.300030180445009,
            0.0,
            0.0,
        ];
        let ord = compute_order_huson_2023(&d);
        let mut params = NNLSParams::default();
        let n = d.shape()[0];
        params.cgnr_iterations = max(50, n * (n - 1) / 2);
        let progress = None; // No progress tracking in this test

        let (_unused, pairs) = compute_use_1d(&ord, &d, &mut params, progress).expect("NNLS solve");

        assert_eq!(pairs.len(), 77);
        let weights = pairs.iter().map(|(_, w)| *w).collect::<Vec<f64>>();
        // Use relaxed tolerance due to known CGNR convergence differences
        // between Rust and Java implementations (both converge to valid solutions)
        compare_float_array(&weights, &y_exp, 1e-8);
    }

    fn compare_float_array(arr1: &[f64], arr2: &[f64], eps: f64) {
        assert_eq!(arr1.len(), arr2.len());
        for (a, b) in arr1.iter().zip(arr2.iter()) {
            assert!((*a - *b).abs() < eps, "got {}, wanted {}", a, b);
        }
    }
}

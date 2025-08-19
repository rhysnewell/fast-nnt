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
    #[arg(short, long, default_value_t = usize::MAX)]
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
            max_iterations: usize::MAX,
            max_time_ms: u64::MAX,
            cgnr_iterations: 5000,
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

            if ok && iters < params.cgnr_iterations {
                break;
            }
            if k_outer > params.max_iterations
                || started.elapsed() >= Duration::from_millis(params.max_time_ms)
            {
                return Ok(());
            }
        }

        x.copy_from_slice(&xstar);
        // projected gradient check at current x
        let (p, r) = (&mut scratch.p, &mut scratch.r);
        eval_gradient(x, d, p, r, n); // p := grad
        debug!("=================================================");
        debug!("Gradient p: {:?}", p);
        // project gradient
        p.par_iter_mut().zip(x.par_iter()).for_each(|(gi, &xi)| {
            if xi == 0.0 {
                *gi = gi.min(0.0)
            }
        });

        let pg = sum_array_squared(p, n);
        if pg < params.proj_grad_bound {
            return Ok(());
        }

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
    let mut num_to_make_active = (f64::from(count as u32) * params.active_set_rho).ceil() as usize;
    if num_to_make_active < 1 {
        num_to_make_active = 1;
    }
    if num_to_make_active > count {
        num_to_make_active = count;
    }

    // Mark chosen split positions active
    for i in 0..num_to_make_active {
        let pos = splits[indices[i]];
        active_set[pos] = true;
    }

    // Blend x toward xstar by tmin on non-active; zero-out active
    for i in 0..npairs {
        if active_set[i] {
            x[i] = 0.0;
        } else {
            x[i] = (1.0 - tmin) * x[i] + tmin * xstar[i];
        }
    }

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

    zero_negative_entries(x);

    // r = d - A x
    calc_ax(x, r, n);
    r.par_iter_mut()
        .zip(d.par_iter())
        .for_each(|(ri, &di)| *ri = di - *ri);

    // z = Aᵀ r; mask actives
    calc_atx(r, z, n);
    mask_elements_branchless(z, active_set);
    p.copy_from_slice(z);

    let mut ztz = sum_array_squared(z, n);
    let mut k = 1usize;

    while k <= max_iters && ztz >= tol {
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

        // for (var i = 0; i < p.length; i++) {
        //     p[i] = z[i] + beta * p[i];
        // }
        // ztz = ztz2;
        for i in 0..p.len() {
            p[i] = z[i] + beta * p[i]; // correct: z + beta * old_p
        }

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
        let mut ord = compute_order_huson_2023(&d);

        // reverse ordering but keep first 2 indices in place
        ord[2..].reverse();
        let mut params = NNLSParams::default();
        let progress = None; // No progress tracking in this test

        let (_unused, pairs) = compute_use_1d(&ord, &d, &mut params, progress).expect("NNLS solve");

        assert_eq!(pairs.len(), 50);
        let weights = pairs.iter().map(|(_, w)| *w).collect::<Vec<f64>>();

        let y_exp = vec![
            0.0,
            0.10148215727947205,
            0.028437836363730214,
            0.0,
            0.0,
            0.047052775085665216,
            0.008853763031522865,
            0.07443643722001457,
            0.0,
            0.06736810055863196,
            0.026822549576149553,
            0.060341035050351,
            0.06858435444370921,
            0.04153424987958803,
            0.09160245732519465,
            0.01210546421230361,
            0.12867548531805645,
            0.0,
            0.09862739100506297,
            0.016628689361152124,
            0.0,
            0.0,
            0.09095016893319723,
            0.0,
            0.09754594263522459,
            0.016205509026668456,
            0.012309965900145266,
            0.020919931657507357,
            0.00907580217061087,
            0.17896233322364297,
            0.011026258136264913,
            0.150249314892246,
            0.007696440192060954,
            0.019619608797314936,
            0.019930903945793096,
            0.09174690622341422,
            0.030615360711634967,
            0.0,
            0.1382751310106152,
            0.0,
            0.008504014141039079,
            0.005364876677063768,
            0.033799900407467734,
            0.022203931806048866,
            0.019651917431310652,
            0.0,
            0.10187592278668167,
            0.09973423762126273,
            0.0,
            0.0,
        ];

        compare_float_array(&weights, &y_exp, 1e-8);
    }

    fn compare_float_array(arr1: &[f64], arr2: &[f64], eps: f64) {
        assert_eq!(arr1.len(), arr2.len());
        for (a, b) in arr1.iter().zip(arr2.iter()) {
            assert!((*a - *b).abs() < eps, "got {}, wanted {}", a, b);
        }
    }
}

use anyhow::Result;
use clap::Args;
use fixedbitset::FixedBitSet;
use ndarray::Array2;
use rayon::prelude::*;
use std::{
    cmp::max,
    time::{Duration, Instant},
};

use crate::splits::asplit::ASplit;

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
#[derive(Clone, Debug, PartialEq)]
pub struct SplitWeights {
    /// x[(i,j)] for 1 ≤ i < j ≤ n, flattened by blocks (1,2..n), (2,3..n), ...
    pub x: Vec<f64>,
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

    // Threshold from ||Aᵀ d|| (reuse `x` as workspace to avoid an extra allocation).
    let mut x = vec![0.0; npairs];
    calc_atx(&d, &mut x, n);
    let norm_atd = sum_array_squared(&x, n).sqrt();
    params.proj_grad_bound = (1e-4 * norm_atd).powi(2);

    // var x = new double[npairs]; //array of split weights
    // calcAinv_y(d, x, n); //Compute unconstrained solution
    // var minVal = minArray(x);
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
        let mut xstar = vec![0.0; npairs];

        params.cgnr_tolerance = params.proj_grad_bound / 2.0;
        params.cgnr_iterations = max(params.cgnr_iterations, n * (n - 1) / 2);
        params.active_set_rho = 0.4;

        active_set_method(
            &mut x,
            &mut xstar,
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
    let mut out_pairs = Vec::with_capacity(npairs);
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
    x: &mut [f64],      // feasible (x >= 0)
    xstar: &mut [f64],  // workspace for candidate solution
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
    let mut k_outer = 0usize;
    debug!("active set {:?}", active);
    debug!("==============================");
    debug!("x: {:?}", x);
    let start = Instant::now();
    loop {
        debug!("Outer Loop {}", k_outer);
        loop {
            xstar[..npairs].copy_from_slice(x);

            let iters = cgnr(
                xstar,
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
                x,
                &xstar[..npairs],
                active,
                tmp_splits,
                idx_sorted,
                vals,
                n,
                params,
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

        x.copy_from_slice(&xstar[..npairs]);
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

    // Length checks to match Java's expectations.
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
        let ord = match (va.is_nan(), vb.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater, // NaN after numbers
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => va.total_cmp(&vb), // IEEE total order: -0.0 < +0.0
        };
        if ord == std::cmp::Ordering::Equal {
            a.cmp(&b) // match Java's stable sort behavior on ties
        } else {
            ord
        }
    });

    let tmin = vals[indices[0]];

    // Java: max(1, ceil(count * activeSetRho))
    let num_to_make_active =
        max(1, (count as f64 * params.active_set_rho).ceil() as usize);

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
    let mut s_index = 1usize;
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
        if *xi <= 0.0 {
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
    fn pair_idx_n5() {
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
        params.cutoff = 0.0;
        params.max_iterations = usize::MAX;
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
        params.cutoff = 0.0;
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
    #[test]
    fn smoke_50() {
        let d = arr2(&[
            [0.000000, 2.083099, 0.050846, 4.126033, 1.493199, 1.842058, 0.968307, 2.830041, 0.808439, 0.621334, 2.164681, 2.810392, 0.871718, 2.766105, 1.774507, 4.790324, 0.456470, 4.893200, 2.060597, 2.519677, 0.740731, 3.594836, 0.949857, 1.707802, 0.117606, 1.697589, 4.837412, 4.893992, 3.722650, 0.017273, 4.701193, 4.353835, 3.854172, 0.894369, 0.497498, 2.072662, 4.427682, 2.890430, 3.682911, 1.163106, 2.617988, 3.546932, 4.124170, 4.035623, 1.161541, 4.365950, 1.081902, 4.009496, 2.775426, 0.929142],
            [2.083099, 0.000000, 2.943043, 2.591197, 4.793313, 0.207696, 0.820691, 4.916463, 4.161024, 0.751362, 1.145565, 2.695691, 0.784904, 1.618804, 0.246596, 3.558386, 0.392814, 4.973021, 4.602849, 3.944144, 3.862029, 1.838762, 3.393531, 3.806294, 2.403881, 0.307625, 3.130520, 1.628804, 3.050445, 2.104142, 4.762654, 1.007973, 3.844282, 3.492809, 2.782815, 0.400231, 0.824759, 4.836410, 1.153213, 0.817715, 1.441478, 2.685603, 2.598067, 0.016659, 0.015925, 2.480332, 0.924272, 3.037504, 3.992315, 0.479366],
            [0.050846, 2.943043, 0.000000, 2.525836, 2.030691, 3.952266, 4.773964, 2.083591, 1.216339, 4.392367, 0.136379, 1.652329, 1.973876, 0.889919, 3.399674, 2.591331, 1.244703, 2.066254, 3.459417, 1.624820, 0.338530, 2.302515, 0.975322, 3.106563, 1.009447, 0.675129, 4.469806, 1.651440, 1.461526, 2.691295, 3.466910, 3.354986, 3.991970, 1.299681, 2.134380, 1.819801, 2.583435, 2.870231, 3.146441, 0.339770, 1.856087, 3.542820, 0.176636, 3.731191, 3.429507, 2.008957, 2.245856, 1.321704, 3.686105, 2.402027],
            [4.126033, 2.591197, 2.525836, 0.000000, 1.325647, 0.146763, 1.725357, 4.604960, 3.380725, 3.128012, 4.669404, 0.153025, 3.012282, 0.939754, 1.984721, 0.909177, 1.745649, 4.506892, 3.626961, 4.231458, 0.377419, 1.504217, 3.920803, 0.202256, 0.948612, 0.409284, 4.087781, 1.161231, 2.121459, 0.069281, 0.568333, 2.569981, 3.956553, 1.705134, 1.596742, 0.204061, 0.498080, 0.742574, 1.031289, 2.849244, 1.794467, 4.888635, 1.420550, 4.281398, 0.458871, 2.638024, 2.747372, 3.286480, 2.050681, 2.619946],
            [1.493199, 4.793313, 2.030691, 1.325647, 0.000000, 3.889062, 0.758009, 2.658097, 2.940407, 3.908485, 0.818556, 2.072932, 0.122846, 3.312780, 4.340148, 1.767638, 4.081329, 3.290143, 0.085193, 2.351200, 0.228695, 1.566110, 4.003195, 0.627301, 2.826848, 0.283594, 0.549128, 4.760821, 0.482037, 0.870570, 4.700376, 1.544254, 1.082453, 0.482278, 2.496302, 0.294456, 1.100267, 1.110592, 3.725730, 0.810652, 2.940689, 1.208457, 4.969106, 4.953191, 4.619268, 4.500418, 2.768470, 4.654278, 3.199300, 0.796954],
            [1.842058, 0.207696, 3.952266, 0.146763, 3.889062, 0.000000, 3.661743, 4.246432, 0.072258, 0.974596, 3.921501, 3.840249, 2.802903, 0.611550, 1.326266, 1.178577, 4.902919, 1.126624, 1.850611, 3.445164, 3.315832, 2.810962, 3.253383, 4.325031, 4.569778, 0.542235, 0.150641, 4.567952, 4.551327, 1.839898, 1.998486, 3.424549, 2.107546, 0.555139, 1.246107, 1.518599, 2.830981, 0.856055, 3.643166, 4.848945, 1.924223, 2.851574, 3.181232, 3.989907, 3.145567, 2.582601, 3.602512, 0.118775, 4.698965, 4.493163],
            [0.968307, 0.820691, 4.773964, 1.725357, 0.758009, 3.661743, 0.000000, 3.469088, 1.008095, 3.128021, 0.140620, 1.988497, 1.076847, 2.806188, 1.252136, 0.461404, 3.969321, 1.256358, 2.798551, 0.082743, 2.817312, 3.946248, 3.924964, 1.641947, 0.642020, 3.047382, 1.532537, 1.429082, 1.050572, 1.966609, 3.411028, 2.007717, 3.232236, 3.639121, 4.051720, 2.853955, 4.951584, 2.510368, 2.685918, 3.291623, 4.856524, 3.509735, 4.584169, 0.793314, 2.109990, 3.517569, 3.663579, 4.150095, 1.331787, 1.573582],
            [2.830041, 4.916463, 2.083591, 4.604960, 2.658097, 4.246432, 3.469088, 0.000000, 2.063615, 1.961836, 0.463492, 1.948803, 3.211942, 3.697351, 1.149378, 3.103388, 3.755521, 0.777613, 0.071614, 0.800625, 3.394652, 4.566601, 1.936312, 1.559989, 3.136317, 1.580937, 0.196053, 1.226418, 3.002709, 2.289003, 1.734495, 1.960818, 0.685594, 3.837018, 1.870163, 3.449871, 2.838960, 4.500559, 3.982228, 0.661444, 2.257120, 0.055733, 3.775453, 1.153789, 4.882942, 1.127138, 2.541062, 2.457941, 4.136656, 2.975859],
            [0.808439, 4.161024, 1.216339, 3.380725, 2.940407, 0.072258, 1.008095, 2.063615, 0.000000, 4.405491, 1.598200, 0.487421, 0.928568, 0.231419, 3.536841, 3.430297, 3.773308, 3.904385, 0.625794, 0.320760, 3.295397, 0.005733, 1.685306, 2.938313, 3.262147, 3.077987, 4.746961, 4.873110, 1.552687, 1.574961, 0.107120, 0.605307, 2.190872, 3.208581, 1.178403, 0.497168, 3.424718, 3.361095, 2.501922, 0.704752, 0.984555, 1.340501, 4.263977, 4.630752, 0.235811, 4.683800, 4.563435, 3.964328, 1.166393, 4.834951],
            [0.621334, 0.751362, 4.392367, 3.128012, 3.908485, 0.974596, 3.128021, 1.961836, 4.405491, 0.000000, 4.043318, 0.696852, 0.178921, 1.889984, 2.897784, 0.123258, 4.773105, 3.364000, 2.583407, 2.459671, 0.523744, 3.738895, 3.707664, 0.112139, 4.034894, 1.771640, 4.665651, 1.710337, 1.059306, 0.979114, 1.657711, 0.353673, 0.812508, 2.362602, 2.760977, 2.787656, 0.660017, 0.123047, 0.733614, 4.497381, 0.681709, 3.418311, 2.777490, 3.000363, 4.684519, 1.278322, 3.512744, 2.587697, 1.842692, 2.005479],
            [2.164681, 1.145565, 0.136379, 4.669404, 0.818556, 3.921501, 0.140620, 0.463492, 1.598200, 4.043318, 0.000000, 1.011807, 0.504173, 4.606767, 3.201599, 0.849250, 4.015566, 0.923628, 3.311311, 2.933227, 3.983882, 2.061272, 4.776040, 3.595437, 2.245929, 0.605340, 1.290511, 0.661742, 2.605812, 4.029272, 3.710895, 3.185374, 2.134257, 1.392321, 4.019396, 3.356059, 3.318026, 2.791079, 3.953266, 2.944387, 1.316682, 1.867702, 1.548693, 1.458351, 4.492104, 3.557789, 3.068133, 2.254528, 2.426935, 0.525705],
            [2.810392, 2.695691, 1.652329, 0.153025, 2.072932, 3.840249, 1.988497, 1.948803, 0.487421, 0.696852, 1.011807, 0.000000, 2.372768, 2.939088, 1.751349, 2.536163, 4.996453, 4.564953, 0.385118, 4.730114, 3.972404, 4.061501, 4.023019, 1.043263, 0.660094, 3.209647, 4.139392, 0.509898, 0.071907, 4.734895, 4.550103, 1.307197, 4.145247, 1.492921, 4.207600, 0.064862, 3.041952, 1.123777, 3.179313, 0.125321, 3.236129, 0.498946, 2.218099, 0.349861, 1.917992, 0.779959, 1.639859, 2.347803, 3.295570, 3.168070],
            [0.871718, 0.784904, 1.973876, 3.012282, 0.122846, 2.802903, 1.076847, 3.211942, 0.928568, 0.178921, 0.504173, 2.372768, 0.000000, 2.617689, 4.263746, 2.054094, 0.720749, 3.426407, 1.785865, 4.673849, 1.950945, 0.069242, 1.719275, 4.798238, 2.389933, 4.114254, 1.839786, 3.042083, 0.379596, 1.012393, 0.520167, 2.012555, 1.617164, 2.789029, 3.243843, 1.199651, 1.340789, 0.122917, 0.608328, 1.907236, 4.496164, 0.941365, 2.039012, 0.064113, 2.943902, 2.104041, 1.655997, 1.949184, 1.363645, 2.040411],
            [2.766105, 1.618804, 0.889919, 0.939754, 3.312780, 0.611550, 2.806188, 3.697351, 0.231419, 1.889984, 4.606767, 2.939088, 2.617689, 0.000000, 0.923871, 3.562778, 4.665679, 4.534024, 1.832122, 0.058138, 3.844555, 2.307730, 2.726214, 3.278975, 4.058022, 2.589716, 2.172113, 0.657280, 0.209293, 0.827049, 0.773329, 4.704005, 2.834260, 0.137171, 1.591118, 2.059918, 4.172370, 2.066375, 2.734092, 1.320249, 1.615714, 0.398736, 3.914670, 2.444876, 3.656967, 2.860831, 0.444975, 1.344222, 1.600837, 3.915768],
            [1.774507, 0.246596, 3.399674, 1.984721, 4.340148, 1.326266, 1.252136, 1.149378, 3.536841, 2.897784, 3.201599, 1.751349, 4.263746, 0.923871, 0.000000, 1.732424, 4.828196, 3.983427, 3.453780, 2.657962, 3.461890, 1.693615, 0.511142, 4.291303, 2.297718, 4.791260, 3.659892, 2.871887, 2.135383, 3.430644, 1.211888, 1.765969, 1.598194, 4.092805, 4.749156, 0.485897, 0.122525, 1.763462, 3.413123, 2.188744, 3.800260, 4.840786, 3.331427, 3.330417, 0.715804, 2.925193, 3.138015, 0.976060, 0.129495, 3.391057],
            [4.790324, 3.558386, 2.591331, 0.909177, 1.767638, 1.178577, 0.461404, 3.103388, 3.430297, 0.123258, 0.849250, 2.536163, 2.054094, 3.562778, 1.732424, 0.000000, 4.966340, 3.233366, 1.003881, 4.352046, 1.587983, 1.706569, 4.067263, 3.970593, 4.655730, 4.423681, 0.011825, 1.143709, 2.914297, 1.158886, 4.255402, 3.681684, 3.780185, 4.719598, 3.538373, 4.817115, 2.332493, 0.679221, 2.583009, 4.743650, 3.071119, 1.610785, 0.492544, 0.670765, 0.613068, 3.553422, 1.642122, 2.173035, 4.877567, 1.660654],
            [0.456470, 0.392814, 1.244703, 1.745649, 4.081329, 4.902919, 3.969321, 3.755521, 3.773308, 4.773105, 4.015566, 4.996453, 0.720749, 4.665679, 4.828196, 4.966340, 0.000000, 3.267521, 1.745286, 1.928095, 2.132466, 2.222841, 0.584439, 3.208879, 2.781171, 0.558233, 4.872220, 4.935031, 1.374616, 1.703197, 3.903224, 2.530946, 2.832780, 3.822075, 3.758308, 1.326859, 2.435654, 4.052261, 4.414208, 4.330584, 3.998415, 3.291124, 0.130551, 1.828619, 3.545313, 1.214739, 0.314848, 2.969883, 3.841693, 4.980706],
            [4.893200, 4.973021, 2.066254, 4.506892, 3.290143, 1.126624, 1.256358, 0.777613, 3.904385, 3.364000, 0.923628, 4.564953, 3.426407, 4.534024, 3.983427, 3.233366, 3.267521, 0.000000, 3.641058, 2.243493, 0.160207, 3.399542, 4.462035, 2.590028, 2.813303, 0.461014, 1.603961, 2.960484, 4.014551, 1.622778, 4.158773, 2.941000, 2.774742, 1.004702, 3.475507, 0.684901, 2.102238, 4.432560, 4.339593, 4.722539, 0.968225, 4.522424, 2.343701, 0.822243, 4.481930, 4.751217, 4.866064, 1.792296, 1.124578, 4.623410],
            [2.060597, 4.602849, 3.459417, 3.626961, 0.085193, 1.850611, 2.798551, 0.071614, 0.625794, 2.583407, 3.311311, 0.385118, 1.785865, 1.832122, 3.453780, 1.003881, 1.745286, 3.641058, 0.000000, 3.563937, 4.627495, 3.647081, 4.923171, 3.650093, 4.099420, 4.614288, 2.148699, 4.599224, 2.068554, 4.805041, 2.710870, 3.204319, 0.048825, 1.697063, 1.374924, 0.059988, 1.565181, 1.285981, 0.543857, 1.278023, 4.301854, 1.077621, 3.499382, 4.959523, 0.045401, 3.555432, 0.672072, 0.924077, 1.539700, 4.589629],
            [2.519677, 3.944144, 1.624820, 4.231458, 2.351200, 3.445164, 0.082743, 0.800625, 0.320760, 2.459671, 2.933227, 4.730114, 4.673849, 0.058138, 2.657962, 4.352046, 1.928095, 2.243493, 3.563937, 0.000000, 4.381249, 0.458791, 3.950612, 0.422894, 1.793485, 4.998987, 4.398131, 0.548729, 2.835365, 2.102860, 3.375525, 4.293317, 2.788302, 4.660664, 4.481979, 0.817374, 0.284792, 0.989932, 2.539555, 3.474871, 4.271650, 3.655394, 3.271691, 3.004884, 3.270524, 2.000552, 3.313997, 4.181303, 4.282303, 3.593027],
            [0.740731, 3.862029, 0.338530, 0.377419, 0.228695, 3.315832, 2.817312, 3.394652, 3.295397, 0.523744, 3.983882, 3.972404, 1.950945, 3.844555, 3.461890, 1.587983, 2.132466, 0.160207, 4.627495, 4.381249, 0.000000, 3.378076, 0.361692, 1.639821, 2.278053, 1.535280, 0.991293, 1.873366, 4.167391, 1.336071, 4.677672, 0.031640, 2.109784, 3.606504, 3.774173, 0.059665, 0.575769, 0.353973, 2.825781, 0.473574, 1.981864, 1.411139, 0.512388, 1.873474, 2.177559, 2.891830, 1.726357, 0.344491, 1.644828, 3.470495],
            [3.594836, 1.838762, 2.302515, 1.504217, 1.566110, 2.810962, 3.946248, 4.566601, 0.005733, 3.738895, 2.061272, 4.061501, 0.069242, 2.307730, 1.693615, 1.706569, 2.222841, 3.399542, 3.647081, 0.458791, 3.378076, 0.000000, 0.017887, 2.609159, 0.097614, 4.321069, 0.983310, 2.668085, 0.456843, 2.736497, 3.964177, 4.572914, 3.948562, 4.468446, 0.038903, 4.028940, 0.245057, 4.352786, 4.981622, 3.588236, 3.031650, 3.666109, 0.433821, 4.317154, 0.437056, 1.193704, 1.350362, 3.437918, 1.967995, 1.929464],
            [0.949857, 3.393531, 0.975322, 3.920803, 4.003195, 3.253383, 3.924964, 1.936312, 1.685306, 3.707664, 4.776040, 4.023019, 1.719275, 2.726214, 0.511142, 4.067263, 0.584439, 4.462035, 4.923171, 3.950612, 0.361692, 0.017887, 0.000000, 2.509926, 3.766515, 4.199139, 4.225532, 1.416294, 0.262051, 0.471615, 0.574763, 3.728410, 0.421260, 3.782438, 0.008961, 3.434525, 1.236576, 3.224729, 3.801444, 3.272783, 0.369979, 0.103684, 1.913397, 4.535039, 2.322416, 1.313940, 2.353268, 4.102281, 1.035441, 0.617023],
            [1.707802, 3.806294, 3.106563, 0.202256, 0.627301, 4.325031, 1.641947, 1.559989, 2.938313, 0.112139, 3.595437, 1.043263, 4.798238, 3.278975, 4.291303, 3.970593, 3.208879, 2.590028, 3.650093, 0.422894, 1.639821, 2.609159, 2.509926, 0.000000, 4.592329, 2.120424, 2.218981, 1.554557, 1.222315, 4.309737, 4.012469, 1.270282, 3.624890, 4.595206, 2.088681, 4.904860, 0.454036, 4.188654, 4.254023, 1.261258, 1.384162, 1.359837, 0.682987, 4.055521, 3.491883, 3.940085, 3.210564, 1.307008, 0.435723, 1.292333],
            [0.117606, 2.403881, 1.009447, 0.948612, 2.826848, 4.569778, 0.642020, 3.136317, 3.262147, 4.034894, 2.245929, 0.660094, 2.389933, 4.058022, 2.297718, 4.655730, 2.781171, 2.813303, 4.099420, 1.793485, 2.278053, 0.097614, 3.766515, 4.592329, 0.000000, 3.941569, 0.098204, 2.667198, 1.607111, 1.442553, 1.673791, 2.958178, 0.844676, 0.795236, 4.247840, 3.772415, 1.798785, 1.243916, 2.922808, 4.662416, 3.940595, 2.215494, 3.172848, 1.796232, 3.546520, 2.589573, 0.135929, 3.369717, 4.357896, 2.423597],
            [1.697589, 0.307625, 0.675129, 0.409284, 0.283594, 0.542235, 3.047382, 1.580937, 3.077987, 1.771640, 0.605340, 3.209647, 4.114254, 2.589716, 4.791260, 4.423681, 0.558233, 0.461014, 4.614288, 4.998987, 1.535280, 4.321069, 4.199139, 2.120424, 3.941569, 0.000000, 0.800223, 1.516854, 0.961946, 4.199752, 3.677097, 1.486750, 1.512745, 0.631583, 2.201320, 4.880346, 3.784540, 4.677788, 0.371939, 4.627633, 4.564238, 4.054163, 0.640635, 4.344282, 4.890442, 3.701706, 2.572788, 1.714666, 4.392189, 2.992629],
            [4.837412, 3.130520, 4.469806, 4.087781, 0.549128, 0.150641, 1.532537, 0.196053, 4.746961, 4.665651, 1.290511, 4.139392, 1.839786, 2.172113, 3.659892, 0.011825, 4.872220, 1.603961, 2.148699, 4.398131, 0.991293, 0.983310, 4.225532, 2.218981, 0.098204, 0.800223, 0.000000, 1.615138, 0.411229, 0.086959, 3.510005, 1.478137, 0.661056, 1.535089, 2.310270, 4.235368, 2.370848, 0.643841, 4.142376, 3.372345, 2.729877, 4.491861, 2.026289, 1.935688, 0.273804, 3.478773, 3.108604, 4.737476, 0.039201, 2.064366],
            [4.893992, 1.628804, 1.651440, 1.161231, 4.760821, 4.567952, 1.429082, 1.226418, 4.873110, 1.710337, 0.661742, 0.509898, 3.042083, 0.657280, 2.871887, 1.143709, 4.935031, 2.960484, 4.599224, 0.548729, 1.873366, 2.668085, 1.416294, 1.554557, 2.667198, 1.516854, 1.615138, 0.000000, 2.186948, 2.076667, 0.216597, 4.073039, 0.214858, 3.200893, 1.562647, 2.123130, 4.189724, 1.020926, 2.557500, 1.333616, 3.050444, 2.721004, 3.336726, 0.477325, 3.163292, 3.310859, 0.338746, 4.845981, 4.253875, 3.965593],
            [3.722650, 3.050445, 1.461526, 2.121459, 0.482037, 4.551327, 1.050572, 3.002709, 1.552687, 1.059306, 2.605812, 0.071907, 0.379596, 0.209293, 2.135383, 2.914297, 1.374616, 4.014551, 2.068554, 2.835365, 4.167391, 0.456843, 0.262051, 1.222315, 1.607111, 0.961946, 0.411229, 2.186948, 0.000000, 3.932170, 2.500630, 4.576739, 0.775054, 3.718592, 4.883214, 0.313779, 3.243917, 4.866038, 0.708879, 2.795682, 4.489314, 1.318777, 1.050042, 3.588635, 3.971566, 4.708564, 3.262776, 0.269246, 4.105864, 1.429849],
            [0.017273, 2.104142, 2.691295, 0.069281, 0.870570, 1.839898, 1.966609, 2.289003, 1.574961, 0.979114, 4.029272, 4.734895, 1.012393, 0.827049, 3.430644, 1.158886, 1.703197, 1.622778, 4.805041, 2.102860, 1.336071, 2.736497, 0.471615, 4.309737, 1.442553, 4.199752, 0.086959, 2.076667, 3.932170, 0.000000, 2.817587, 3.604432, 4.890757, 2.756903, 0.723893, 2.558683, 2.774625, 3.760477, 0.955294, 2.758115, 0.248238, 4.925072, 0.113111, 2.705121, 0.872852, 1.373749, 2.011227, 1.265440, 1.305557, 1.543273],
            [4.701193, 4.762654, 3.466910, 0.568333, 4.700376, 1.998486, 3.411028, 1.734495, 0.107120, 1.657711, 3.710895, 4.550103, 0.520167, 0.773329, 1.211888, 4.255402, 3.903224, 4.158773, 2.710870, 3.375525, 4.677672, 3.964177, 0.574763, 4.012469, 1.673791, 3.677097, 3.510005, 0.216597, 2.500630, 2.817587, 0.000000, 2.495977, 4.782225, 1.045972, 0.863213, 0.214456, 4.578191, 3.915264, 4.152762, 0.747778, 2.207104, 3.357418, 2.656650, 3.661635, 0.971072, 4.512880, 2.959763, 3.059417, 2.930140, 2.214446],
            [4.353835, 1.007973, 3.354986, 2.569981, 1.544254, 3.424549, 2.007717, 1.960818, 0.605307, 0.353673, 3.185374, 1.307197, 2.012555, 4.704005, 1.765969, 3.681684, 2.530946, 2.941000, 3.204319, 4.293317, 0.031640, 4.572914, 3.728410, 1.270282, 2.958178, 1.486750, 1.478137, 4.073039, 4.576739, 3.604432, 2.495977, 0.000000, 0.062873, 0.818657, 2.406008, 0.409545, 2.975067, 0.950124, 3.423171, 2.025043, 3.288015, 1.309853, 0.622928, 3.321603, 4.127570, 1.054949, 0.183257, 4.343107, 3.766284, 3.834404],
            [3.854172, 3.844282, 3.991970, 3.956553, 1.082453, 2.107546, 3.232236, 0.685594, 2.190872, 0.812508, 2.134257, 4.145247, 1.617164, 2.834260, 1.598194, 3.780185, 2.832780, 2.774742, 0.048825, 2.788302, 2.109784, 3.948562, 0.421260, 3.624890, 0.844676, 1.512745, 0.661056, 0.214858, 0.775054, 4.890757, 4.782225, 0.062873, 0.000000, 3.953466, 2.086265, 4.755253, 4.037869, 3.778779, 4.703167, 4.540493, 2.825018, 0.282331, 2.950008, 2.478572, 0.084581, 1.571289, 4.101159, 3.123973, 4.651353, 3.779543],
            [0.894369, 3.492809, 1.299681, 1.705134, 0.482278, 0.555139, 3.639121, 3.837018, 3.208581, 2.362602, 1.392321, 1.492921, 2.789029, 0.137171, 4.092805, 4.719598, 3.822075, 1.004702, 1.697063, 4.660664, 3.606504, 4.468446, 3.782438, 4.595206, 0.795236, 0.631583, 1.535089, 3.200893, 3.718592, 2.756903, 1.045972, 0.818657, 3.953466, 0.000000, 0.928531, 3.566400, 0.234560, 4.346815, 1.471372, 4.855156, 0.238475, 3.712220, 0.506765, 4.315298, 4.086918, 2.561314, 2.815868, 1.099655, 1.822248, 0.741675],
            [0.497498, 2.782815, 2.134380, 1.596742, 2.496302, 1.246107, 4.051720, 1.870163, 1.178403, 2.760977, 4.019396, 4.207600, 3.243843, 1.591118, 4.749156, 3.538373, 3.758308, 3.475507, 1.374924, 4.481979, 3.774173, 0.038903, 0.008961, 2.088681, 4.247840, 2.201320, 2.310270, 1.562647, 4.883214, 0.723893, 0.863213, 2.406008, 2.086265, 0.928531, 0.000000, 1.967487, 0.689721, 3.170717, 2.464658, 1.776816, 1.731286, 4.759734, 4.739103, 4.896174, 1.812906, 0.410021, 0.813845, 4.302149, 1.754678, 0.735244],
            [2.072662, 0.400231, 1.819801, 0.204061, 0.294456, 1.518599, 2.853955, 3.449871, 0.497168, 2.787656, 3.356059, 0.064862, 1.199651, 2.059918, 0.485897, 4.817115, 1.326859, 0.684901, 0.059988, 0.817374, 0.059665, 4.028940, 3.434525, 4.904860, 3.772415, 4.880346, 4.235368, 2.123130, 0.313779, 2.558683, 0.214456, 0.409545, 4.755253, 3.566400, 1.967487, 0.000000, 4.778261, 1.115418, 2.801622, 2.067811, 0.690571, 1.861135, 3.941944, 2.715003, 4.288984, 0.080260, 0.539049, 2.160519, 3.210619, 2.057242],
            [4.427682, 0.824759, 2.583435, 0.498080, 1.100267, 2.830981, 4.951584, 2.838960, 3.424718, 0.660017, 3.318026, 3.041952, 1.340789, 4.172370, 0.122525, 2.332493, 2.435654, 2.102238, 1.565181, 0.284792, 0.575769, 0.245057, 1.236576, 0.454036, 1.798785, 3.784540, 2.370848, 4.189724, 3.243917, 2.774625, 4.578191, 2.975067, 4.037869, 0.234560, 0.689721, 4.778261, 0.000000, 1.380039, 4.071774, 4.264136, 3.246008, 0.662267, 3.704538, 3.037937, 3.674069, 3.678084, 2.816426, 1.922200, 4.187071, 4.627423],
            [2.890430, 4.836410, 2.870231, 0.742574, 1.110592, 0.856055, 2.510368, 4.500559, 3.361095, 0.123047, 2.791079, 1.123777, 0.122917, 2.066375, 1.763462, 0.679221, 4.052261, 4.432560, 1.285981, 0.989932, 0.353973, 4.352786, 3.224729, 4.188654, 1.243916, 4.677788, 0.643841, 1.020926, 4.866038, 3.760477, 3.915264, 0.950124, 3.778779, 4.346815, 3.170717, 1.115418, 1.380039, 0.000000, 4.756819, 1.517611, 1.057309, 0.962971, 3.404366, 4.821456, 1.194845, 2.246382, 2.725632, 2.082387, 2.743286, 1.854499],
            [3.682911, 1.153213, 3.146441, 1.031289, 3.725730, 3.643166, 2.685918, 3.982228, 2.501922, 0.733614, 3.953266, 3.179313, 0.608328, 2.734092, 3.413123, 2.583009, 4.414208, 4.339593, 0.543857, 2.539555, 2.825781, 4.981622, 3.801444, 4.254023, 2.922808, 0.371939, 4.142376, 2.557500, 0.708879, 0.955294, 4.152762, 3.423171, 4.703167, 1.471372, 2.464658, 2.801622, 4.071774, 4.756819, 0.000000, 2.933901, 4.890349, 3.820712, 4.814479, 2.106088, 0.747433, 1.010657, 1.157279, 3.849371, 1.115670, 2.806305],
            [1.163106, 0.817715, 0.339770, 2.849244, 0.810652, 4.848945, 3.291623, 0.661444, 0.704752, 4.497381, 2.944387, 0.125321, 1.907236, 1.320249, 2.188744, 4.743650, 4.330584, 4.722539, 1.278023, 3.474871, 0.473574, 3.588236, 3.272783, 1.261258, 4.662416, 4.627633, 3.372345, 1.333616, 2.795682, 2.758115, 0.747778, 2.025043, 4.540493, 4.855156, 1.776816, 2.067811, 4.264136, 1.517611, 2.933901, 0.000000, 4.636261, 2.106360, 1.185469, 4.437344, 4.895221, 0.340331, 4.113034, 4.871385, 1.482033, 1.791145],
            [2.617988, 1.441478, 1.856087, 1.794467, 2.940689, 1.924223, 4.856524, 2.257120, 0.984555, 0.681709, 1.316682, 3.236129, 4.496164, 1.615714, 3.800260, 3.071119, 3.998415, 0.968225, 4.301854, 4.271650, 1.981864, 3.031650, 0.369979, 1.384162, 3.940595, 4.564238, 2.729877, 3.050444, 4.489314, 0.248238, 2.207104, 3.288015, 2.825018, 0.238475, 1.731286, 0.690571, 3.246008, 1.057309, 4.890349, 4.636261, 0.000000, 0.959004, 1.084279, 3.847565, 2.208437, 1.546430, 1.822734, 4.537025, 4.983816, 1.598022],
            [3.546932, 2.685603, 3.542820, 4.888635, 1.208457, 2.851574, 3.509735, 0.055733, 1.340501, 3.418311, 1.867702, 0.498946, 0.941365, 0.398736, 4.840786, 1.610785, 3.291124, 4.522424, 1.077621, 3.655394, 1.411139, 3.666109, 0.103684, 1.359837, 2.215494, 4.054163, 4.491861, 2.721004, 1.318777, 4.925072, 3.357418, 1.309853, 0.282331, 3.712220, 4.759734, 1.861135, 0.662267, 0.962971, 3.820712, 2.106360, 0.959004, 0.000000, 1.336482, 1.346684, 3.919973, 4.543901, 0.964081, 0.523727, 1.663682, 4.151066],
            [4.124170, 2.598067, 0.176636, 1.420550, 4.969106, 3.181232, 4.584169, 3.775453, 4.263977, 2.777490, 1.548693, 2.218099, 2.039012, 3.914670, 3.331427, 0.492544, 0.130551, 2.343701, 3.499382, 3.271691, 0.512388, 0.433821, 1.913397, 0.682987, 3.172848, 0.640635, 2.026289, 3.336726, 1.050042, 0.113111, 2.656650, 0.622928, 2.950008, 0.506765, 4.739103, 3.941944, 3.704538, 3.404366, 4.814479, 1.185469, 1.084279, 1.336482, 0.000000, 4.236881, 4.750422, 2.100492, 0.216428, 0.229195, 4.032547, 1.873862],
            [4.035623, 0.016659, 3.731191, 4.281398, 4.953191, 3.989907, 0.793314, 1.153789, 4.630752, 3.000363, 1.458351, 0.349861, 0.064113, 2.444876, 3.330417, 0.670765, 1.828619, 0.822243, 4.959523, 3.004884, 1.873474, 4.317154, 4.535039, 4.055521, 1.796232, 4.344282, 1.935688, 0.477325, 3.588635, 2.705121, 3.661635, 3.321603, 2.478572, 4.315298, 4.896174, 2.715003, 3.037937, 4.821456, 2.106088, 4.437344, 3.847565, 1.346684, 4.236881, 0.000000, 0.851791, 1.409121, 0.500474, 4.581500, 2.150855, 2.524315],
            [1.161541, 0.015925, 3.429507, 0.458871, 4.619268, 3.145567, 2.109990, 4.882942, 0.235811, 4.684519, 4.492104, 1.917992, 2.943902, 3.656967, 0.715804, 0.613068, 3.545313, 4.481930, 0.045401, 3.270524, 2.177559, 0.437056, 2.322416, 3.491883, 3.546520, 4.890442, 0.273804, 3.163292, 3.971566, 0.872852, 0.971072, 4.127570, 0.084581, 4.086918, 1.812906, 4.288984, 3.674069, 1.194845, 0.747433, 4.895221, 2.208437, 3.919973, 4.750422, 0.851791, 0.000000, 0.999710, 2.613379, 1.268458, 4.695318, 4.326118],
            [4.365950, 2.480332, 2.008957, 2.638024, 4.500418, 2.582601, 3.517569, 1.127138, 4.683800, 1.278322, 3.557789, 0.779959, 2.104041, 2.860831, 2.925193, 3.553422, 1.214739, 4.751217, 3.555432, 2.000552, 2.891830, 1.193704, 1.313940, 3.940085, 2.589573, 3.701706, 3.478773, 3.310859, 4.708564, 1.373749, 4.512880, 1.054949, 1.571289, 2.561314, 0.410021, 0.080260, 3.678084, 2.246382, 1.010657, 0.340331, 1.546430, 4.543901, 2.100492, 1.409121, 0.999710, 0.000000, 3.592892, 2.821185, 3.652580, 0.423533],
            [1.081902, 0.924272, 2.245856, 2.747372, 2.768470, 3.602512, 3.663579, 2.541062, 4.563435, 3.512744, 3.068133, 1.639859, 1.655997, 0.444975, 3.138015, 1.642122, 0.314848, 4.866064, 0.672072, 3.313997, 1.726357, 1.350362, 2.353268, 3.210564, 0.135929, 2.572788, 3.108604, 0.338746, 3.262776, 2.011227, 2.959763, 0.183257, 4.101159, 2.815868, 0.813845, 0.539049, 2.816426, 2.725632, 1.157279, 4.113034, 1.822734, 0.964081, 0.216428, 0.500474, 2.613379, 3.592892, 0.000000, 1.809051, 4.405796, 3.955704],
            [4.009496, 3.037504, 1.321704, 3.286480, 4.654278, 0.118775, 4.150095, 2.457941, 3.964328, 2.587697, 2.254528, 2.347803, 1.949184, 1.344222, 0.976060, 2.173035, 2.969883, 1.792296, 0.924077, 4.181303, 0.344491, 3.437918, 4.102281, 1.307008, 3.369717, 1.714666, 4.737476, 4.845981, 0.269246, 1.265440, 3.059417, 4.343107, 3.123973, 1.099655, 4.302149, 2.160519, 1.922200, 2.082387, 3.849371, 4.871385, 4.537025, 0.523727, 0.229195, 4.581500, 1.268458, 2.821185, 1.809051, 0.000000, 1.578291, 1.921269],
            [2.775426, 3.992315, 3.686105, 2.050681, 3.199300, 4.698965, 1.331787, 4.136656, 1.166393, 1.842692, 2.426935, 3.295570, 1.363645, 1.600837, 0.129495, 4.877567, 3.841693, 1.124578, 1.539700, 4.282303, 1.644828, 1.967995, 1.035441, 0.435723, 4.357896, 4.392189, 0.039201, 4.253875, 4.105864, 1.305557, 2.930140, 3.766284, 4.651353, 1.822248, 1.754678, 3.210619, 4.187071, 2.743286, 1.115670, 1.482033, 4.983816, 1.663682, 4.032547, 2.150855, 4.695318, 3.652580, 4.405796, 1.578291, 0.000000, 0.696803],
            [0.929142, 0.479366, 2.402027, 2.619946, 0.796954, 4.493163, 1.573582, 2.975859, 4.834951, 2.005479, 0.525705, 3.168070, 2.040411, 3.915768, 3.391057, 1.660654, 4.980706, 4.623410, 4.589629, 3.593027, 3.470495, 1.929464, 0.617023, 1.292333, 2.423597, 2.992629, 2.064366, 3.965593, 1.429849, 1.543273, 2.214446, 3.834404, 3.779543, 0.741675, 0.735244, 2.057242, 4.627423, 1.854499, 2.806305, 1.791145, 1.598022, 4.151066, 1.873862, 2.524315, 4.326118, 0.423533, 3.955704, 1.921269, 0.696803, 0.000000],
        ]);

        let ord = compute_order_huson_2023(&d);
        let mut params = NNLSParams::default();
        let progress = None; // No progress tracking in this test

        let (_unused, pairs) = compute_use_1d(&ord, &d, &mut params, progress).expect("NNLS solve");

        assert_eq!(pairs.len(), 129);
        let weights = pairs.iter().map(|(_, w)| *w).collect::<Vec<f64>>();

        let y_exp = vec![
            0.000000000000000,
            0.161190175166236,
            0.164705732155278,
            0.139528208240468,
            0.028240589079014,
            0.142087911412386,
            0.000000000000000,
            0.000000000000000,
            0.332751247102983,
            0.167850187940077,
            0.336489243977206,
            0.060510574884377,
            0.054784866535271,
            0.000000000000000,
            0.317202467409326,
            0.124794124382408,
            0.148588425762573,
            0.548799667217925,
            0.000000000000000,
            0.219814181113177,
            0.000000000000000,
            0.531985886122720,
            0.000000000000000,
            0.170989426537264,
            0.041956844456512,
            0.196630816392622,
            0.000000000000000,
            0.105875442339658,
            0.000000000000000,
            0.060234661574742,
            0.259529178486389,
            0.055630837822080,
            0.065619491406164,
            0.000000000000000,
            0.516082514789635,
            0.000000000000000,
            0.107545175068810,
            0.115954571223383,
            0.116739458863403,
            0.332294729723632,
            0.000000000000000,
            0.211837016260703,
            0.644698758988559,
            0.000000000000000,
            0.000000000000000,
            0.217559289207700,
            0.134127510319267,
            0.159441056175300,
            0.056130013573385,
            0.157347272282953,
            0.182537079390532,
            0.464115002266020,
            0.003982422631320,
            0.266494355797880,
            0.103168871942282,
            0.000000000000000,
            0.265450991432271,
            0.000000000000000,
            0.111047439821420,
            0.173111470043319,
            0.034823249895385,
            0.000000000000000,
            0.428956537203584,
            0.092290533829459,
            0.000000000000000,
            0.340790081005035,
            0.070130712080645,
            0.096796519415925,
            0.031841768250600,
            0.000000000000000,
            0.303921015509547,
            0.000000000000000,
            0.106860715772064,
            0.250523742087842,
            0.000000000000000,
            0.288724990280000,
            0.019503799419622,
            0.106277340748329,
            0.000000000000000,
            0.121145165640012,
            0.000000000000000,
            0.224706992435671,
            0.000000000000000,
            0.418136098023793,
            0.000000000000000,
            0.000000000000000,
            0.150562770354065,
            0.276365316864724,
            0.126223140690440,
            0.441712075052916,
            0.036354538562347,
            0.357498195910894,
            0.054437589626163,
            0.000000000000000,
            0.000000000000000,
            0.424894265566855,
            0.243884987419416,
            0.011112908254137,
            0.074832941664891,
            0.000000000000000,
            0.118725161475241,
            0.000000000000000,
            0.409250234472879,
            0.000000000000000,
            0.406250143283461,
            0.266696856367982,
            0.000000000000000,
            0.396712649726746,
            0.013587853133640,
            0.026293872096706,
            0.000000000000000,
            0.390407108139497,
            0.000000000000000,
            0.413098496751176,
            0.000000000000000,
            0.288168284738308,
            0.131609201470551,
            0.000000000000000,
            0.000000000000000,
            0.233347965173055,
            0.093402401287672,
            0.809867252802476,
            0.000000000000000,
            0.000000000000000,
            0.230094095177081,
            0.358709575007578,
            0.077240090987548,
            0.350505146120972,
            0.000000000000000,
        ];

        compare_float_array(&weights, &y_exp, 1e-8);
    }

    #[test]
    fn smoke_60() {
        let d = arr2(&[
            [0.000000, 4.837424, 0.832798, 3.332415, 4.828242, 3.191555, 2.997325, 0.227514, 2.959949, 1.841346, 0.766698, 0.298299, 4.471449, 1.911140, 4.326375, 1.102303, 0.950971, 4.002400, 0.593340, 2.671809, 1.253259, 0.936924, 4.533513, 1.929792, 4.528932, 0.394465, 3.011129, 2.278091, 3.891108, 4.247570, 1.235028, 1.940352, 2.623409, 0.906379, 3.268118, 4.162085, 4.732443, 4.382857, 2.071456, 4.663953, 1.353513, 3.393886, 4.247184, 1.794547, 4.833405, 0.135464, 0.376182, 4.576669, 0.080695, 0.276691, 1.296933, 0.935806, 3.551193, 0.716795, 1.801765, 3.505492, 4.780772, 1.909776, 3.279791, 0.578301],
            [4.837424, 0.000000, 1.864613, 4.344745, 0.021442, 2.783183, 4.790510, 2.240472, 3.407910, 0.037062, 2.823816, 2.021936, 4.592926, 1.807576, 3.666022, 2.602430, 4.810026, 0.076716, 2.189595, 0.720286, 1.464143, 1.093856, 2.902376, 3.790560, 0.009445, 4.309982, 4.038246, 0.575106, 1.368263, 0.029719, 1.056487, 2.517721, 0.309230, 4.477829, 3.950387, 4.244929, 0.392727, 0.457889, 0.009377, 0.304110, 0.551220, 3.885758, 0.271506, 0.130646, 4.772263, 2.146542, 0.681796, 1.763619, 3.035987, 3.649296, 4.064706, 4.289919, 2.411282, 0.804869, 2.268771, 2.311098, 2.949519, 4.670592, 1.918577, 1.866127],
            [0.832798, 1.864613, 0.000000, 2.722912, 2.905735, 3.815122, 2.858525, 4.838301, 4.422848, 1.649139, 1.908280, 0.772152, 2.374436, 0.772028, 3.874913, 3.622809, 4.366053, 4.097645, 4.938588, 2.270258, 2.677068, 4.253642, 1.009916, 0.555468, 1.529501, 0.193180, 4.533945, 1.624096, 1.304133, 3.812512, 3.343093, 2.405410, 2.362015, 2.328213, 1.082086, 1.763505, 4.985049, 0.849519, 0.673563, 2.952564, 4.706393, 0.359912, 3.739034, 4.428220, 1.703105, 0.192710, 1.896135, 0.312967, 4.528480, 2.021798, 2.888916, 4.798650, 0.898716, 2.530143, 0.294771, 1.248161, 3.321985, 2.678289, 3.800891, 1.879479],
            [3.332415, 4.344745, 2.722912, 0.000000, 0.081872, 3.424320, 1.408956, 2.538527, 2.193163, 1.546209, 3.857384, 4.445469, 3.790211, 0.457615, 0.662680, 3.692745, 4.092742, 0.883420, 4.264063, 0.903185, 3.547084, 2.405816, 0.220751, 0.627154, 2.180579, 4.281124, 2.945118, 3.114342, 1.850049, 2.123019, 1.894481, 4.804733, 2.076674, 0.260492, 2.040455, 2.849782, 3.045587, 4.962545, 2.297129, 2.541173, 0.140594, 2.014026, 0.162418, 0.102457, 2.037226, 4.626958, 3.888976, 2.280618, 0.381775, 1.042477, 0.633150, 1.152656, 2.302290, 3.700051, 3.720028, 3.157625, 0.321973, 0.627631, 1.208220, 4.698178],
            [4.828242, 0.021442, 2.905735, 0.081872, 0.000000, 0.832429, 4.226340, 0.719857, 4.024577, 4.984613, 3.430924, 1.573615, 3.818073, 0.845160, 1.792025, 1.276251, 4.830157, 2.602544, 1.071495, 1.255187, 2.673863, 4.536155, 3.404383, 0.254200, 0.391035, 2.220042, 4.772773, 1.164473, 0.527870, 4.390809, 4.736401, 0.420039, 4.896554, 3.907651, 0.270675, 0.725090, 1.939082, 4.813466, 4.581269, 1.417995, 2.360987, 1.776643, 2.660236, 3.426088, 3.493518, 4.314176, 3.211678, 1.465118, 3.146517, 4.399113, 4.269318, 4.126538, 4.666094, 3.455189, 3.872908, 1.767299, 0.331327, 4.983391, 1.711828, 0.754905],
            [3.191555, 2.783183, 3.815122, 3.424320, 0.832429, 0.000000, 0.883734, 1.042560, 2.811832, 3.523880, 4.022423, 1.903498, 1.823577, 0.518114, 3.464982, 3.597824, 2.138490, 2.708400, 1.626421, 0.098687, 2.345086, 2.179776, 0.596617, 4.086930, 2.061867, 2.985933, 1.202428, 2.086099, 4.856677, 3.086692, 3.826332, 2.543594, 3.763313, 1.770965, 1.674429, 1.620840, 0.588086, 1.245331, 3.992857, 3.412501, 2.836147, 1.113323, 2.852918, 1.124315, 2.939961, 1.525044, 1.572943, 1.788098, 2.747591, 0.653038, 4.169466, 0.085314, 3.761569, 1.073782, 0.045199, 3.312412, 4.764350, 4.360613, 4.274578, 3.766793],
            [2.997325, 4.790510, 2.858525, 1.408956, 4.226340, 0.883734, 0.000000, 0.227322, 4.245631, 4.462441, 4.596176, 4.261034, 1.375596, 1.878011, 1.659286, 4.031473, 3.722538, 1.078590, 0.689426, 4.554372, 4.497578, 2.858471, 4.032564, 0.447861, 2.155135, 1.614283, 2.274204, 0.084526, 3.201127, 2.569029, 2.494879, 1.303854, 1.561656, 4.692629, 1.577760, 2.067671, 0.655899, 2.507014, 2.583064, 3.697204, 0.152935, 1.243407, 3.326437, 4.272031, 0.989585, 0.091231, 3.922196, 4.582176, 4.181724, 4.290174, 3.154676, 0.938573, 2.758261, 0.267571, 2.316687, 4.405737, 1.749346, 4.632079, 4.591950, 3.679949],
            [0.227514, 2.240472, 4.838301, 2.538527, 0.719857, 1.042560, 0.227322, 0.000000, 1.954741, 0.440733, 2.575604, 1.312586, 2.729541, 1.732530, 0.908909, 0.771296, 3.152719, 2.799418, 1.396849, 4.145377, 3.457965, 3.019040, 1.090368, 1.361116, 4.010932, 2.294353, 4.451566, 2.967579, 3.606828, 4.947365, 1.572366, 4.577603, 2.330103, 1.408378, 2.556421, 4.571382, 3.375118, 0.240776, 4.831864, 3.338249, 1.025402, 2.915041, 0.049472, 0.203018, 3.399892, 0.048765, 4.420342, 3.977592, 4.075844, 3.133843, 3.379602, 2.309023, 4.248765, 1.514478, 4.490559, 2.004759, 4.920040, 3.107156, 0.422530, 0.609873],
            [2.959949, 3.407910, 4.422848, 2.193163, 4.024577, 2.811832, 4.245631, 1.954741, 0.000000, 2.404969, 0.263221, 3.957931, 2.147690, 3.927485, 0.327600, 4.274831, 4.963614, 2.025388, 1.133612, 4.596681, 1.903844, 4.509516, 2.680317, 3.958803, 0.162864, 3.024867, 1.925188, 3.793332, 3.849982, 3.130097, 1.359590, 3.372991, 4.500923, 4.942924, 4.019406, 4.643648, 2.489298, 0.347074, 0.746405, 0.434635, 4.021537, 0.153783, 2.326262, 2.223805, 2.350088, 2.284224, 0.858999, 1.634151, 4.216220, 3.387086, 0.840550, 0.547182, 0.321957, 3.402527, 2.477834, 1.769592, 3.570122, 3.346267, 1.805863, 0.162039],
            [1.841346, 0.037062, 1.649139, 1.546209, 4.984613, 3.523880, 4.462441, 0.440733, 2.404969, 0.000000, 3.725936, 1.920440, 0.412571, 1.151131, 2.881204, 1.821957, 1.722472, 0.384654, 0.822559, 2.074475, 3.288589, 0.470675, 1.845986, 0.315288, 2.650736, 2.211156, 2.388283, 1.360544, 4.008502, 0.861693, 2.320583, 0.563237, 4.015079, 2.805868, 3.730932, 3.783213, 1.176920, 2.206006, 0.336046, 4.529905, 2.024054, 4.542389, 3.236796, 3.029416, 4.825286, 3.826399, 3.963440, 4.465776, 0.460911, 3.954527, 0.519465, 3.007273, 3.632102, 1.407734, 2.077486, 0.993017, 3.832742, 3.575097, 1.162514, 2.308727],
            [0.766698, 2.823816, 1.908280, 3.857384, 3.430924, 4.022423, 4.596176, 2.575604, 0.263221, 3.725936, 0.000000, 2.881604, 1.064223, 2.759600, 4.786061, 4.818884, 3.889918, 1.363877, 3.344775, 4.593918, 1.399071, 4.228350, 4.790541, 3.880170, 1.958238, 3.001087, 4.885484, 2.611789, 2.659688, 1.987289, 1.100727, 2.835768, 4.772562, 1.933191, 0.950218, 0.603744, 2.052847, 3.101863, 3.653004, 3.789053, 4.813670, 4.543833, 0.619222, 1.180575, 1.478553, 2.360326, 4.507857, 0.976420, 2.605675, 3.904548, 2.707060, 0.180641, 0.136478, 1.925227, 3.552859, 4.366248, 3.650884, 4.022225, 2.954301, 2.007433],
            [0.298299, 2.021936, 0.772152, 4.445469, 1.573615, 1.903498, 4.261034, 1.312586, 3.957931, 1.920440, 2.881604, 0.000000, 4.055814, 1.780994, 4.574320, 3.750130, 4.899981, 2.401119, 1.617225, 1.499389, 1.662120, 3.022842, 4.172647, 2.088071, 4.910830, 4.479784, 3.406650, 3.246211, 4.391616, 0.755072, 1.681903, 3.000125, 0.310457, 3.272252, 0.352362, 3.540198, 0.036674, 3.511318, 4.451053, 0.600683, 2.689234, 2.569403, 1.295581, 4.736811, 3.678685, 3.955345, 2.795331, 4.783374, 4.108021, 3.194207, 0.881371, 1.109341, 0.405306, 4.057840, 0.682993, 1.938035, 4.930478, 3.486304, 4.139867, 3.501251],
            [4.471449, 4.592926, 2.374436, 3.790211, 3.818073, 1.823577, 1.375596, 2.729541, 2.147690, 0.412571, 1.064223, 4.055814, 0.000000, 2.758782, 4.086609, 3.752040, 3.185597, 3.894589, 1.464993, 1.259820, 1.193930, 2.526301, 3.169169, 2.814347, 3.647668, 2.461893, 1.979820, 4.898866, 2.247973, 0.400747, 1.233248, 2.811783, 2.789597, 3.273626, 1.362878, 4.858515, 3.932315, 2.521776, 4.632550, 2.037640, 2.223333, 1.514916, 4.154831, 0.236626, 3.923209, 3.296590, 4.256215, 4.199450, 3.898680, 0.522252, 0.044862, 3.772689, 1.631894, 4.697229, 1.257825, 1.224489, 2.610562, 4.048734, 3.301932, 3.632100],
            [1.911140, 1.807576, 0.772028, 0.457615, 0.845160, 0.518114, 1.878011, 1.732530, 3.927485, 1.151131, 2.759600, 1.780994, 2.758782, 0.000000, 3.897809, 4.429983, 3.263733, 2.505158, 1.473888, 3.368159, 1.105188, 3.115128, 2.645365, 0.773924, 4.898661, 1.548820, 4.514156, 0.805717, 3.076717, 4.339374, 2.131935, 0.306748, 2.247647, 4.470880, 1.255402, 2.780185, 1.223630, 3.531542, 1.057299, 2.814402, 2.256272, 0.897695, 1.902180, 3.172581, 4.360308, 3.024468, 1.201396, 4.826447, 3.357652, 1.420780, 0.247670, 3.691586, 4.327313, 2.268224, 3.210739, 3.330016, 3.506149, 1.938942, 2.395789, 0.866009],
            [4.326375, 3.666022, 3.874913, 0.662680, 1.792025, 3.464982, 1.659286, 0.908909, 0.327600, 2.881204, 4.786061, 4.574320, 4.086609, 3.897809, 0.000000, 1.872865, 0.799193, 0.049859, 1.575273, 1.245050, 4.015357, 0.942456, 4.095467, 3.869991, 0.131279, 0.192408, 0.060892, 0.386292, 2.972672, 4.582459, 0.551643, 1.690592, 1.996005, 0.807009, 0.744497, 1.630387, 0.021416, 3.145596, 3.621523, 0.089259, 4.195691, 0.837284, 2.406764, 3.394430, 4.344838, 0.546346, 3.803976, 1.572687, 4.831601, 2.591901, 3.860758, 0.749865, 2.639344, 2.460858, 4.758538, 2.425257, 4.884011, 2.986020, 4.875675, 3.385674],
            [1.102303, 2.602430, 3.622809, 3.692745, 1.276251, 3.597824, 4.031473, 0.771296, 4.274831, 1.821957, 4.818884, 3.750130, 3.752040, 4.429983, 1.872865, 0.000000, 1.745753, 1.549629, 1.879903, 4.641808, 2.169957, 2.759936, 4.544756, 2.825945, 1.866756, 0.432817, 0.780946, 0.127469, 2.839588, 1.542166, 0.908229, 1.529800, 3.649365, 0.861096, 1.202184, 1.111680, 3.743798, 0.670154, 2.662211, 3.789238, 0.131182, 1.954146, 3.343658, 4.333732, 2.086340, 3.112516, 2.691318, 0.808006, 2.621543, 0.403945, 4.654905, 0.602263, 4.906300, 4.136716, 4.429813, 0.602028, 3.365609, 2.971241, 3.789562, 3.066466],
            [0.950971, 4.810026, 4.366053, 4.092742, 4.830157, 2.138490, 3.722538, 3.152719, 4.963614, 1.722472, 3.889918, 4.899981, 3.185597, 3.263733, 0.799193, 1.745753, 0.000000, 3.856907, 0.836881, 0.121451, 3.905049, 3.328142, 2.320701, 0.798115, 2.041523, 4.525799, 2.921913, 1.046056, 3.759468, 4.988583, 0.656883, 1.509049, 0.949621, 2.952923, 4.270870, 1.798065, 2.349321, 2.524172, 2.640571, 0.043479, 0.441699, 3.504016, 1.794180, 1.979887, 3.902324, 0.269248, 4.016872, 0.298132, 3.793810, 1.943700, 2.804006, 0.926175, 0.180905, 1.369478, 4.861463, 3.480472, 2.129016, 1.951300, 4.976929, 1.818494],
            [4.002400, 0.076716, 4.097645, 0.883420, 2.602544, 2.708400, 1.078590, 2.799418, 2.025388, 0.384654, 1.363877, 2.401119, 3.894589, 2.505158, 0.049859, 1.549629, 3.856907, 0.000000, 2.175447, 4.419974, 2.225687, 4.262509, 2.910030, 0.656046, 1.280877, 4.982486, 1.123511, 1.073033, 2.878453, 2.139426, 1.443052, 3.514175, 2.638369, 2.055671, 1.681802, 4.205315, 3.866396, 3.352028, 1.862473, 1.482930, 0.840226, 3.252560, 3.419789, 0.930483, 2.586766, 4.242961, 2.527969, 2.944669, 3.441346, 2.289460, 4.650035, 0.876976, 1.122211, 1.765485, 1.842646, 2.386345, 0.973155, 0.536714, 4.669097, 3.490561],
            [0.593340, 2.189595, 4.938588, 4.264063, 1.071495, 1.626421, 0.689426, 1.396849, 1.133612, 0.822559, 3.344775, 1.617225, 1.464993, 1.473888, 1.575273, 1.879903, 0.836881, 2.175447, 0.000000, 2.337528, 4.389888, 0.429353, 3.777646, 0.455156, 3.289732, 3.416800, 0.342139, 1.382794, 1.629592, 2.816162, 1.811311, 1.887928, 1.896505, 4.949244, 4.755068, 1.337599, 4.897151, 1.792621, 4.124004, 2.162248, 2.918579, 1.088548, 2.735375, 3.649442, 0.828591, 0.762261, 2.510486, 2.391486, 3.515317, 1.993019, 3.769534, 0.097871, 1.151932, 2.462095, 0.177724, 1.039212, 3.789697, 1.656167, 4.561233, 2.887619],
            [2.671809, 0.720286, 2.270258, 0.903185, 1.255187, 0.098687, 4.554372, 4.145377, 4.596681, 2.074475, 4.593918, 1.499389, 1.259820, 3.368159, 1.245050, 4.641808, 0.121451, 4.419974, 2.337528, 0.000000, 3.658046, 4.171807, 4.808632, 3.284653, 3.972630, 2.602459, 4.810601, 0.722144, 4.386259, 1.445284, 0.656506, 3.481949, 2.380881, 2.006778, 4.406560, 3.266301, 3.607285, 0.873272, 4.910301, 1.544065, 4.291099, 0.107664, 0.910718, 0.398525, 2.880729, 2.120323, 0.485831, 4.609234, 1.767385, 4.926727, 2.220393, 4.572142, 3.864179, 2.005216, 2.276095, 4.320765, 4.799725, 4.076398, 0.217887, 4.276602],
            [1.253259, 1.464143, 2.677068, 3.547084, 2.673863, 2.345086, 4.497578, 3.457965, 1.903844, 3.288589, 1.399071, 1.662120, 1.193930, 1.105188, 4.015357, 2.169957, 3.905049, 2.225687, 4.389888, 3.658046, 0.000000, 3.964035, 3.837514, 3.297753, 3.011595, 1.585895, 2.225044, 1.968544, 0.973749, 2.842675, 3.682241, 2.375040, 1.483908, 3.852734, 2.835891, 1.851400, 3.654487, 0.654033, 1.276035, 1.950799, 0.805801, 4.470587, 3.269156, 4.891732, 3.482947, 3.867346, 2.366309, 1.239789, 3.493215, 3.083093, 1.690626, 0.940404, 1.139149, 1.700636, 1.140751, 4.601716, 0.678118, 0.787089, 3.907672, 1.125150],
            [0.936924, 1.093856, 4.253642, 2.405816, 4.536155, 2.179776, 2.858471, 3.019040, 4.509516, 0.470675, 4.228350, 3.022842, 2.526301, 3.115128, 0.942456, 2.759936, 3.328142, 4.262509, 0.429353, 4.171807, 3.964035, 0.000000, 3.398291, 3.554785, 1.548784, 3.490062, 4.536994, 3.503409, 0.615411, 2.950554, 4.566035, 3.451319, 3.504277, 1.808088, 0.418481, 2.447030, 4.798872, 4.729418, 1.388265, 4.912378, 4.151858, 0.900942, 0.311546, 0.628895, 0.824366, 4.525002, 2.371989, 2.218868, 0.529814, 0.679861, 0.837842, 4.054946, 2.749604, 2.267400, 0.517864, 1.179128, 1.825991, 2.367203, 2.483036, 0.523532],
            [4.533513, 2.902376, 1.009916, 0.220751, 3.404383, 0.596617, 4.032564, 1.090368, 2.680317, 1.845986, 4.790541, 4.172647, 3.169169, 2.645365, 4.095467, 4.544756, 2.320701, 2.910030, 3.777646, 4.808632, 3.837514, 3.398291, 0.000000, 2.482292, 0.646844, 2.452108, 1.210220, 3.679945, 2.818627, 4.920600, 3.575334, 0.927978, 1.772728, 4.500408, 4.006561, 4.276250, 4.689365, 1.204344, 3.624237, 0.238252, 1.436534, 3.561483, 2.491315, 3.642014, 4.506699, 0.740822, 4.992621, 1.711212, 4.087607, 2.498478, 1.062926, 0.931418, 0.541770, 2.585574, 4.272887, 0.041355, 3.511422, 2.666902, 0.393545, 1.740235],
            [1.929792, 3.790560, 0.555468, 0.627154, 0.254200, 4.086930, 0.447861, 1.361116, 3.958803, 0.315288, 3.880170, 2.088071, 2.814347, 0.773924, 3.869991, 2.825945, 0.798115, 0.656046, 0.455156, 3.284653, 3.297753, 3.554785, 2.482292, 0.000000, 0.610700, 0.444240, 1.851567, 0.385516, 3.775309, 0.181103, 4.568512, 0.677812, 3.016607, 3.954864, 2.931835, 4.948381, 1.203834, 2.047539, 0.296559, 4.010590, 3.773725, 3.490126, 0.555886, 2.324361, 2.250739, 3.234649, 3.846814, 2.709576, 0.796187, 1.984141, 0.752990, 1.903087, 3.880792, 4.048820, 3.831304, 3.805398, 0.487117, 1.039013, 1.493766, 4.613632],
            [4.528932, 0.009445, 1.529501, 2.180579, 0.391035, 2.061867, 2.155135, 4.010932, 0.162864, 2.650736, 1.958238, 4.910830, 3.647668, 4.898661, 0.131279, 1.866756, 2.041523, 1.280877, 3.289732, 3.972630, 3.011595, 1.548784, 0.646844, 0.610700, 0.000000, 2.450530, 2.415397, 4.105256, 3.530466, 3.641220, 2.763878, 1.998192, 2.213470, 0.598759, 4.945883, 3.790108, 1.411579, 0.452616, 2.072453, 1.309842, 4.713926, 4.712003, 4.490842, 0.723911, 4.322395, 1.528581, 2.254836, 3.338470, 1.753176, 1.129440, 3.964489, 2.781849, 0.476305, 0.897724, 4.573812, 3.630389, 3.492788, 1.905432, 0.727580, 0.144959],
            [0.394465, 4.309982, 0.193180, 4.281124, 2.220042, 2.985933, 1.614283, 2.294353, 3.024867, 2.211156, 3.001087, 4.479784, 2.461893, 1.548820, 0.192408, 0.432817, 4.525799, 4.982486, 3.416800, 2.602459, 1.585895, 3.490062, 2.452108, 0.444240, 2.450530, 0.000000, 1.587028, 1.672700, 0.943887, 1.848842, 0.536643, 1.218469, 0.868053, 1.819528, 4.580583, 0.855316, 4.934563, 1.337261, 3.166524, 4.232537, 1.904665, 0.149289, 2.249486, 0.048001, 2.415627, 1.471793, 0.042179, 3.737854, 3.120305, 3.569047, 1.150892, 0.404438, 2.102975, 4.729139, 2.794172, 2.208655, 3.256975, 0.139964, 2.062460, 1.213244],
            [3.011129, 4.038246, 4.533945, 2.945118, 4.772773, 1.202428, 2.274204, 4.451566, 1.925188, 2.388283, 4.885484, 3.406650, 1.979820, 4.514156, 0.060892, 0.780946, 2.921913, 1.123511, 0.342139, 4.810601, 2.225044, 4.536994, 1.210220, 1.851567, 2.415397, 1.587028, 0.000000, 1.761120, 0.323549, 3.076613, 3.632349, 0.792249, 2.858790, 2.449274, 1.010100, 2.087334, 1.046412, 1.482224, 4.438722, 1.564971, 2.326575, 2.809580, 0.068237, 3.004494, 0.677554, 2.428466, 1.051925, 2.953099, 3.501096, 4.205105, 4.677860, 0.135013, 3.161792, 2.448084, 4.854430, 0.081425, 0.621002, 4.136022, 1.580623, 4.692459],
            [2.278091, 0.575106, 1.624096, 3.114342, 1.164473, 2.086099, 0.084526, 2.967579, 3.793332, 1.360544, 2.611789, 3.246211, 4.898866, 0.805717, 0.386292, 0.127469, 1.046056, 1.073033, 1.382794, 0.722144, 1.968544, 3.503409, 3.679945, 0.385516, 4.105256, 1.672700, 1.761120, 0.000000, 2.004459, 2.481621, 0.593791, 1.097553, 0.166008, 2.164430, 4.361436, 4.761523, 4.005110, 3.041625, 0.864016, 0.594244, 2.545732, 2.870155, 2.850922, 0.439851, 4.203103, 0.649499, 3.806976, 4.688302, 4.879821, 4.693874, 2.448539, 3.359725, 0.652529, 1.801610, 1.868006, 3.960021, 2.616153, 0.553678, 3.397559, 1.090261],
            [3.891108, 1.368263, 1.304133, 1.850049, 0.527870, 4.856677, 3.201127, 3.606828, 3.849982, 4.008502, 2.659688, 4.391616, 2.247973, 3.076717, 2.972672, 2.839588, 3.759468, 2.878453, 1.629592, 4.386259, 0.973749, 0.615411, 2.818627, 3.775309, 3.530466, 0.943887, 0.323549, 2.004459, 0.000000, 2.241532, 2.742479, 1.538202, 3.501801, 4.725983, 3.255864, 3.362326, 3.548850, 4.073461, 1.404182, 1.580920, 3.898620, 4.189281, 0.937569, 1.334324, 1.087342, 2.240501, 2.456785, 2.665936, 4.808532, 1.655461, 4.036302, 1.264348, 4.624135, 4.842919, 2.299224, 3.178402, 1.380132, 2.599551, 4.058215, 3.142597],
            [4.247570, 0.029719, 3.812512, 2.123019, 4.390809, 3.086692, 2.569029, 4.947365, 3.130097, 0.861693, 1.987289, 0.755072, 0.400747, 4.339374, 4.582459, 1.542166, 4.988583, 2.139426, 2.816162, 1.445284, 2.842675, 2.950554, 4.920600, 0.181103, 3.641220, 1.848842, 3.076613, 2.481621, 2.241532, 0.000000, 0.927584, 2.379167, 4.882702, 1.162307, 3.386260, 4.061276, 0.860754, 1.245891, 2.047696, 0.792212, 4.738701, 4.197776, 0.269658, 4.348005, 0.699389, 4.583586, 3.374511, 3.448514, 4.836829, 4.905915, 1.107979, 0.925735, 0.181932, 3.754917, 1.810964, 3.549117, 3.075335, 3.991997, 3.372191, 0.548423],
            [1.235028, 1.056487, 3.343093, 1.894481, 4.736401, 3.826332, 2.494879, 1.572366, 1.359590, 2.320583, 1.100727, 1.681903, 1.233248, 2.131935, 0.551643, 0.908229, 0.656883, 1.443052, 1.811311, 0.656506, 3.682241, 4.566035, 3.575334, 4.568512, 2.763878, 0.536643, 3.632349, 0.593791, 2.742479, 0.927584, 0.000000, 4.636239, 0.669406, 3.003815, 0.507563, 0.970512, 0.067715, 1.694390, 4.458971, 1.831379, 2.330990, 4.829932, 4.798991, 1.318196, 0.375305, 2.188781, 4.392226, 0.959635, 0.940413, 1.961422, 4.482298, 0.667575, 4.948401, 4.808866, 4.513722, 1.611555, 0.329090, 1.808077, 0.615058, 4.770665],
            [1.940352, 2.517721, 2.405410, 4.804733, 0.420039, 2.543594, 1.303854, 4.577603, 3.372991, 0.563237, 2.835768, 3.000125, 2.811783, 0.306748, 1.690592, 1.529800, 1.509049, 3.514175, 1.887928, 3.481949, 2.375040, 3.451319, 0.927978, 0.677812, 1.998192, 1.218469, 0.792249, 1.097553, 1.538202, 2.379167, 4.636239, 0.000000, 4.313072, 1.784115, 4.167587, 2.392493, 0.569093, 2.255844, 0.056358, 3.921198, 2.856799, 1.602515, 4.515796, 4.987449, 2.548371, 3.919352, 4.537016, 4.736776, 0.159031, 4.439007, 3.723979, 1.023324, 4.391564, 0.866400, 0.395608, 2.965174, 1.327390, 4.842653, 3.447220, 2.724690],
            [2.623409, 0.309230, 2.362015, 2.076674, 4.896554, 3.763313, 1.561656, 2.330103, 4.500923, 4.015079, 4.772562, 0.310457, 2.789597, 2.247647, 1.996005, 3.649365, 0.949621, 2.638369, 1.896505, 2.380881, 1.483908, 3.504277, 1.772728, 3.016607, 2.213470, 0.868053, 2.858790, 0.166008, 3.501801, 4.882702, 0.669406, 4.313072, 0.000000, 1.193224, 0.493570, 3.896718, 0.964922, 4.525006, 1.938786, 1.929800, 4.442321, 2.193095, 1.800157, 2.904924, 4.708134, 3.707067, 0.160669, 2.557743, 0.806934, 3.105019, 0.847165, 2.685986, 3.690149, 3.309067, 0.509561, 1.488083, 0.243266, 1.442032, 0.532808, 0.958075],
            [0.906379, 4.477829, 2.328213, 0.260492, 3.907651, 1.770965, 4.692629, 1.408378, 4.942924, 2.805868, 1.933191, 3.272252, 3.273626, 4.470880, 0.807009, 0.861096, 2.952923, 2.055671, 4.949244, 2.006778, 3.852734, 1.808088, 4.500408, 3.954864, 0.598759, 1.819528, 2.449274, 2.164430, 4.725983, 1.162307, 3.003815, 1.784115, 1.193224, 0.000000, 2.944518, 4.122092, 3.861706, 3.476445, 3.285803, 0.457095, 3.893158, 3.030192, 3.011258, 3.971534, 4.142507, 1.734152, 3.519228, 3.058650, 1.339064, 4.691506, 3.710544, 4.835460, 0.076133, 4.967032, 2.715686, 2.942235, 1.318977, 4.666276, 1.210098, 1.242323],
            [3.268118, 3.950387, 1.082086, 2.040455, 0.270675, 1.674429, 1.577760, 2.556421, 4.019406, 3.730932, 0.950218, 0.352362, 1.362878, 1.255402, 0.744497, 1.202184, 4.270870, 1.681802, 4.755068, 4.406560, 2.835891, 0.418481, 4.006561, 2.931835, 4.945883, 4.580583, 1.010100, 4.361436, 3.255864, 3.386260, 0.507563, 4.167587, 0.493570, 2.944518, 0.000000, 4.763077, 1.550875, 4.982127, 2.109926, 3.202963, 4.211743, 0.740155, 2.373843, 2.823424, 4.934635, 0.323210, 2.771419, 2.914745, 2.512732, 3.151984, 1.116564, 0.442435, 1.623340, 0.124277, 2.370001, 3.421593, 1.741587, 0.959568, 3.262073, 1.781041],
            [4.162085, 4.244929, 1.763505, 2.849782, 0.725090, 1.620840, 2.067671, 4.571382, 4.643648, 3.783213, 0.603744, 3.540198, 4.858515, 2.780185, 1.630387, 1.111680, 1.798065, 4.205315, 1.337599, 3.266301, 1.851400, 2.447030, 4.276250, 4.948381, 3.790108, 0.855316, 2.087334, 4.761523, 3.362326, 4.061276, 0.970512, 2.392493, 3.896718, 4.122092, 4.763077, 0.000000, 0.225436, 4.800311, 4.375172, 4.154831, 4.012894, 4.081092, 0.233928, 0.865104, 2.761054, 1.119373, 1.758027, 4.837541, 0.171586, 1.039191, 4.754945, 2.600553, 0.137943, 3.948027, 3.723771, 4.439128, 4.075015, 4.691283, 4.385047, 4.032523],
            [4.732443, 0.392727, 4.985049, 3.045587, 1.939082, 0.588086, 0.655899, 3.375118, 2.489298, 1.176920, 2.052847, 0.036674, 3.932315, 1.223630, 0.021416, 3.743798, 2.349321, 3.866396, 4.897151, 3.607285, 3.654487, 4.798872, 4.689365, 1.203834, 1.411579, 4.934563, 1.046412, 4.005110, 3.548850, 0.860754, 0.067715, 0.569093, 0.964922, 3.861706, 1.550875, 0.225436, 0.000000, 1.144891, 2.159994, 4.558883, 1.890012, 2.560964, 0.933540, 4.639736, 4.766880, 0.771071, 0.401261, 3.208022, 2.514754, 0.825443, 4.133039, 2.208417, 1.124130, 1.946316, 3.456136, 3.254598, 4.040188, 0.649267, 4.128773, 3.604754],
            [4.382857, 0.457889, 0.849519, 4.962545, 4.813466, 1.245331, 2.507014, 0.240776, 0.347074, 2.206006, 3.101863, 3.511318, 2.521776, 3.531542, 3.145596, 0.670154, 2.524172, 3.352028, 1.792621, 0.873272, 0.654033, 4.729418, 1.204344, 2.047539, 0.452616, 1.337261, 1.482224, 3.041625, 4.073461, 1.245891, 1.694390, 2.255844, 4.525006, 3.476445, 4.982127, 4.800311, 1.144891, 0.000000, 4.023439, 3.704265, 0.872956, 2.165327, 1.746551, 0.309754, 0.261021, 0.855719, 4.752249, 0.465320, 2.367903, 4.828939, 3.050040, 1.769798, 2.633894, 3.418368, 0.033088, 4.539869, 1.358781, 0.308184, 4.553165, 0.118994],
            [2.071456, 0.009377, 0.673563, 2.297129, 4.581269, 3.992857, 2.583064, 4.831864, 0.746405, 0.336046, 3.653004, 4.451053, 4.632550, 1.057299, 3.621523, 2.662211, 2.640571, 1.862473, 4.124004, 4.910301, 1.276035, 1.388265, 3.624237, 0.296559, 2.072453, 3.166524, 4.438722, 0.864016, 1.404182, 2.047696, 4.458971, 0.056358, 1.938786, 3.285803, 2.109926, 4.375172, 2.159994, 4.023439, 0.000000, 1.346753, 2.741684, 1.768626, 3.168829, 4.759260, 3.904098, 1.822006, 0.557117, 4.820428, 2.366127, 3.140177, 0.211633, 4.883731, 4.705922, 2.955644, 4.511437, 0.806948, 4.521171, 1.173282, 4.599974, 3.266292],
            [4.663953, 0.304110, 2.952564, 2.541173, 1.417995, 3.412501, 3.697204, 3.338249, 0.434635, 4.529905, 3.789053, 0.600683, 2.037640, 2.814402, 0.089259, 3.789238, 0.043479, 1.482930, 2.162248, 1.544065, 1.950799, 4.912378, 0.238252, 4.010590, 1.309842, 4.232537, 1.564971, 0.594244, 1.580920, 0.792212, 1.831379, 3.921198, 1.929800, 0.457095, 3.202963, 4.154831, 4.558883, 3.704265, 1.346753, 0.000000, 0.054698, 3.170779, 2.918579, 2.075068, 4.457390, 2.656872, 3.589651, 4.533786, 3.468598, 4.264010, 3.650409, 1.757155, 2.989545, 4.070868, 2.265062, 4.839337, 4.576493, 1.255771, 1.897206, 3.062662],
            [1.353513, 0.551220, 4.706393, 0.140594, 2.360987, 2.836147, 0.152935, 1.025402, 4.021537, 2.024054, 4.813670, 2.689234, 2.223333, 2.256272, 4.195691, 0.131182, 0.441699, 0.840226, 2.918579, 4.291099, 0.805801, 4.151858, 1.436534, 3.773725, 4.713926, 1.904665, 2.326575, 2.545732, 3.898620, 4.738701, 2.330990, 2.856799, 4.442321, 3.893158, 4.211743, 4.012894, 1.890012, 0.872956, 2.741684, 0.054698, 0.000000, 1.366829, 1.567826, 2.438195, 3.475709, 0.832502, 4.655581, 2.823529, 3.710343, 1.397707, 0.377569, 1.081160, 1.651888, 0.598018, 2.954570, 0.793012, 2.221705, 4.396192, 4.473021, 1.429883],
            [3.393886, 3.885758, 0.359912, 2.014026, 1.776643, 1.113323, 1.243407, 2.915041, 0.153783, 4.542389, 4.543833, 2.569403, 1.514916, 0.897695, 0.837284, 1.954146, 3.504016, 3.252560, 1.088548, 0.107664, 4.470587, 0.900942, 3.561483, 3.490126, 4.712003, 0.149289, 2.809580, 2.870155, 4.189281, 4.197776, 4.829932, 1.602515, 2.193095, 3.030192, 0.740155, 4.081092, 2.560964, 2.165327, 1.768626, 3.170779, 1.366829, 0.000000, 1.540124, 0.961393, 3.052081, 1.047302, 3.217874, 1.301536, 1.452735, 0.765428, 2.276259, 1.473696, 1.781762, 1.054873, 3.735404, 4.622936, 2.832182, 3.650357, 2.576310, 4.811675],
            [4.247184, 0.271506, 3.739034, 0.162418, 2.660236, 2.852918, 3.326437, 0.049472, 2.326262, 3.236796, 0.619222, 1.295581, 4.154831, 1.902180, 2.406764, 3.343658, 1.794180, 3.419789, 2.735375, 0.910718, 3.269156, 0.311546, 2.491315, 0.555886, 4.490842, 2.249486, 0.068237, 2.850922, 0.937569, 0.269658, 4.798991, 4.515796, 1.800157, 3.011258, 2.373843, 0.233928, 0.933540, 1.746551, 3.168829, 2.918579, 1.567826, 1.540124, 0.000000, 1.866182, 2.967401, 0.045763, 2.527248, 2.416823, 3.578698, 0.363674, 0.323229, 3.838486, 1.969122, 3.022354, 0.165025, 3.238283, 1.732885, 0.192397, 4.866311, 3.201344],
            [1.794547, 0.130646, 4.428220, 0.102457, 3.426088, 1.124315, 4.272031, 0.203018, 2.223805, 3.029416, 1.180575, 4.736811, 0.236626, 3.172581, 3.394430, 4.333732, 1.979887, 0.930483, 3.649442, 0.398525, 4.891732, 0.628895, 3.642014, 2.324361, 0.723911, 0.048001, 3.004494, 0.439851, 1.334324, 4.348005, 1.318196, 4.987449, 2.904924, 3.971534, 2.823424, 0.865104, 4.639736, 0.309754, 4.759260, 2.075068, 2.438195, 0.961393, 1.866182, 0.000000, 1.932425, 4.573109, 3.155915, 4.084310, 1.345891, 4.547584, 1.285495, 0.812160, 2.932580, 2.439979, 2.234995, 3.850326, 2.774368, 1.477338, 2.008026, 2.428944],
            [4.833405, 4.772263, 1.703105, 2.037226, 3.493518, 2.939961, 0.989585, 3.399892, 2.350088, 4.825286, 1.478553, 3.678685, 3.923209, 4.360308, 4.344838, 2.086340, 3.902324, 2.586766, 0.828591, 2.880729, 3.482947, 0.824366, 4.506699, 2.250739, 4.322395, 2.415627, 0.677554, 4.203103, 1.087342, 0.699389, 0.375305, 2.548371, 4.708134, 4.142507, 4.934635, 2.761054, 4.766880, 0.261021, 3.904098, 4.457390, 3.475709, 3.052081, 2.967401, 1.932425, 0.000000, 2.915442, 2.514319, 1.501085, 4.190244, 0.087511, 0.222872, 1.045609, 1.485673, 4.667311, 3.104965, 3.399484, 4.153070, 2.567277, 2.742404, 0.810146],
            [0.135464, 2.146542, 0.192710, 4.626958, 4.314176, 1.525044, 0.091231, 0.048765, 2.284224, 3.826399, 2.360326, 3.955345, 3.296590, 3.024468, 0.546346, 3.112516, 0.269248, 4.242961, 0.762261, 2.120323, 3.867346, 4.525002, 0.740822, 3.234649, 1.528581, 1.471793, 2.428466, 0.649499, 2.240501, 4.583586, 2.188781, 3.919352, 3.707067, 1.734152, 0.323210, 1.119373, 0.771071, 0.855719, 1.822006, 2.656872, 0.832502, 1.047302, 0.045763, 4.573109, 2.915442, 0.000000, 0.989777, 0.982425, 3.458862, 4.057322, 3.368848, 4.746406, 2.911408, 0.867436, 2.227476, 1.068688, 1.488340, 4.245121, 0.878970, 1.859134],
            [0.376182, 0.681796, 1.896135, 3.888976, 3.211678, 1.572943, 3.922196, 4.420342, 0.858999, 3.963440, 4.507857, 2.795331, 4.256215, 1.201396, 3.803976, 2.691318, 4.016872, 2.527969, 2.510486, 0.485831, 2.366309, 2.371989, 4.992621, 3.846814, 2.254836, 0.042179, 1.051925, 3.806976, 2.456785, 3.374511, 4.392226, 4.537016, 0.160669, 3.519228, 2.771419, 1.758027, 0.401261, 4.752249, 0.557117, 3.589651, 4.655581, 3.217874, 2.527248, 3.155915, 2.514319, 0.989777, 0.000000, 2.384109, 3.987105, 3.089226, 4.555358, 0.732718, 2.008816, 2.314136, 0.514982, 1.319526, 2.003742, 4.660763, 1.491077, 1.322731],
            [4.576669, 1.763619, 0.312967, 2.280618, 1.465118, 1.788098, 4.582176, 3.977592, 1.634151, 4.465776, 0.976420, 4.783374, 4.199450, 4.826447, 1.572687, 0.808006, 0.298132, 2.944669, 2.391486, 4.609234, 1.239789, 2.218868, 1.711212, 2.709576, 3.338470, 3.737854, 2.953099, 4.688302, 2.665936, 3.448514, 0.959635, 4.736776, 2.557743, 3.058650, 2.914745, 4.837541, 3.208022, 0.465320, 4.820428, 4.533786, 2.823529, 1.301536, 2.416823, 4.084310, 1.501085, 0.982425, 2.384109, 0.000000, 4.700623, 1.592994, 0.242643, 4.352643, 0.968926, 3.130114, 0.081884, 0.194814, 3.290392, 3.097050, 2.850637, 2.318373],
            [0.080695, 3.035987, 4.528480, 0.381775, 3.146517, 2.747591, 4.181724, 4.075844, 4.216220, 0.460911, 2.605675, 4.108021, 3.898680, 3.357652, 4.831601, 2.621543, 3.793810, 3.441346, 3.515317, 1.767385, 3.493215, 0.529814, 4.087607, 0.796187, 1.753176, 3.120305, 3.501096, 4.879821, 4.808532, 4.836829, 0.940413, 0.159031, 0.806934, 1.339064, 2.512732, 0.171586, 2.514754, 2.367903, 2.366127, 3.468598, 3.710343, 1.452735, 3.578698, 1.345891, 4.190244, 3.458862, 3.987105, 4.700623, 0.000000, 3.570214, 4.936664, 2.117743, 4.833628, 3.232347, 4.608893, 4.749386, 4.778185, 4.045783, 2.106479, 3.343167],
            [0.276691, 3.649296, 2.021798, 1.042477, 4.399113, 0.653038, 4.290174, 3.133843, 3.387086, 3.954527, 3.904548, 3.194207, 0.522252, 1.420780, 2.591901, 0.403945, 1.943700, 2.289460, 1.993019, 4.926727, 3.083093, 0.679861, 2.498478, 1.984141, 1.129440, 3.569047, 4.205105, 4.693874, 1.655461, 4.905915, 1.961422, 4.439007, 3.105019, 4.691506, 3.151984, 1.039191, 0.825443, 4.828939, 3.140177, 4.264010, 1.397707, 0.765428, 0.363674, 4.547584, 0.087511, 4.057322, 3.089226, 1.592994, 3.570214, 0.000000, 0.648903, 3.352438, 4.614618, 0.696003, 0.377052, 2.685521, 3.055711, 4.827546, 0.039899, 3.472236],
            [1.296933, 4.064706, 2.888916, 0.633150, 4.269318, 4.169466, 3.154676, 3.379602, 0.840550, 0.519465, 2.707060, 0.881371, 0.044862, 0.247670, 3.860758, 4.654905, 2.804006, 4.650035, 3.769534, 2.220393, 1.690626, 0.837842, 1.062926, 0.752990, 3.964489, 1.150892, 4.677860, 2.448539, 4.036302, 1.107979, 4.482298, 3.723979, 0.847165, 3.710544, 1.116564, 4.754945, 4.133039, 3.050040, 0.211633, 3.650409, 0.377569, 2.276259, 0.323229, 1.285495, 0.222872, 3.368848, 4.555358, 0.242643, 4.936664, 0.648903, 0.000000, 1.335315, 1.285969, 1.054475, 0.778420, 2.240364, 4.002956, 1.232909, 2.067272, 3.394922],
            [0.935806, 4.289919, 4.798650, 1.152656, 4.126538, 0.085314, 0.938573, 2.309023, 0.547182, 3.007273, 0.180641, 1.109341, 3.772689, 3.691586, 0.749865, 0.602263, 0.926175, 0.876976, 0.097871, 4.572142, 0.940404, 4.054946, 0.931418, 1.903087, 2.781849, 0.404438, 0.135013, 3.359725, 1.264348, 0.925735, 0.667575, 1.023324, 2.685986, 4.835460, 0.442435, 2.600553, 2.208417, 1.769798, 4.883731, 1.757155, 1.081160, 1.473696, 3.838486, 0.812160, 1.045609, 4.746406, 0.732718, 4.352643, 2.117743, 3.352438, 1.335315, 0.000000, 1.498164, 2.822097, 1.913044, 3.351013, 0.163420, 2.493127, 0.269349, 4.491096],
            [3.551193, 2.411282, 0.898716, 2.302290, 4.666094, 3.761569, 2.758261, 4.248765, 0.321957, 3.632102, 0.136478, 0.405306, 1.631894, 4.327313, 2.639344, 4.906300, 0.180905, 1.122211, 1.151932, 3.864179, 1.139149, 2.749604, 0.541770, 3.880792, 0.476305, 2.102975, 3.161792, 0.652529, 4.624135, 0.181932, 4.948401, 4.391564, 3.690149, 0.076133, 1.623340, 0.137943, 1.124130, 2.633894, 4.705922, 2.989545, 1.651888, 1.781762, 1.969122, 2.932580, 1.485673, 2.911408, 2.008816, 0.968926, 4.833628, 4.614618, 1.285969, 1.498164, 0.000000, 2.719333, 2.544889, 2.519545, 0.638635, 3.162921, 4.575447, 1.884916],
            [0.716795, 0.804869, 2.530143, 3.700051, 3.455189, 1.073782, 0.267571, 1.514478, 3.402527, 1.407734, 1.925227, 4.057840, 4.697229, 2.268224, 2.460858, 4.136716, 1.369478, 1.765485, 2.462095, 2.005216, 1.700636, 2.267400, 2.585574, 4.048820, 0.897724, 4.729139, 2.448084, 1.801610, 4.842919, 3.754917, 4.808866, 0.866400, 3.309067, 4.967032, 0.124277, 3.948027, 1.946316, 3.418368, 2.955644, 4.070868, 0.598018, 1.054873, 3.022354, 2.439979, 4.667311, 0.867436, 2.314136, 3.130114, 3.232347, 0.696003, 1.054475, 2.822097, 2.719333, 0.000000, 4.283796, 1.690145, 4.147512, 0.005050, 4.643430, 3.005559],
            [1.801765, 2.268771, 0.294771, 3.720028, 3.872908, 0.045199, 2.316687, 4.490559, 2.477834, 2.077486, 3.552859, 0.682993, 1.257825, 3.210739, 4.758538, 4.429813, 4.861463, 1.842646, 0.177724, 2.276095, 1.140751, 0.517864, 4.272887, 3.831304, 4.573812, 2.794172, 4.854430, 1.868006, 2.299224, 1.810964, 4.513722, 0.395608, 0.509561, 2.715686, 2.370001, 3.723771, 3.456136, 0.033088, 4.511437, 2.265062, 2.954570, 3.735404, 0.165025, 2.234995, 3.104965, 2.227476, 0.514982, 0.081884, 4.608893, 0.377052, 0.778420, 1.913044, 2.544889, 4.283796, 0.000000, 2.449759, 1.826208, 2.020085, 0.672939, 0.947123],
            [3.505492, 2.311098, 1.248161, 3.157625, 1.767299, 3.312412, 4.405737, 2.004759, 1.769592, 0.993017, 4.366248, 1.938035, 1.224489, 3.330016, 2.425257, 0.602028, 3.480472, 2.386345, 1.039212, 4.320765, 4.601716, 1.179128, 0.041355, 3.805398, 3.630389, 2.208655, 0.081425, 3.960021, 3.178402, 3.549117, 1.611555, 2.965174, 1.488083, 2.942235, 3.421593, 4.439128, 3.254598, 4.539869, 0.806948, 4.839337, 0.793012, 4.622936, 3.238283, 3.850326, 3.399484, 1.068688, 1.319526, 0.194814, 4.749386, 2.685521, 2.240364, 3.351013, 2.519545, 1.690145, 2.449759, 0.000000, 2.690276, 0.470755, 1.532737, 4.369441],
            [4.780772, 2.949519, 3.321985, 0.321973, 0.331327, 4.764350, 1.749346, 4.920040, 3.570122, 3.832742, 3.650884, 4.930478, 2.610562, 3.506149, 4.884011, 3.365609, 2.129016, 0.973155, 3.789697, 4.799725, 0.678118, 1.825991, 3.511422, 0.487117, 3.492788, 3.256975, 0.621002, 2.616153, 1.380132, 3.075335, 0.329090, 1.327390, 0.243266, 1.318977, 1.741587, 4.075015, 4.040188, 1.358781, 4.521171, 4.576493, 2.221705, 2.832182, 1.732885, 2.774368, 4.153070, 1.488340, 2.003742, 3.290392, 4.778185, 3.055711, 4.002956, 0.163420, 0.638635, 4.147512, 1.826208, 2.690276, 0.000000, 3.338047, 0.052525, 4.896631],
            [1.909776, 4.670592, 2.678289, 0.627631, 4.983391, 4.360613, 4.632079, 3.107156, 3.346267, 3.575097, 4.022225, 3.486304, 4.048734, 1.938942, 2.986020, 2.971241, 1.951300, 0.536714, 1.656167, 4.076398, 0.787089, 2.367203, 2.666902, 1.039013, 1.905432, 0.139964, 4.136022, 0.553678, 2.599551, 3.991997, 1.808077, 4.842653, 1.442032, 4.666276, 0.959568, 4.691283, 0.649267, 0.308184, 1.173282, 1.255771, 4.396192, 3.650357, 0.192397, 1.477338, 2.567277, 4.245121, 4.660763, 3.097050, 4.045783, 4.827546, 1.232909, 2.493127, 3.162921, 0.005050, 2.020085, 0.470755, 3.338047, 0.000000, 0.121494, 4.661399],
            [3.279791, 1.918577, 3.800891, 1.208220, 1.711828, 4.274578, 4.591950, 0.422530, 1.805863, 1.162514, 2.954301, 4.139867, 3.301932, 2.395789, 4.875675, 3.789562, 4.976929, 4.669097, 4.561233, 0.217887, 3.907672, 2.483036, 0.393545, 1.493766, 0.727580, 2.062460, 1.580623, 3.397559, 4.058215, 3.372191, 0.615058, 3.447220, 0.532808, 1.210098, 3.262073, 4.385047, 4.128773, 4.553165, 4.599974, 1.897206, 4.473021, 2.576310, 4.866311, 2.008026, 2.742404, 0.878970, 1.491077, 2.850637, 2.106479, 0.039899, 2.067272, 0.269349, 4.575447, 4.643430, 0.672939, 1.532737, 0.052525, 0.121494, 0.000000, 3.443382],
            [0.578301, 1.866127, 1.879479, 4.698178, 0.754905, 3.766793, 3.679949, 0.609873, 0.162039, 2.308727, 2.007433, 3.501251, 3.632100, 0.866009, 3.385674, 3.066466, 1.818494, 3.490561, 2.887619, 4.276602, 1.125150, 0.523532, 1.740235, 4.613632, 0.144959, 1.213244, 4.692459, 1.090261, 3.142597, 0.548423, 4.770665, 2.724690, 0.958075, 1.242323, 1.781041, 4.032523, 3.604754, 0.118994, 3.266292, 3.062662, 1.429883, 4.811675, 3.201344, 2.428944, 0.810146, 1.859134, 1.322731, 2.318373, 3.343167, 3.472236, 3.394922, 4.491096, 1.884916, 3.005559, 0.947123, 4.369441, 4.896631, 4.661399, 3.443382, 0.000000],
        ]);

        let ord = vec![
            0, 1, 19, 46, 7, 8, 52, 11, 9, 38, 60, 25, 34, 53, 12, 37, 2, 30, 43, 32,
            39, 10, 24, 14, 4, 5, 16, 28, 15, 18, 40, 41, 23, 56, 27, 29, 21, 58, 54, 35,
            22, 13, 51, 45, 50, 55, 48, 3, 31, 57, 59, 33, 47, 49, 36, 44, 26, 42, 6, 20,
            17,
        ];
        let mut params = NNLSParams::default();
        let progress = None; // No progress tracking in this test

        let (_unused, pairs) = compute_use_1d(&ord, &d, &mut params, progress).expect("NNLS solve");

        let weights = pairs.iter().map(|(_, w)| *w).collect::<Vec<f64>>();

        let y_exp = vec![
            0.000000000000000,
            0.481924405907773,
            0.030184573810940,
            0.281612666140457,
            0.296565198097998,
            0.000000000000000,
            0.176038124722451,
            0.060505583475459,
            0.421332053762197,
            0.000000000000000,
            0.244870390322182,
            0.008280506237954,
            0.559187281508848,
            0.027130758279508,
            0.181660629814013,
            0.296525748592014,
            0.105609104374512,
            0.000000000000000,
            0.121897256304520,
            0.468323046398055,
            0.046597268417497,
            0.476993040518840,
            0.202666014188152,
            0.202598690178842,
            0.002929631201874,
            0.000000000000000,
            0.454592835583253,
            0.000000000000000,
            0.138406354901053,
            0.005363889100627,
            0.000000000000000,
            0.224495139394220,
            0.024063884727485,
            0.000000000000000,
            0.188875785299142,
            0.493229545008547,
            0.257164045208390,
            0.000000000000000,
            0.392838961696301,
            0.176190100581660,
            0.367796595186249,
            0.191954451445900,
            0.034682185475202,
            0.000000000000000,
            0.161720439425202,
            0.063009206977795,
            0.000000000000000,
            0.200296195288211,
            0.021552118146121,
            0.374985748587505,
            0.000000000000000,
            0.000000000000000,
            0.356265145389597,
            0.243956976067910,
            0.271262997595177,
            0.019273155389703,
            0.042958205586782,
            0.021209826788552,
            0.000000000000000,
            0.136247572482737,
            0.084974454329419,
            0.000000000000000,
            0.000000000000000,
            0.351310473942671,
            0.000000000000000,
            0.229227569435172,
            0.000000000000000,
            0.157112295675265,
            0.095723790701113,
            0.078393292040265,
            0.000000000000000,
            0.087838890475328,
            0.215411158281699,
            0.159132152786242,
            0.523967288131605,
            0.000000000000000,
            0.379544516308312,
            0.000000000000000,
            0.207601689850593,
            0.203558159126705,
            0.000000000000000,
            0.098752600327642,
            0.026325280011368,
            0.000000000000000,
            0.472090517554450,
            0.000000000000000,
            0.402051577713342,
            0.000000000000000,
            0.334508130465384,
            0.154993830604739,
            0.080408792012590,
            0.000000000000000,
            0.237071580255705,
            0.040766437414413,
            0.000000000000000,
            0.341457220984455,
            0.000000000000000,
            0.610073437245372,
            0.182568604094364,
            0.010324814241373,
            0.000000000000000,
            0.323405477224583,
            0.000000000000000,
            0.466631841878099,
            0.260053600359002,
            0.000000000000000,
            0.165634759346461,
            0.000000000000000,
            0.240384479490172,
            0.110775653020913,
            0.170295367696976,
            0.328159322381664,
            0.706103570952814,
            0.000000000000000,
            0.203256748950749,
            0.175889341692030,
            0.000000000000000,
            0.625725930164290,
            0.000000000000000,
            0.071559990510537,
            0.000000000000000,
            0.417445754838486,
            0.000000000000000,
            0.408849445221295,
            0.083749491750887,
            0.000000000000000,
            0.149967941963181,
            0.329211799248652,
            0.328596740091549,
            0.067602728908789,
            0.000000000000000,
            0.486656684361125,
            0.000000000000000,
            0.004883540895702,
            0.074325018073390,
            0.000000000000000,
            0.013162926619161,
            0.412882789598215,
            0.188125773073844,
            0.240625413432562,
            0.791863946409626,
            0.338831207788113,
            0.000000000000000,
            0.212873483422917,
            0.000000000000000,
            0.230000419448313,
            0.000000000000000,
            0.117693172314804,
            0.000000000000000,
            0.445167242733498,
            0.066321195694377,
            0.000000000000000,
        ];

        // Compare only non-zero weights to avoid mismatches from dropped zero/negative splits.
        let nz_eps = 1e-12;
        let weights_nz = weights
            .iter()
            .copied()
            .filter(|w| *w > nz_eps)
            .collect::<Vec<f64>>();
        let y_exp_nz = y_exp
            .iter()
            .copied()
            .filter(|w| *w > nz_eps)
            .collect::<Vec<f64>>();

        let sum_weights = weights_nz.iter().sum::<f64>();
        let sum_exp = y_exp_nz.iter().sum::<f64>();
        let max_weights = weights_nz
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_exp = y_exp_nz
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert_eq!(weights_nz.len(), 106);
        assert_eq!(y_exp_nz.len(), 106);
        assert!(
            (sum_weights - sum_exp).abs() < 1e-2,
            "sum_weights={} sum_exp={}",
            sum_weights,
            sum_exp
        );
        assert!(
            (max_weights - max_exp).abs() < 1e-2,
            "max_weights={} max_exp={}",
            max_weights,
            max_exp
        );
    }


    fn compare_float_array(arr1: &[f64], arr2: &[f64], eps: f64) {
        assert_eq!(arr1.len(), arr2.len());
        for (a, b) in arr1.iter().zip(arr2.iter()) {
            assert!((*a - *b).abs() < eps, "got {}, wanted {}", a, b);
        }
    }
}

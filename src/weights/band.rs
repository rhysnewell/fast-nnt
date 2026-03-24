use std::time::{Duration, Instant};

/// Band-major (anti-diagonal) index for packed upper triangle.
/// Band k (1-based width) contains elements (i, i+k) for i=0..n-k.
/// Elements within each band are stored contiguously.
#[derive(Debug)]
pub struct BandIndex {
    pub n: usize,
    pub npairs: usize,
    /// band_offsets[k] = start of band k+1 in the flat array (0-indexed: band_offsets[0] = start of band 1)
    pub band_offsets: Vec<usize>,
    /// Precomputed: for each row-major index, the corresponding band-major index.
    pub row_to_band_map: Vec<usize>,
    /// Precomputed inverse: for each band-major index, the corresponding row-major index.
    pub band_to_row_map: Vec<usize>,
}

impl BandIndex {
    pub fn new(n: usize) -> Self {
        let npairs = n * (n - 1) / 2;
        let mut band_offsets = vec![0usize; n];
        let mut offset = 0;
        for k in 1..n {
            band_offsets[k - 1] = offset;
            offset += n - k;
        }

        // Precompute the row-to-band mapping
        let mut row_to_band_map = vec![0usize; npairs];
        let mut row_idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let k = j - i;
                row_to_band_map[row_idx] = band_offsets[k - 1] + i;
                row_idx += 1;
            }
        }

        // Precompute the band-to-row mapping (inverse)
        let mut band_to_row_map = vec![0usize; npairs];
        for (row_idx, &band_idx) in row_to_band_map.iter().enumerate() {
            band_to_row_map[band_idx] = row_idx;
        }

        Self {
            n,
            npairs,
            band_offsets,
            row_to_band_map,
            band_to_row_map,
        }
    }

    /// Get the slice range for band k (1-based).
    #[inline]
    pub fn band_range(&self, k: usize) -> (usize, usize) {
        let start = self.band_offsets[k - 1];
        let len = self.n - k;
        (start, start + len)
    }
}

/// Convert from row-major packed upper triangle to band-major layout.
#[inline]
pub fn row_to_band(src: &[f64], dst: &mut [f64], band: &BandIndex) {
    debug_assert_eq!(src.len(), band.npairs);
    debug_assert_eq!(dst.len(), band.npairs);
    for (row_idx, &band_idx) in band.row_to_band_map.iter().enumerate() {
        dst[band_idx] = src[row_idx];
    }
}

/// Convert from band-major layout to row-major packed upper triangle.
#[inline]
pub fn band_to_row(src: &[f64], dst: &mut [f64], band: &BandIndex) {
    debug_assert_eq!(src.len(), band.npairs);
    debug_assert_eq!(dst.len(), band.npairs);
    for (row_idx, &band_idx) in band.row_to_band_map.iter().enumerate() {
        dst[row_idx] = src[band_idx];
    }
}

/// Compute row sums from band-major data in the same accumulation order as
/// row-major batch_rowsums, to ensure FP-identical results.
pub fn batch_rowsums_band(b: &[f64], sums: &mut [f64], band: &BandIndex) {
    let n = band.n;
    sums[..n].fill(0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let k = j - i;
            let val = b[band.band_offsets[k - 1] + i];
            sums[i] += val;
            sums[j] += val;
        }
    }
}

/// Compute row sums from band-major data using band-sequential traversal.
/// Iterates bands k=1..n-1 in order, reading each band contiguously (stride-1).
/// Much faster than batch_rowsums_band for large n because it avoids stride-n
/// random access through the 42MB array.
/// Accumulation order differs (band-order vs row-order), so FP results may
/// differ by ~n*eps*max_val. This is well below CG tolerance.
pub fn batch_rowsums_band_sequential(b: &[f64], sums: &mut [f64], band: &BandIndex) {
    let n = band.n;
    sums[..n].fill(0.0);
    for k in 1..n {
        let (start, end) = band.band_range(k);
        let band_slice = &b[start..end];
        for (i, &val) in band_slice.iter().enumerate() {
            // SAFETY: band k has (n-k) elements so i < n-k; thus i < n and i+k < n,
            // both within the sums[..n] range.
            unsafe {
                *sums.get_unchecked_mut(i) += val;
                *sums.get_unchecked_mut(i + k) += val;
            }
        }
    }
}

/// Compute row sums from band-major data in the same accumulation order as
/// calc_ax_indexed Pass 1: for each vertex v, row entries first (v as first index),
/// then column entries (v as second index). This matches the active_set forward operator.
pub fn batch_rowsums_band_forward(b: &[f64], sums: &mut [f64], band: &BandIndex) {
    let n = band.n;
    sums[..n].fill(0.0);
    for v in 0..n {
        // Row entries first: b[pair(v, j)] for j > v
        for j in (v + 1)..n {
            let k = j - v;
            sums[v] += b[band.band_offsets[k - 1] + v];
        }
        // Column entries: b[pair(i, v)] for i < v
        for i in 0..v {
            let k = v - i;
            sums[v] += b[band.band_offsets[k - 1] + i];
        }
    }
}

// NOTE: The four kernel functions below (calculate_forward_band, calculate_ab_band,
// and their _with_norm_sq variants) share the same recurrence but are intentionally
// kept as separate functions. The _with_norm variants fuse norm-squared accumulation
// into the band iteration to avoid a second pass over the output. Unifying them via
// closures/generics risks changing FP accumulation order (the code is tuned for
// FP-identical results across kernel variants) and cache behavior in these hot paths.

/// Band-major kernel for the active_set forward operator (A*x).
/// Mathematically equivalent to calc_ax_indexed but operates on band-major data.
/// Uses forward accumulation order for row sums to ensure FP-identical results.
pub fn calculate_forward_band(b: &[f64], d: &mut [f64], band: &BandIndex, row_sums: &mut [f64]) {
    let n = band.n;

    // Pass 1: row sums in forward order (row-first-then-column for each vertex)
    batch_rowsums_band_forward(b, row_sums, band);

    // Band 1: d(i, i+1) = row_sums[i+1] (forward uses NEXT vertex)
    let (d1_start, d1_end) = band.band_range(1);
    d[d1_start..d1_end].copy_from_slice(&row_sums[1..n]);

    let mut shifted = vec![0.0f64; n];

    // Pass 2+3: same recurrence as calculate_atx_band
    for k in 2..n {
        let band_len = n - k;
        let (km1_off, _) = band.band_range(k - 1);
        let (k_off, _) = band.band_range(k);
        let b_km1_off = band.band_offsets[k - 2];

        let (d_lo, d_hi) = d.split_at_mut(k_off);
        let d_out = &mut d_hi[..band_len];
        let d_km1 = &d_lo[km1_off..km1_off + band_len + 1];
        let b_inp = &b[b_km1_off + 1..b_km1_off + 1 + band_len];

        shifted[..band_len].copy_from_slice(&d_km1[1..band_len + 1]);
        let d_km1_base = &d_km1[..band_len];
        let shifted_slice = &shifted[..band_len];

        if k == 2 {
            vectorized_add3(d_out, d_km1_base, shifted_slice, b_inp);
        } else {
            let km2_off = band.band_offsets[k - 3];
            let d_km2 = &d_lo[km2_off + 1..km2_off + 1 + band_len];
            vectorized_add4(d_out, d_km1_base, shifted_slice, d_km2, b_inp);
        }
    }
}

/// Band-major kernel: compute d = A * b where A is the circular split design matrix.
/// Both b and d are in band-major layout.
pub fn calculate_ab_band(
    b: &[f64],
    d: &mut [f64],
    band: &BandIndex,
    row_sums: &mut [f64],
    shifted: &mut [f64],
) {
    let n = band.n;

    // Pass 1: compute row sums of b (using row-major accumulation order for FP compatibility)
    batch_rowsums_band(b, row_sums, band);

    // Initialize band 1 of d: d[i, i+1] = row_sums[i]
    let (d1_start, d1_end) = band.band_range(1);
    d[d1_start..d1_end].copy_from_slice(&row_sums[..n - 1]);

    // Pass 2+3: for k=2..n-1, compute band k from bands k-1 and k-2
    for k in 2..n {
        let band_len = n - k;
        let (km1_off, _) = band.band_range(k - 1);
        let (k_off, _) = band.band_range(k);
        let b_km1_off = band.band_offsets[k - 2];

        // Use split_at_mut to give the compiler non-aliasing slices.
        let (d_lo, d_hi) = d.split_at_mut(k_off);
        let d_out = &mut d_hi[..band_len];
        let d_km1 = &d_lo[km1_off..km1_off + band_len + 1];
        let b_inp = &b[b_km1_off..b_km1_off + band_len];

        // Copy the shifted view into a separate buffer so all loop accesses
        // are stride-1 with no overlap — enabling SIMD vectorization.
        shifted[..band_len].copy_from_slice(&d_km1[1..band_len + 1]);
        let d_km1_base = &d_km1[..band_len];
        let shifted_slice = &shifted[..band_len];

        if k == 2 {
            vectorized_add3(d_out, d_km1_base, shifted_slice, b_inp);
        } else {
            let km2_off = band.band_offsets[k - 3];
            let d_km2 = &d_lo[km2_off + 1..km2_off + 1 + band_len];
            vectorized_add4(d_out, d_km1_base, shifted_slice, d_km2, b_inp);
        }
    }
}

/// Band-major forward kernel with fused norm-squared computation.
/// Returns sum of element-wise squares of the output, accumulated per-row
/// in increasing band order (FP-identical to sum_array_squared_band).
pub fn calculate_forward_band_with_norm_sq(
    b: &[f64],
    d: &mut [f64],
    band: &BandIndex,
    row_sums: &mut [f64],
    row_sq_accum: &mut [f64],
    shifted: &mut [f64],
) -> f64 {
    let n = band.n;
    row_sq_accum[..n].fill(0.0);

    // Pass 1: row sums in forward order
    batch_rowsums_band_forward(b, row_sums, band);

    // Band 1: d(i, i+1) = row_sums[i+1]
    let (d1_start, d1_end) = band.band_range(1);
    d[d1_start..d1_end].copy_from_slice(&row_sums[1..n]);

    // Accumulate band 1 squares: element i in band 1 -> row i
    for i in 0..(n - 1) {
        let val = d[d1_start + i];
        row_sq_accum[i] += val * val;
    }

    // Pass 2+3: same recurrence as calculate_forward_band
    for k in 2..n {
        let band_len = n - k;
        let (km1_off, _) = band.band_range(k - 1);
        let (k_off, _) = band.band_range(k);
        let b_km1_off = band.band_offsets[k - 2];

        let (d_lo, d_hi) = d.split_at_mut(k_off);
        let d_out = &mut d_hi[..band_len];
        let d_km1 = &d_lo[km1_off..km1_off + band_len + 1];
        let b_inp = &b[b_km1_off + 1..b_km1_off + 1 + band_len];

        shifted[..band_len].copy_from_slice(&d_km1[1..band_len + 1]);
        let d_km1_base = &d_km1[..band_len];
        let shifted_slice = &shifted[..band_len];

        if k == 2 {
            vectorized_add3(d_out, d_km1_base, shifted_slice, b_inp);
        } else {
            let km2_off = band.band_offsets[k - 3];
            let d_km2 = &d_lo[km2_off + 1..km2_off + 1 + band_len];
            vectorized_add4(d_out, d_km1_base, shifted_slice, d_km2, b_inp);
        }

        // Accumulate band k squares: element i in band k -> row i
        for i in 0..band_len {
            let val = d_out[i];
            row_sq_accum[i] += val * val;
        }
    }

    // Sum per-row accumulators (matches row-wise order of sum_array_squared_band)
    let mut total = 0.0f64;
    for i in 0..n {
        total += row_sq_accum[i];
    }
    total
}

/// Band-major adjoint kernel with fused masked norm-squared computation.
/// Returns sum of element-wise squares of the output for non-active entries,
/// accumulated per-row in increasing band order (FP-identical to sum_array_squared_masked_band).
pub fn calculate_ab_band_with_masked_norm_sq(
    b: &[f64],
    d: &mut [f64],
    band: &BandIndex,
    row_sums: &mut [f64],
    row_sq_accum: &mut [f64],
    active_set: &[bool],
    shifted: &mut [f64],
) -> f64 {
    let n = band.n;
    row_sq_accum[..n].fill(0.0);

    // Pass 1: compute row sums of b (row-major accumulation order for FP compatibility)
    batch_rowsums_band(b, row_sums, band);

    // Initialize band 1 of d: d[i, i+1] = row_sums[i]
    let (d1_start, d1_end) = band.band_range(1);
    d[d1_start..d1_end].copy_from_slice(&row_sums[..n - 1]);

    // Accumulate band 1 masked squares: element i in band 1 -> row i
    for i in 0..(n - 1) {
        let band_idx = d1_start + i;
        if !active_set[band_idx] {
            let val = d[band_idx];
            row_sq_accum[i] += val * val;
        }
    }

    // Pass 2+3: for k=2..n-1, compute band k from bands k-1 and k-2
    for k in 2..n {
        let band_len = n - k;
        let (km1_off, _) = band.band_range(k - 1);
        let (k_off, _) = band.band_range(k);
        let b_km1_off = band.band_offsets[k - 2];

        let (d_lo, d_hi) = d.split_at_mut(k_off);
        let d_out = &mut d_hi[..band_len];
        let d_km1 = &d_lo[km1_off..km1_off + band_len + 1];
        let b_inp = &b[b_km1_off..b_km1_off + band_len];

        shifted[..band_len].copy_from_slice(&d_km1[1..band_len + 1]);
        let d_km1_base = &d_km1[..band_len];
        let shifted_slice = &shifted[..band_len];

        if k == 2 {
            vectorized_add3(d_out, d_km1_base, shifted_slice, b_inp);
        } else {
            let km2_off = band.band_offsets[k - 3];
            let d_km2 = &d_lo[km2_off + 1..km2_off + 1 + band_len];
            vectorized_add4(d_out, d_km1_base, shifted_slice, d_km2, b_inp);
        }

        // Accumulate band k masked squares
        let k_band_start = band.band_offsets[k - 1];
        for i in 0..band_len {
            let band_idx = k_band_start + i;
            if !active_set[band_idx] {
                let val = d_out[i];
                row_sq_accum[i] += val * val;
            }
        }
    }

    // Sum per-row accumulators
    let mut total = 0.0f64;
    for i in 0..n {
        total += row_sq_accum[i];
    }
    total
}

/// Band-major kernel: compute p = A^T * d where A is the circular split design matrix.
/// Both d and p are in band-major layout.
pub fn calculate_atx_band(
    d: &[f64],
    p: &mut [f64],
    band: &BandIndex,
    row_sums: &mut [f64],
    shifted: &mut [f64],
) {
    let n = band.n;

    // Pass 1: compute row sums of d (using row-major accumulation order for FP compatibility)
    batch_rowsums_band(d, row_sums, band);

    // Initialize band 1 of p: p[i, i+1] = row_sums[i+1]
    let (p1_start, p1_end) = band.band_range(1);
    p[p1_start..p1_end].copy_from_slice(&row_sums[1..n]);

    // Pass 2+3: for k=2..n-1, compute band k from bands k-1 and k-2
    // ATX: p_band_k[i] = p_band_{k-1}[i] + p_band_{k-1}[i+1] - p_band_{k-2}[i+1] - 2*d_band_{k-1}[i+1]
    for k in 2..n {
        let band_len = n - k;
        let (km1_off, _) = band.band_range(k - 1);
        let (k_off, _) = band.band_range(k);
        let d_km1_off = band.band_offsets[k - 2];

        // Use split_at_mut for non-aliasing slices
        let (p_lo, p_hi) = p.split_at_mut(k_off);
        let p_out = &mut p_hi[..band_len];
        let p_km1 = &p_lo[km1_off..km1_off + band_len + 1];
        let d_inp = &d[d_km1_off + 1..d_km1_off + 1 + band_len];

        // Copy the shifted view into a separate buffer so all loop accesses
        // are stride-1 with no overlap — enabling SIMD vectorization.
        shifted[..band_len].copy_from_slice(&p_km1[1..band_len + 1]);
        let p_km1_base = &p_km1[..band_len];
        let shifted_slice = &shifted[..band_len];

        if k == 2 {
            vectorized_add3(p_out, p_km1_base, shifted_slice, d_inp);
        } else {
            let km2_off = band.band_offsets[k - 3];
            let p_km2 = &p_lo[km2_off + 1..km2_off + 1 + band_len];
            vectorized_add4(p_out, p_km1_base, shifted_slice, p_km2, d_inp);
        }
    }
}

/// Compute out[i] = a[i] + b[i] - c[i] - 2*d[i] using platform SIMD.
/// All slices must have the same length.
#[inline]
pub fn vectorized_add4(out: &mut [f64], a: &[f64], b_arr: &[f64], c: &[f64], d_arr: &[f64]) {
    let len = out.len();
    debug_assert_eq!(len, a.len());
    debug_assert_eq!(len, b_arr.len());
    debug_assert_eq!(len, c.len());
    debug_assert_eq!(len, d_arr.len());

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = len / 2;
        // SAFETY: All pointers derived from slices of equal length `len` (checked by
        // debug_assert above). Vectorized loop reads/writes at offsets 0..chunks*2 ≤ len.
        // The scalar tail handles the final element when len is odd.
        unsafe {
            let two = vdupq_n_f64(2.0);
            let out_ptr = out.as_mut_ptr();
            let a_ptr = a.as_ptr();
            let b_ptr = b_arr.as_ptr();
            let c_ptr = c.as_ptr();
            let d_ptr = d_arr.as_ptr();
            for i in 0..chunks {
                let off = i * 2;
                let va = vld1q_f64(a_ptr.add(off));
                let vb = vld1q_f64(b_ptr.add(off));
                let vc = vld1q_f64(c_ptr.add(off));
                let vd = vld1q_f64(d_ptr.add(off));
                let result = vsubq_f64(vsubq_f64(vaddq_f64(va, vb), vc), vmulq_f64(two, vd));
                vst1q_f64(out_ptr.add(off), result);
            }
            if len % 2 != 0 {
                let i = chunks * 2;
                *out_ptr.add(i) =
                    *a_ptr.add(i) + *b_ptr.add(i) - *c_ptr.add(i) - 2.0 * *d_ptr.add(i);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: All pointers derived from slices of equal length `len` (checked by
        // debug_assert above). AVX reads 4 f64s per iteration, SSE2 reads 2. Scalar
        // tails handle remaining elements. Unaligned loads/stores are used throughout.
        unsafe {
            // AVX path: 4×f64 (256-bit)
            #[cfg(target_feature = "avx")]
            {
                let two = _mm256_set1_pd(2.0);
                let chunks = len / 4;
                let out_ptr = out.as_mut_ptr();
                let a_ptr = a.as_ptr();
                let b_ptr = b_arr.as_ptr();
                let c_ptr = c.as_ptr();
                let d_ptr = d_arr.as_ptr();
                for i in 0..chunks {
                    let off = i * 4;
                    let va = _mm256_loadu_pd(a_ptr.add(off));
                    let vb = _mm256_loadu_pd(b_ptr.add(off));
                    let vc = _mm256_loadu_pd(c_ptr.add(off));
                    let vd = _mm256_loadu_pd(d_ptr.add(off));
                    let result = _mm256_sub_pd(
                        _mm256_sub_pd(_mm256_add_pd(va, vb), vc),
                        _mm256_mul_pd(two, vd),
                    );
                    _mm256_storeu_pd(out_ptr.add(off), result);
                }
                for i in (chunks * 4)..len {
                    *out_ptr.add(i) =
                        *a_ptr.add(i) + *b_ptr.add(i) - *c_ptr.add(i) - 2.0 * *d_ptr.add(i);
                }
            }

            // SSE2 path: 2×f64 (128-bit, always available on x86_64)
            #[cfg(not(target_feature = "avx"))]
            {
                let two = _mm_set1_pd(2.0);
                let chunks = len / 2;
                let out_ptr = out.as_mut_ptr();
                let a_ptr = a.as_ptr();
                let b_ptr = b_arr.as_ptr();
                let c_ptr = c.as_ptr();
                let d_ptr = d_arr.as_ptr();
                for i in 0..chunks {
                    let off = i * 2;
                    let va = _mm_loadu_pd(a_ptr.add(off));
                    let vb = _mm_loadu_pd(b_ptr.add(off));
                    let vc = _mm_loadu_pd(c_ptr.add(off));
                    let vd = _mm_loadu_pd(d_ptr.add(off));
                    let result =
                        _mm_sub_pd(_mm_sub_pd(_mm_add_pd(va, vb), vc), _mm_mul_pd(two, vd));
                    _mm_storeu_pd(out_ptr.add(off), result);
                }
                if len % 2 != 0 {
                    let i = chunks * 2;
                    *out_ptr.add(i) =
                        *a_ptr.add(i) + *b_ptr.add(i) - *c_ptr.add(i) - 2.0 * *d_ptr.add(i);
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..len {
            out[i] = a[i] + b_arr[i] - c[i] - 2.0 * d_arr[i];
        }
    }
}

/// Compute out[i] = a[i] + b[i] - 2*c[i] using platform SIMD.
/// All slices must have the same length.
#[inline]
pub fn vectorized_add3(out: &mut [f64], a: &[f64], b_arr: &[f64], c: &[f64]) {
    let len = out.len();
    debug_assert_eq!(len, a.len());
    debug_assert_eq!(len, b_arr.len());
    debug_assert_eq!(len, c.len());

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let chunks = len / 2;
        // SAFETY: All pointers derived from slices of equal length `len` (checked by
        // debug_assert above). Vectorized loop reads/writes at offsets 0..chunks*2 ≤ len.
        // The scalar tail handles the final element when len is odd.
        unsafe {
            let two = vdupq_n_f64(2.0);
            let out_ptr = out.as_mut_ptr();
            let a_ptr = a.as_ptr();
            let b_ptr = b_arr.as_ptr();
            let c_ptr = c.as_ptr();
            for i in 0..chunks {
                let off = i * 2;
                let va = vld1q_f64(a_ptr.add(off));
                let vb = vld1q_f64(b_ptr.add(off));
                let vc = vld1q_f64(c_ptr.add(off));
                let result = vsubq_f64(vaddq_f64(va, vb), vmulq_f64(two, vc));
                vst1q_f64(out_ptr.add(off), result);
            }
            if len % 2 != 0 {
                let i = chunks * 2;
                *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i) - 2.0 * *c_ptr.add(i);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // SAFETY: All pointers derived from slices of equal length `len` (checked by
        // debug_assert above). AVX reads 4 f64s per iteration, SSE2 reads 2. Scalar
        // tails handle remaining elements. Unaligned loads/stores are used throughout.
        unsafe {
            #[cfg(target_feature = "avx")]
            {
                let two = _mm256_set1_pd(2.0);
                let chunks = len / 4;
                let out_ptr = out.as_mut_ptr();
                let a_ptr = a.as_ptr();
                let b_ptr = b_arr.as_ptr();
                let c_ptr = c.as_ptr();
                for i in 0..chunks {
                    let off = i * 4;
                    let va = _mm256_loadu_pd(a_ptr.add(off));
                    let vb = _mm256_loadu_pd(b_ptr.add(off));
                    let vc = _mm256_loadu_pd(c_ptr.add(off));
                    let result = _mm256_sub_pd(_mm256_add_pd(va, vb), _mm256_mul_pd(two, vc));
                    _mm256_storeu_pd(out_ptr.add(off), result);
                }
                for i in (chunks * 4)..len {
                    *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i) - 2.0 * *c_ptr.add(i);
                }
            }

            #[cfg(not(target_feature = "avx"))]
            {
                let two = _mm_set1_pd(2.0);
                let chunks = len / 2;
                let out_ptr = out.as_mut_ptr();
                let a_ptr = a.as_ptr();
                let b_ptr = b_arr.as_ptr();
                let c_ptr = c.as_ptr();
                for i in 0..chunks {
                    let off = i * 2;
                    let va = _mm_loadu_pd(a_ptr.add(off));
                    let vb = _mm_loadu_pd(b_ptr.add(off));
                    let vc = _mm_loadu_pd(c_ptr.add(off));
                    let result = _mm_sub_pd(_mm_add_pd(va, vb), _mm_mul_pd(two, vc));
                    _mm_storeu_pd(out_ptr.add(off), result);
                }
                if len % 2 != 0 {
                    let i = chunks * 2;
                    *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i) - 2.0 * *c_ptr.add(i);
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..len {
            out[i] = a[i] + b_arr[i] - 2.0 * c[i];
        }
    }
}

/* -------- tests -------- */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_index_n4_offsets_and_npairs() {
        let b = BandIndex::new(4);
        assert_eq!(b.n, 4);
        assert_eq!(b.npairs, 6); // 4*3/2
        // band 1: pairs (0,1),(1,2),(2,3) → 3 elements starting at 0
        assert_eq!(b.band_range(1), (0, 3));
        // band 2: pairs (0,2),(1,3) → 2 elements
        assert_eq!(b.band_range(2), (3, 5));
        // band 3: pair (0,3) → 1 element
        assert_eq!(b.band_range(3), (5, 6));
    }

    #[test]
    fn row_to_band_roundtrip() {
        for n in [3, 5, 8, 10] {
            let b = BandIndex::new(n);
            let src: Vec<f64> = (0..b.npairs).map(|i| i as f64).collect();
            let mut band_buf = vec![0.0; b.npairs];
            let mut row_buf = vec![0.0; b.npairs];

            row_to_band(&src, &mut band_buf, &b);
            band_to_row(&band_buf, &mut row_buf, &b);

            assert_eq!(src, row_buf, "roundtrip failed for n={n}");
        }
    }

    #[test]
    fn band_to_row_identity_mapping() {
        let b = BandIndex::new(4);
        // Place 1.0 in each band position and verify it maps to the correct row-major slot
        for band_idx in 0..b.npairs {
            let mut band_buf = vec![0.0; b.npairs];
            band_buf[band_idx] = 1.0;
            let mut row_buf = vec![0.0; b.npairs];
            band_to_row(&band_buf, &mut row_buf, &b);
            // Exactly one element should be 1.0
            assert_eq!(
                row_buf.iter().filter(|&&v| v == 1.0).count(),
                1,
                "band_idx={band_idx}"
            );
        }
    }

    #[test]
    fn batch_rowsums_band_matches_naive() {
        let n = 5;
        let b = BandIndex::new(n);
        // Row-major data: pair(i,j) = (i+1)*(j+1)
        let mut row_data = vec![0.0; b.npairs];
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                row_data[idx] = ((i + 1) * (j + 1)) as f64;
                idx += 1;
            }
        }

        // Naive row sums from row-major
        let mut expected = vec![0.0; n];
        idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                expected[i] += row_data[idx];
                expected[j] += row_data[idx];
                idx += 1;
            }
        }

        // Convert to band and compute
        let mut band_data = vec![0.0; b.npairs];
        row_to_band(&row_data, &mut band_data, &b);
        let mut sums = vec![0.0; n];
        batch_rowsums_band(&band_data, &mut sums, &b);

        for i in 0..n {
            assert!(
                (sums[i] - expected[i]).abs() < 1e-12,
                "rowsum mismatch at i={i}: got={} expected={}",
                sums[i],
                expected[i]
            );
        }
    }

    #[test]
    fn batch_rowsums_sequential_matches_standard() {
        let n = 6;
        let b = BandIndex::new(n);
        let band_data: Vec<f64> = (0..b.npairs).map(|i| (i as f64) * 0.1 + 1.0).collect();

        let mut sums_std = vec![0.0; n];
        let mut sums_seq = vec![0.0; n];
        batch_rowsums_band(&band_data, &mut sums_std, &b);
        batch_rowsums_band_sequential(&band_data, &mut sums_seq, &b);

        for i in 0..n {
            assert!(
                (sums_std[i] - sums_seq[i]).abs() < 1e-10,
                "sequential vs standard mismatch at i={i}: std={} seq={}",
                sums_std[i],
                sums_seq[i]
            );
        }
    }

    #[test]
    fn batch_rowsums_band_forward_total_matches() {
        // Forward sums should have the same total as standard sums
        let n = 7;
        let b = BandIndex::new(n);
        let band_data: Vec<f64> = (0..b.npairs).map(|i| (i as f64) * 0.3 + 0.5).collect();

        let mut sums_std = vec![0.0; n];
        let mut sums_fwd = vec![0.0; n];
        batch_rowsums_band(&band_data, &mut sums_std, &b);
        batch_rowsums_band_forward(&band_data, &mut sums_fwd, &b);

        let total_std: f64 = sums_std.iter().sum();
        let total_fwd: f64 = sums_fwd.iter().sum();
        assert!(
            (total_std - total_fwd).abs() < 1e-10,
            "forward sum total mismatch: std={total_std} fwd={total_fwd}"
        );
    }

    #[test]
    fn vectorized_add4_matches_scalar() {
        let len = 17; // odd, tests scalar tail
        let a: Vec<f64> = (0..len).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..len).map(|i| (i * 2) as f64).collect();
        let c: Vec<f64> = (0..len).map(|i| (i * 3) as f64).collect();
        let d: Vec<f64> = (0..len).map(|i| (i + 1) as f64).collect();

        let mut out = vec![0.0; len];
        vectorized_add4(&mut out, &a, &b, &c, &d);

        for i in 0..len {
            let expected = a[i] + b[i] - c[i] - 2.0 * d[i];
            assert!(
                (out[i] - expected).abs() < 1e-12,
                "add4 mismatch at i={i}: got={} expected={expected}",
                out[i]
            );
        }
    }

    #[test]
    fn vectorized_add3_matches_scalar() {
        let len = 13;
        let a: Vec<f64> = (0..len).map(|i| (i * 5) as f64).collect();
        let b: Vec<f64> = (0..len).map(|i| (i * 3) as f64).collect();
        let c: Vec<f64> = (0..len).map(|i| (i + 2) as f64).collect();

        let mut out = vec![0.0; len];
        vectorized_add3(&mut out, &a, &b, &c);

        for i in 0..len {
            let expected = a[i] + b[i] - 2.0 * c[i];
            assert!(
                (out[i] - expected).abs() < 1e-12,
                "add3 mismatch at i={i}: got={} expected={expected}",
                out[i]
            );
        }
    }

    #[test]
    fn vectorized_empty_slices() {
        let mut out: Vec<f64> = vec![];
        vectorized_add3(&mut out, &[], &[], &[]);
        vectorized_add4(&mut out, &[], &[], &[], &[]);
        // Should not panic
    }

    #[test]
    fn calculate_ab_band_is_linear_operator() {
        // A*(alpha*x) should equal alpha*A*x for any scalar alpha
        let n = 5;
        let b = BandIndex::new(n);
        let alpha = 3.7;
        let x: Vec<f64> = (0..b.npairs).map(|i| (i as f64) * 0.1 + 1.0).collect();
        let x_scaled: Vec<f64> = x.iter().map(|&v| v * alpha).collect();

        let mut d1 = vec![0.0; b.npairs];
        let mut d2 = vec![0.0; b.npairs];
        let mut rs = vec![0.0; n];
        let mut shifted = vec![0.0; n];

        calculate_ab_band(&x, &mut d1, &b, &mut rs, &mut shifted);
        calculate_ab_band(&x_scaled, &mut d2, &b, &mut rs, &mut shifted);

        for i in 0..b.npairs {
            assert!(
                (d2[i] - alpha * d1[i]).abs() < 1e-9,
                "linearity failed at i={i}: A*(alpha*x)={} vs alpha*A*x={}",
                d2[i],
                alpha * d1[i]
            );
        }
    }

    #[test]
    fn calculate_atx_band_transpose_dot_product_identity() {
        // For any x,y: <Ax, y> should equal <x, A^T y>
        let n = 5;
        let b = BandIndex::new(n);
        let x: Vec<f64> = (0..b.npairs).map(|i| (i as f64) * 0.2 + 0.5).collect();
        let y: Vec<f64> = (0..b.npairs).map(|i| (i as f64) * 0.3 + 1.0).collect();

        let mut ax = vec![0.0; b.npairs];
        let mut aty = vec![0.0; b.npairs];
        let mut rs = vec![0.0; n];
        let mut shifted = vec![0.0; n];

        calculate_ab_band(&x, &mut ax, &b, &mut rs, &mut shifted);
        calculate_atx_band(&y, &mut aty, &b, &mut rs, &mut shifted);

        let dot_ax_y: f64 = ax.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let dot_x_aty: f64 = x.iter().zip(aty.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (dot_ax_y - dot_x_aty).abs() < 1e-8,
            "<Ax,y>={dot_ax_y} != <x,A^Ty>={dot_x_aty}"
        );
    }

    #[test]
    fn fused_norm_matches_separate_computation() {
        let n = 6;
        let b = BandIndex::new(n);
        let x: Vec<f64> = (0..b.npairs).map(|i| (i as f64) * 0.1 + 0.5).collect();

        let mut d1 = vec![0.0; b.npairs];
        let mut d2 = vec![0.0; b.npairs];
        let mut rs = vec![0.0; n];
        let mut row_sq = vec![0.0; n];
        let mut shifted = vec![0.0; n];

        // Fused version
        let fused_norm = calculate_forward_band_with_norm_sq(
            &x, &mut d1, &b, &mut rs, &mut row_sq, &mut shifted,
        );

        // Separate: compute forward then sum squares
        calculate_forward_band(&x, &mut d2, &b, &mut rs);
        let separate_norm: f64 = d2.iter().map(|v| v * v).sum();

        assert!(
            (fused_norm - separate_norm).abs() < 1e-10,
            "fused={fused_norm} vs separate={separate_norm}"
        );

        // Outputs should also match
        for i in 0..b.npairs {
            assert!(
                (d1[i] - d2[i]).abs() < 1e-12,
                "output mismatch at i={i}: fused={} separate={}",
                d1[i],
                d2[i]
            );
        }
    }
}

#[derive(Default, Debug)]
pub struct BandKernelPerf {
    pub calc_ab_calls: usize,
    pub calc_atx_calls: usize,
    pub calc_ab_time: Duration,
    pub calc_atx_time: Duration,
}

#[inline]
pub fn profiled_calculate_ab_band(
    b: &[f64],
    d: &mut [f64],
    band: &BandIndex,
    perf: &mut BandKernelPerf,
    track: bool,
    row_sums: &mut [f64],
    shifted: &mut [f64],
) {
    if track {
        let t0 = Instant::now();
        calculate_ab_band(b, d, band, row_sums, shifted);
        perf.calc_ab_calls += 1;
        perf.calc_ab_time += t0.elapsed();
    } else {
        calculate_ab_band(b, d, band, row_sums, shifted);
    }
}

#[inline]
pub fn profiled_calculate_atx_band(
    d: &[f64],
    p: &mut [f64],
    band: &BandIndex,
    perf: &mut BandKernelPerf,
    track: bool,
    row_sums: &mut [f64],
    shifted: &mut [f64],
) {
    if track {
        let t0 = Instant::now();
        calculate_atx_band(d, p, band, row_sums, shifted);
        perf.calc_atx_calls += 1;
        perf.calc_atx_time += t0.elapsed();
    } else {
        calculate_atx_band(d, p, band, row_sums, shifted);
    }
}

#[inline]
pub fn avg_band_time_us(total: Duration, calls: usize) -> f64 {
    if calls == 0 {
        0.0
    } else {
        total.as_secs_f64() * 1e6 / calls as f64
    }
}

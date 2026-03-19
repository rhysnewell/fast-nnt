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
pub fn calculate_ab_band(b: &[f64], d: &mut [f64], band: &BandIndex, row_sums: &mut [f64]) {
    let n = band.n;

    // Pass 1: compute row sums of b (using row-major accumulation order for FP compatibility)
    batch_rowsums_band(b, row_sums, band);

    // Initialize band 1 of d: d[i, i+1] = row_sums[i]
    let (d1_start, d1_end) = band.band_range(1);
    d[d1_start..d1_end].copy_from_slice(&row_sums[..n - 1]);

    // Pre-allocate buffer for the shifted slice d_km1[1..].
    // This eliminates the overlapping access pattern d_km1[i]/d_km1[i+1]
    // that prevents SIMD auto-vectorization.
    let mut shifted = vec![0.0f64; n];

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

/// Band-major kernel: compute p = A^T * d where A is the circular split design matrix.
/// Both d and p are in band-major layout.
pub fn calculate_atx_band(d: &[f64], p: &mut [f64], band: &BandIndex, row_sums: &mut [f64]) {
    let n = band.n;

    // Pass 1: compute row sums of d (using row-major accumulation order for FP compatibility)
    batch_rowsums_band(d, row_sums, band);

    // Initialize band 1 of p: p[i, i+1] = row_sums[i+1]
    let (p1_start, p1_end) = band.band_range(1);
    p[p1_start..p1_end].copy_from_slice(&row_sums[1..n]);

    // Pre-allocate buffer for the shifted slice p_km1[1..].
    let mut shifted = vec![0.0f64; n];

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
) {
    if track {
        let t0 = Instant::now();
        calculate_ab_band(b, d, band, row_sums);
        perf.calc_ab_calls += 1;
        perf.calc_ab_time += t0.elapsed();
    } else {
        calculate_ab_band(b, d, band, row_sums);
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
) {
    if track {
        let t0 = Instant::now();
        calculate_atx_band(d, p, band, row_sums);
        perf.calc_atx_calls += 1;
        perf.calc_atx_time += t0.elapsed();
    } else {
        calculate_atx_band(d, p, band, row_sums);
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

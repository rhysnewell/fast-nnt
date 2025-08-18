use crate::splits::asplit::ASplit; // adjust path if needed
use fixedbitset::FixedBitSet;
use ndarray::Array2;
use rayon::prelude::*;

/// Compute the least-squares fit (%) of the given splits to the distances.
/// - `distances`: symmetric n×n (0-based indices)
/// - `splits`: list of ASplit with 1-based bitsets (bit 0 ignored)
///
/// Returns a percentage in [0, 100]. Matches Java behavior:
///   100 * (1 - sum((s_ij - d_ij)^2) / sum(d_ij^2))  over i<j, 1-based.
pub fn compute_least_squares_fit(distances: &Array2<f64>, splits: &[ASplit]) -> f32 {
    let n = distances.nrows();
    assert_eq!(n, distances.ncols(), "distances must be square");

    // Build the split-induced distance matrix S (0-based) in parallel:
    // S[i,j] = Σ_{splits} weight * [i∈A && j∈B or i∈B && j∈A]
    let split_dist = splits
        .par_iter()
        .map(|s| {
            let mut m = Array2::<f64>::zeros((n, n));
            let w = s.get_weight();
            let a: &FixedBitSet = s.get_a();
            let b: &FixedBitSet = s.get_b();

            for i1 in a.ones() {
                if i1 == 0 || i1 > n {
                    continue;
                }
                let ii = i1 - 1; // 0-based
                for j1 in b.ones() {
                    if j1 == 0 || j1 > n {
                        continue;
                    }
                    let jj = j1 - 1;
                    m[[ii, jj]] += w;
                    m[[jj, ii]] += w; // symmetric
                }
            }
            m
        })
        .reduce(
            || Array2::<f64>::zeros((n, n)),
            |mut acc, m| {
                acc.zip_mut_with(&m, |a, b| *a += *b);
                acc
            },
        );

    // Sum over the upper triangle (i<j) in parallel
    let (sum_diff_sq, sum_d_sq) = (0..n - 1)
        .into_par_iter()
        .map(|i| {
            let mut diff_sum = 0.0;
            let mut d_sum = 0.0;
            for j in (i + 1)..n {
                let sij = split_dist[[i, j]];
                let dij = distances[[i, j]];
                let diff = sij - dij;
                diff_sum += diff * diff;
                d_sum += dij * dij;
            }
            (diff_sum, d_sum)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    let fit = if sum_d_sq > 0.0 {
        100.0 * (1.0 - (sum_diff_sq / sum_d_sq))
    } else {
        0.0
    };

    fit as f32
}

#[cfg(test)]
mod lsq_tests {
    use crate::weights::active_set_weights::{NNLSParams, compute_asplits};

    use super::*;
    use ndarray::{Array2, arr2};

    fn bs_from(indices: &[usize], len: usize) -> FixedBitSet {
        let mut bs = FixedBitSet::with_capacity(len + 1);
        bs.grow(len + 1);
        for &i in indices {
            bs.set(i, true); // 1-based
        }
        bs
    }

    /// Build a symmetric distance matrix from a set of splits (same logic as the evaluator).
    fn distances_from_splits(n: usize, splits: &[ASplit]) -> Array2<f64> {
        let mut d = Array2::<f64>::zeros((n, n));
        for s in splits {
            let w = s.get_weight();
            let a = s.get_a();
            let b = s.get_b();
            for i1 in a.ones() {
                if i1 == 0 || i1 > n {
                    continue;
                }
                let ii = i1 - 1;
                for j1 in b.ones() {
                    if j1 == 0 || j1 > n {
                        continue;
                    }
                    let jj = j1 - 1;
                    d[[ii, jj]] += w;
                    d[[jj, ii]] += w;
                }
            }
        }
        d
    }

    #[test]
    fn lsq_perfect_fit() {
        let n = 6;

        // Construct a few splits on 1-based taxa
        let s1 = ASplit::from_a_ntax_with_weight(bs_from(&[1, 2], n), n, 1.0);
        let s2 = ASplit::from_a_ntax_with_weight(bs_from(&[2, 3, 4], n), n, 0.7);
        let s3 = ASplit::from_a_ntax_with_weight(bs_from(&[5], n), n, 0.4); // trivial

        let splits = vec![s1, s2, s3];
        let d = distances_from_splits(n, &splits);

        let fit = compute_least_squares_fit(&d, &splits);
        assert!((fit - 100.0).abs() < 1e-6, "fit was {fit}");
    }

    #[test]
    fn lsq_imperfect_fit_with_noise() {
        let n = 6;
        let s1 = ASplit::from_a_ntax_with_weight(bs_from(&[1, 2], n), n, 1.0);
        let s2 = ASplit::from_a_ntax_with_weight(bs_from(&[2, 3, 4], n), n, 0.7);
        let s3 = ASplit::from_a_ntax_with_weight(bs_from(&[5], n), n, 0.4);

        let splits = vec![s1, s2, s3];
        let mut d = distances_from_splits(n, &splits);

        // Add small noise on upper triangle
        for i in 0..n {
            for j in (i + 1)..n {
                d[[i, j]] += 0.01 * ((i + j) as f64);
                d[[j, i]] = d[[i, j]];
            }
        }

        let fit = compute_least_squares_fit(&d, &splits);
        assert!(
            fit < 100.0 && fit > 0.0,
            "fit should drop below 100, got {fit}"
        );
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

        let splits = compute_asplits(&ord, &d, &mut params, None).expect("ASplits solved");
        let fit = compute_least_squares_fit(&d, &splits);

        println!("Fit: {}", fit);

        assert_eq!(fit, 93.477936, "Expected perfect fit for this example");
    }
}

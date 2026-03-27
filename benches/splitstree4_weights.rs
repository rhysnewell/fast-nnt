use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use anon_nnt::weights::splitstree4_weights::compute_splits;

fn make_distance_matrix(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let v = rng.gen_range(0.0..10.0);
            m[[i, j]] = v;
            m[[j, i]] = v;
        }
    }
    m
}

fn bench_splitstree4_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("splitstree4_weights");
    group.sample_size(10);

    // Small sizes (unpreconditioned path, npairs < 100_000)
    for &n in &[40usize, 80, 160] {
        let dist = make_distance_matrix(n, 2024);
        // cycle: 0-prefixed identity ordering [0, 1, 2, ..., n]
        let cycle: Vec<usize> = (0..=n).collect();

        group.bench_with_input(
            BenchmarkId::new("weights", n),
            &(dist, cycle),
            |b, (d, c)| {
                b.iter(|| {
                    let _ = compute_splits(black_box(c), black_box(d)).unwrap();
                });
            },
        );
    }

    // Larger sizes that exercise the preconditioner (npairs > 100_000 => n > ~450)
    for &n in &[320usize, 500] {
        let dist = make_distance_matrix(n, 2024);
        let cycle: Vec<usize> = (0..=n).collect();

        group.bench_with_input(
            BenchmarkId::new("weights", n),
            &(dist, cycle),
            |b, (d, c)| {
                b.iter(|| {
                    let _ = compute_splits(black_box(c), black_box(d)).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_splitstree4_weights);
criterion_main!(benches);

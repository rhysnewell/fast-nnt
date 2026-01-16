use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::ThreadPoolBuilder;

use fast_nnt::weights::active_set_weights::{NNLSParams, compute_use_1d};

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

fn bench_nnls(c: &mut Criterion) {
    let mut group = c.benchmark_group("nnls");
    for &n in &[40usize, 60, 80] {
        let dist = make_distance_matrix(n, 2024);
        let mut cycle = Vec::with_capacity(n + 1);
        cycle.push(0);
        cycle.extend(1..=n);

        let max_threads = num_cpus::get();
        let thread_counts: Vec<usize> = [1usize, 2, 4, 8, 16]
            .into_iter()
            .filter(|&t| t <= max_threads)
            .collect();

        for &threads in &thread_counts {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let id = BenchmarkId::new(format!("t{}", threads), n);
            group.bench_with_input(id, &dist, |b, d| {
                b.iter(|| {
                    pool.install(|| {
                        let mut params = NNLSParams::default();
                        let _ = compute_use_1d(black_box(&cycle), black_box(d), &mut params, None)
                            .unwrap();
                    });
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_nnls);
criterion_main!(benches);

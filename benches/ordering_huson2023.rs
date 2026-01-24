use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::ThreadPoolBuilder;
use std::time::Duration;

use fastnnt::ordering::ordering_huson2023::compute_order_huson_2023;

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

fn bench_huson2023(c: &mut Criterion) {
    let mut group = c.benchmark_group("huson2023_ordering");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for &n in &[60usize, 120, 240] {
        let dist = make_distance_matrix(n, 1337);

        let max_threads = num_cpus::get();
        let thread_counts: Vec<usize> = [4, 8, 16]
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
                        let _ = compute_order_huson_2023(black_box(d));
                    });
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_huson2023);
criterion_main!(benches);

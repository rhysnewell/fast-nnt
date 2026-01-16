use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fast_nnt::ordering::ordering_splitstree4::{
    SxMode, compute_order_splits_tree4_with_sx,
};
use rayon::ThreadPoolBuilder;

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

fn bench_splitstree4_sx(c: &mut Criterion) {
    let mut group = c.benchmark_group("splitstree4_sx");
    group.sample_size(10);
    for &n in &[80usize, 160, 320, 640] {
        let dist = make_distance_matrix(n, 42);

        let max_threads = num_cpus::get();
        let thread_counts: Vec<usize> = [2usize, 4, 8, 16]
            .into_iter()
            .filter(|&t| t <= max_threads)
            .collect();

        for &threads in &thread_counts {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let id = BenchmarkId::new(format!("parallel_t{}", threads), n);
            group.bench_with_input(id, &dist, |b, d| {
                b.iter(|| {
                    pool.install(|| {
                        let _ = compute_order_splits_tree4_with_sx(
                            black_box(d),
                            SxMode::Parallel,
                        )
                        .unwrap();
                    });
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("serial", n), &dist, |b, d| {
            b.iter(|| {
                let _ = compute_order_splits_tree4_with_sx(black_box(d), SxMode::Serial).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_splitstree4_sx);
criterion_main!(benches);

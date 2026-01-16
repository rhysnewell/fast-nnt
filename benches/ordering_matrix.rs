use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fast_nnt::ordering::ordering_matrix::neighbor_net_ordering;

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

fn bench_ordering_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("ordering_matrix");
    group.sample_size(10);
    for &n in &[60usize, 120, 240] {
        let dist = make_distance_matrix(n, 4242);
        group.bench_with_input(BenchmarkId::new("ordering", n), &dist, |b, d| {
            b.iter(|| {
                let _ = neighbor_net_ordering(black_box(d));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ordering_matrix);
criterion_main!(benches);

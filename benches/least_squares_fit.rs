use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use fixedbitset::FixedBitSet;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use fastnnt::splits::asplit::ASplit;
use fastnnt::utils::compute_least_squares_fit;

fn make_random_split(ntax: usize, rng: &mut StdRng) -> ASplit {
    let mut a = FixedBitSet::with_capacity(ntax + 1);
    a.grow(ntax + 1);

    // Ensure A is non-empty and not full.
    let target = rng.gen_range(1..ntax);
    while a.count_ones(..) < target {
        let t = rng.gen_range(1..=ntax);
        a.insert(t);
    }
    let w = rng.gen_range(0.1..2.0);
    ASplit::from_a_ntax_with_weight(a, ntax, w)
}

fn distances_from_splits(ntax: usize, splits: &[ASplit]) -> Array2<f64> {
    let mut d = Array2::<f64>::zeros((ntax, ntax));
    for s in splits {
        let w = s.get_weight();
        let a = s.get_a();
        let b = s.get_b();
        for i1 in a.ones() {
            if i1 == 0 || i1 > ntax {
                continue;
            }
            let ii = i1 - 1;
            for j1 in b.ones() {
                if j1 == 0 || j1 > ntax {
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

fn bench_least_squares_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("least_squares_fit");
    group.sample_size(10);
    for &ntax in &[50usize, 100, 200] {
        let mut rng = StdRng::seed_from_u64(7);
        let nsplits = ntax * 2;
        let splits: Vec<ASplit> = (0..nsplits).map(|_| make_random_split(ntax, &mut rng)).collect();
        let distances = distances_from_splits(ntax, &splits);

        group.bench_with_input(BenchmarkId::new("fit", ntax), &distances, |b, d| {
            b.iter(|| {
                let _ = compute_least_squares_fit(black_box(d), black_box(&splits));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_least_squares_fit);
criterion_main!(benches);

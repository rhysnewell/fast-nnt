use fixedbitset::FixedBitSet;

/// Create a FixedBitSet from 1-based indices with capacity `len + 1`.
pub fn bs_from(indices: &[usize], len: usize) -> FixedBitSet {
    let mut bs = FixedBitSet::with_capacity(len + 1);
    bs.grow(len + 1);
    for &i in indices {
        bs.set(i, true);
    }
    bs
}

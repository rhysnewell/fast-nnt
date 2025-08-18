use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use fixedbitset::FixedBitSet;

use crate::splits::asplit::ASplit;
use crate::algorithms::equal_angle::normalize_cycle_1based;

/// Parity with Java enum; keep it simple for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compatibility {
    Unknown,
    Compatible,
    Cyclic,
    WeaklyCompatible,
    Incompatible,
}

impl Default for Compatibility {
    fn default() -> Self { Compatibility::Unknown }
}

/// Placeholder; extend as needed to mirror SplitsFormat from Java.
#[derive(Debug, Clone, Default)]
pub struct SplitsFormat {
    pub labels: bool,
    pub weights: bool,
    pub confidences: bool,
    pub intervals: bool,
    pub show_both_sides: bool,
}

/// Container of circular splits for Equal-Angle / NEXUS writing.
#[derive(Debug, Clone, Default)]
pub struct SplitsBlock {
    pub splits: Vec<ASplit>,                 // 1-based access via get(i)
    compatibility: Compatibility,
    fit: f32,
    threshold: f32,
    partial: bool,
    cycle: Option<Vec<usize>>,           // 1-based: [0, t1, ..., tn]
    split_labels: BTreeMap<usize, String>,
    format: SplitsFormat,
}

impl SplitsBlock {
    pub fn new() -> Self {
        Self {
            splits: Vec::new(),
            compatibility: Compatibility::Unknown,
            fit: -1.0,
            threshold: 0.0,
            partial: false,
            cycle: None,
            split_labels: BTreeMap::new(),
            format: SplitsFormat { labels: false, weights: true, confidences: false, intervals: false, show_both_sides: false },
        }
    }

    pub fn clear(&mut self) {
        self.splits.clear();
        self.compatibility = Compatibility::Unknown;
        self.fit = -1.0;
        self.threshold = 0.0;
        self.partial = false;
        self.cycle = None;
        self.split_labels.clear();
        self.format = SplitsFormat::default();
    }

    pub fn set_splits(&mut self, splits: Vec<ASplit>) {
        self.splits = splits;
    }

    pub fn nsplits(&self) -> usize { self.splits.len() }
    pub fn splits(&self) -> impl Iterator<Item=&ASplit> { self.splits.iter() }
    pub fn splits_mut(&mut self) -> impl Iterator<Item=&mut ASplit> { self.splits.iter_mut() }
    pub fn get_splits(&self) -> &[ASplit] { &self.splits }

    /// 1-based accessor (panics if out of range, matches Java’s get(i))
    pub fn get(&self, i: usize) -> &ASplit {
        &self.splits[i - 1]
    }

    /// Push a split; returns its 1-based id.
    pub fn push(&mut self, s: ASplit) -> usize {
        self.splits.push(s);
        self.splits.len()
    }

    pub fn set_compatibility(&mut self, c: Compatibility) { self.compatibility = c; }
    pub fn compatibility(&self) -> Compatibility { self.compatibility }

    pub fn set_fit(&mut self, fit: f32) { self.fit = fit; }
    pub fn fit(&self) -> f32 { self.fit }

    pub fn set_threshold(&mut self, thr: f32) { self.threshold = thr; }
    pub fn threshold(&self) -> f32 { self.threshold }

    pub fn set_partial(&mut self, partial: bool) { self.partial = partial; }
    pub fn partial(&self) -> bool { self.partial }

    pub fn format(&self) -> &SplitsFormat { &self.format }
    pub fn format_mut(&mut self) -> &mut SplitsFormat { &mut self.format }

    pub fn split_labels(&self) -> &BTreeMap<usize, String> { &self.split_labels }
    pub fn split_labels_mut(&mut self) -> &mut BTreeMap<usize, String> { &mut self.split_labels }
    pub fn set_split_label<S: Into<String>>(&mut self, sid: usize, label: S) {
        self.split_labels.insert(sid, label.into());
    }

    /// Get 1-based cycle; returns `None` if not set.
    /// This is the `[0, t1, ..., tn]` representation expected by equal-angle.
    pub fn cycle(&self) -> Option<&[usize]> {
        self.cycle.as_deref()
    }

    /// Set a **1-based** cycle `[0, t1..tn]`. If `normalize=true`, rotate so that `cycle[1]==1`.
    pub fn set_cycle(&mut self, mut cycle: Vec<usize>, normalize: bool) -> Result<()> {
        if normalize {
            cycle = normalize_cycle_1based(&cycle)?;
        }
        // sanity: ensure it's a permutation (rough check)
        let mut seen = FixedBitSet::with_capacity(cycle.len() + 1);
        for &v in &cycle {
            if v >= seen.len() { seen.grow(v + 1); }
            if seen.contains(v) { // duplicate
                self.cycle = None;
                return Err(anyhow::anyhow!("cycle is not a permutation"));
            }
            seen.insert(v);
        }
        self.cycle = Some(cycle);
        Ok(())
    }

    /// Set from 0-based list `[t1..tn]` (no sentinel). We’ll prepend a 0.
    pub fn set_cycle_from_zero_based(&mut self, order: &[usize], normalize: bool) {
        let mut c = Vec::with_capacity(order.len() + 1);
        c.push(0);
        c.extend_from_slice(order);
        self.set_cycle(c, normalize);
    }

    /// If cycle is missing, default to identity `[0, 1, 2, ..., ntax]`.
    pub fn ensure_cycle_default(&mut self, ntax: usize) {
        if self.cycle.is_none() {
            let mut c = Vec::with_capacity(ntax + 1);
            c.push(0);
            for t in 1..=ntax { c.push(t); }
            // normalize will keep 1 at position 1
            self.set_cycle(c, true);
        }
    }

    /// Returns the set of taxa in (P.sideP) ∩ (Q.sideQ).
    /// `side=true` means A-part; `false` means B-part.
    pub fn intersect2(&self, split_p: usize, side_p: bool, split_q: usize, side_q: bool) -> FixedBitSet {
        let sp = self.get(split_p);
        let sq = self.get(split_q);
        let pa = if side_p { sp.get_a() } else { sp.get_b() };
        let qb = if side_q { sq.get_a() } else { sq.get_b() };
        fb_and(pa, qb)
    }

    pub fn has_confidence_values(&self) -> bool {
        self.splits.iter().any(|s| s.get_confidence() != -1.0)
    }

    /// Find a split equal to `split` (by A/B equality); returns 1-based id if found.
    pub fn index_of(&self, target: &ASplit) -> Option<usize> {
        self.splits.iter().position(|s| s == target).map(|i| i + 1)
    }

}


/* ---------- FixedBitSet helpers ---------- */

fn fb_grow_to(mut bs: FixedBitSet, len: usize) -> FixedBitSet {
    if bs.len() < len { bs.grow(len); }
    bs
}

fn fb_and(a: &FixedBitSet, b: &FixedBitSet) -> FixedBitSet {
    let len = a.len().max(b.len());
    let mut out = FixedBitSet::with_capacity(len);
    out.grow(len);
    // Make owned copies grown to same length to use bitwise ops
    let mut ca = a.clone(); ca.grow(len);
    let mut cb = b.clone(); cb.grow(len);
    ca.intersect_with(&cb);
    ca
}

/* ---------- Equal-Angle adapter ---------- */
pub trait SplitsProvider {
    fn nsplits(&self) -> usize;
    fn split(&self, id: usize) -> &ASplit;
    fn cycle(&self) -> Option<&[usize]>;
}

impl SplitsProvider for SplitsBlock {
    fn nsplits(&self) -> usize { self.nsplits() }
    fn split(&self, id: usize) -> &ASplit { self.get(id) }
    fn cycle(&self) -> Option<&[usize]> { self.cycle() }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::equal_angle::{assign_angles_to_splits, normalize_cycle_1based};
    use fixedbitset::FixedBitSet;

    fn mk_bitset(ntax: usize, elems: &[usize]) -> FixedBitSet {
        let mut bs = FixedBitSet::with_capacity(ntax + 1);
        bs.grow(ntax + 1);
        for &t in elems {
            bs.insert(t);
        }
        bs
    }

    // You’ll need a constructor for ASplit like:
    // ASplit::from_parts(a: FixedBitSet, ntax: usize, weight: f64, confidence: f64, label: Option<String>)
    // or adapt to whatever you implemented earlier.
    fn mk_split(a: &[usize], ntax: usize, w: f64) -> ASplit {
        let a_bs = mk_bitset(ntax, a);
        ASplit::from_a_ntax_with_weight_conf(a_bs, ntax, w, -1.0)
    }

    #[test]
    fn angles_compute_on_small_block() {
        let ntax = 4;
        // cycle [0,1,2,3,4]
        let mut cycle = vec![0, 1, 2, 3, 4];
        cycle = normalize_cycle_1based(&cycle).expect("valid cycle");

        let mut sb = SplitsBlock::new();
        // split {1}|{2,3,4} weight 1.0
        sb.push(mk_split(&[1], ntax, 1.0));
        // split {1,2}|{3,4} weight 0.5
        sb.push(mk_split(&[1,2], ntax, 0.5));
        sb.set_cycle(cycle.clone(), true);

        let angs = assign_angles_to_splits(ntax, &sb, sb.cycle().unwrap(), 360.0);
        // sanity: we got angles for 2 splits
        assert_eq!(angs.len(), sb.nsplits() + 1);
        // angles must be in [0,360)
        for s in 1..=sb.nsplits() {
            assert!(angs[s] >= 0.0 && angs[s] < 360.0);
        }
    }

    #[test]
    fn intersect2_basic() {
        let ntax = 5;
        let mut sb = SplitsBlock::new();
        // P: {1,2} | {3,4,5}
        let p = sb.push(mk_split(&[1,2], ntax, 1.0));
        // Q: {2,3} | {1,4,5}
        let q = sb.push(mk_split(&[2,3], ntax, 1.0));

        // A(P) ∩ A(Q) = {2}
        let a_and_a = sb.intersect2(p, true, q, true);
        assert!(a_and_a.contains(2));
        assert_eq!(a_and_a.count_ones(..) - if a_and_a.contains(0) {1} else {0}, 1);

        // B(P) ∩ A(Q) = {3}
        let b_and_a = sb.intersect2(p, false, q, true);
        assert!(b_and_a.contains(3));
        assert_eq!(b_and_a.count_ones(..) - if b_and_a.contains(0) {1} else {0}, 1);
    }
}

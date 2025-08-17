use fixedbitset::FixedBitSet;

use crate::splits::bipartition::BiPartition;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

/// A split with weight/confidence/label, extending `BiPartition` semantics.
/// Indices are 1-based (bit 0 is ignored), just like the Java code.
#[derive(Debug, Clone)]
pub struct ASplit {
    pub base: BiPartition,
    pub weight: f64,
    pub confidence: f64,
    pub label: Option<String>,
}

impl ASplit {
    /* -------- constructors: A,B variants -------- */

    /// ASplit(A,B) with weight=1, confidence=-1, label=None
    pub fn new(a: FixedBitSet, b: FixedBitSet) -> Self {
        Self::new_full(a, b, 1.0, -1.0, None)
    }

    /// ASplit(A,B, weight)
    pub fn new_with_weight(a: FixedBitSet, b: FixedBitSet, weight: f64) -> Self {
        Self::new_full(a, b, weight, -1.0, None)
    }

    /// ASplit(A,B, weight, confidence)
    pub fn new_with_weight_conf(
        a: FixedBitSet,
        b: FixedBitSet,
        weight: f64,
        confidence: f64,
    ) -> Self {
        Self::new_full(a, b, weight, confidence, None)
    }

    /// ASplit(A,B, weight, confidence, label)
    pub fn new_full(
        a: FixedBitSet,
        b: FixedBitSet,
        weight: f64,
        confidence: f64,
        label: Option<String>,
    ) -> Self {
        let base = BiPartition::new(a, b);
        Self {
            base,
            weight,
            confidence,
            label,
        }
    }

    /* -------- constructors: A, ntax variants (B is complement of A in 1..=ntax) -------- */

    pub fn from_a_ntax(a: FixedBitSet, ntax: usize) -> Self {
        Self::from_a_ntax_full(a, ntax, 1.0, -1.0, None)
    }

    pub fn from_a_ntax_with_weight(a: FixedBitSet, ntax: usize, weight: f64) -> Self {
        Self::from_a_ntax_full(a, ntax, weight, -1.0, None)
    }

    pub fn from_a_ntax_with_weight_conf(
        a: FixedBitSet,
        ntax: usize,
        weight: f64,
        confidence: f64,
    ) -> Self {
        Self::from_a_ntax_full(a, ntax, weight, confidence, None)
    }

    pub fn from_a_ntax_full(
        a: FixedBitSet,
        ntax: usize,
        weight: f64,
        confidence: f64,
        label: Option<String>,
    ) -> Self {
        let b = complement_1_based(&a, ntax);
        Self::new_full(a, b, weight, confidence, label)
    }

    /// Copy constructor (like Java).
    pub fn from_other(src: &ASplit) -> Self {
        Self::new_full(
            src.get_a().clone(),
            src.get_b().clone(),
            src.weight,
            src.confidence,
            src.label.clone(),
        )
    }

    /* -------- getters / setters -------- */

    pub fn get_weight(&self) -> f64 {
        self.weight
    }
    pub fn set_weight(&mut self, w: f64) {
        self.weight = w;
    }

    pub fn get_confidence(&self) -> f64 {
        self.confidence
    }
    pub fn set_confidence(&mut self, c: f64) {
        self.confidence = c;
    }

    pub fn get_label(&self) -> Option<&str> {
        self.label.as_deref()
    }
    pub fn set_label<S: Into<String>>(&mut self, s: S) {
        self.label = Some(s.into());
    }

    /* -------- BiPartition delegation -------- */

    pub fn get_a(&self) -> &FixedBitSet {
        self.base.get_a()
    }
    pub fn get_b(&self) -> &FixedBitSet {
        self.base.get_b()
    }
    pub fn ntax(&self) -> usize {
        self.base.ntax()
    }
    pub fn size(&self) -> usize {
        self.base.size()
    }
    pub fn part_containing(&self, t: usize) -> &FixedBitSet {
        self.base.part_containing(t)
    }
    pub fn part_not_containing(&self, t: usize) -> &FixedBitSet {
        self.base.part_not_containing(t)
    }
    pub fn smaller_part(&self) -> &FixedBitSet {
        self.base.smaller_part()
    }
    pub fn is_trivial(&self) -> bool {
        self.base.is_trivial()
    }
    pub fn separates(&self, a: usize, b: usize) -> bool {
        self.base.separates(a, b)
    }

    /// Union of A and B (i.e., the full taxon set present in this split).
    pub fn get_all_taxa(&self) -> FixedBitSet {
        union_1_based(self.get_a(), self.get_b())
    }

    /// Access the underlying `BiPartition`.
    pub fn base(&self) -> &BiPartition {
        &self.base
    }
}

/* -------- Equality / Hash / Display -------- */

impl PartialEq for ASplit {
    /// Matches Java: equality depends only on the partition (ignores weight, confidence, label).
    /// Comparison is *anchored at taxon 1*: compare the part containing 1 and the part not containing 1.
    fn eq(&self, other: &Self) -> bool {
        same_bits_1based(self.part_containing(1), other.part_containing(1))
            && same_bits_1based(self.part_not_containing(1), other.part_not_containing(1))
    }
}
impl Eq for ASplit {}

impl Hash for ASplit {
    /// Hash only the partition (same as Java behavior via BiPartition#hashCode).
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.base.hash(state);
    }
}

impl Display for ASplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} weight={} confidence={} label={}",
            self.base,
            self.weight,
            self.confidence,
            self.label.as_deref().unwrap_or("null")
        )
    }
}

/* -------- helpers (BitSet-like) -------- */

fn same_bits_1based(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    // Compare membership for indices >=1 (ignore bit 0).
    // Iterate over the union of ones for speed.
    let mut seen = FixedBitSet::with_capacity(a.len().max(b.len()));
    seen.union_with(a);
    seen.union_with(b);
    for i in seen.ones() {
        if i == 0 {
            continue;
        }
        let ai = a.contains(i);
        let bi = b.contains(i);
        if ai != bi {
            return false;
        }
    }
    true
}

fn complement_1_based(a: &FixedBitSet, ntax: usize) -> FixedBitSet {
    let mut out = FixedBitSet::with_capacity(ntax + 1);
    out.grow(ntax + 1);
    for i in 1..=ntax {
        if !a.contains(i) {
            out.set(i, true);
        }
    }
    out
}

fn union_1_based(a: &FixedBitSet, b: &FixedBitSet) -> FixedBitSet {
    let mut out = FixedBitSet::with_capacity(a.len().max(b.len()));
    out.grow(a.len().max(b.len()));
    out.union_with(a);
    out.union_with(b);
    out
}

/* ---------------- Minimal adapter over your split type ---------------- */

/// If your ASplit stores weight/confidence differently, adapt here.
/// We assume ASplit provides:
///  - fn weight(&self) -> f64
///  - fn size(&self) -> usize
///  - fn smaller_part(&self) -> &FixedBitSet
pub trait ASplitView {
    fn weight(&self) -> f64;
    fn size(&self) -> usize;
    fn smaller_part(&self) -> &fixedbitset::FixedBitSet;
    fn ntax(&self) -> usize;
}

impl ASplitView for ASplit {
    fn weight(&self) -> f64 { self.weight }
    fn size(&self) -> usize { self.size() }
    fn smaller_part(&self) -> &fixedbitset::FixedBitSet { self.smaller_part() }
    fn ntax(&self) -> usize { self.ntax() }
}

/* -------- tests -------- */

#[cfg(test)]
mod tests {
    use super::*;
    use fixedbitset::FixedBitSet;

    fn bs_from(indices: &[usize], len: usize) -> FixedBitSet {
        let mut bs = FixedBitSet::with_capacity(len + 1);
        bs.grow(len + 1);
        for &i in indices {
            bs.set(i, true);
        }
        bs
    }

    #[test]
    fn constructors_and_complement() {
        let a = bs_from(&[2, 4], 5);
        let s = ASplit::from_a_ntax(a.clone(), 5);
        let all = s.get_all_taxa();
        for i in 1..=5 {
            assert!(all.contains(i));
        }
        // new_full
        let _s2 = ASplit::new_full(
            a.clone(),
            bs_from(&[1, 3, 5], 5),
            2.5,
            0.7,
            Some("X".into()),
        );
    }

    #[test]
    fn equals_anchored_on_one() {
        // Split {1,3}|{2,4,5} vs {2,4,5}|{1,3} — equal under ASplit semantics
        let s1 = ASplit::from_a_ntax(bs_from(&[1, 3], 5), 5);
        let s2 = ASplit::new(bs_from(&[2, 4, 5], 5), bs_from(&[1, 3], 5));
        assert_eq!(s1, s2);

        // Different partition:
        let s3 = ASplit::from_a_ntax(bs_from(&[1, 4], 5), 5);
        assert_ne!(s1, s3);
    }

    #[test]
    fn display_shape() {
        let mut s = ASplit::from_a_ntax(bs_from(&[2, 4], 5), 5);
        s.set_weight(3.14);
        s.set_confidence(0.9);
        s.set_label("my-split");
        let s = s.to_string();
        assert!(s.contains("weight=") && s.contains("confidence=") && s.contains("label="));
    }
}

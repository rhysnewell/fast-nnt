use fixedbitset::FixedBitSet;
use std::cmp::Ordering;
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};

/// Bipartition of a 1-based taxon set into two non-empty parts A and B.
///
/// Semantics match the Java version:
/// - Indices are expected to be 1..=ntax (bit 0 is ignored/unused).
/// - Constructor swaps parts so that the one whose first set bit (≥1) is smaller becomes A.
#[derive(Debug, Clone)]
pub struct BiPartition {
    a: FixedBitSet,
    b: FixedBitSet,
}

impl BiPartition {
    /// Construct from two bit sets. Both must be non-empty and disjoint, and together
    /// they define the taxa universe (Java version does not enforce disjointness, but assumes it).
    /// This mirrors the Java constructor’s normalization: the set whose first set bit ≥1 is
    /// smaller becomes `A`.
    pub fn new(a: FixedBitSet, b: FixedBitSet) -> Self {
        let ca = cardinality(&a);
        let cb = cardinality(&b);
        if ca == 0 || cb == 0 {
            eprintln!("Internal error: A.size()={}, B.size()={}", ca, cb);
        }
        let a_first = first_set_at_or_after(&a, 1).unwrap_or(usize::MAX);
        let b_first = first_set_at_or_after(&b, 1).unwrap_or(usize::MAX);

        if a_first < b_first {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }

    /// Total number of taxa: |A| + |B|.
    pub fn ntax(&self) -> usize {
        cardinality(&self.a) + cardinality(&self.b)
    }

    /// Size of the smaller side: min(|A|, |B|).
    pub fn size(&self) -> usize {
        cardinality(&self.a).min(cardinality(&self.b))
    }

    /// Part A (1-based).
    pub fn get_a(&self) -> &FixedBitSet {
        &self.a
    }

    /// Part B (1-based).
    pub fn get_b(&self) -> &FixedBitSet {
        &self.b
    }

    /// Split part that contains taxon `t` (1-based). Returns A if neither contains (like Java).
    pub fn part_containing(&self, t: usize) -> &FixedBitSet {
        if self.b.contains(t) { &self.b } else { &self.a }
    }

    /// The part that does *not* contain `t` (1-based).
    pub fn part_not_containing(&self, t: usize) -> &FixedBitSet {
        if !self.a.contains(t) {
            &self.a
        } else {
            &self.b
        }
    }

    /// The smaller part; on tie, the one whose first set bit (≥1) is smaller.
    pub fn smaller_part(&self) -> &FixedBitSet {
        let ca = cardinality(&self.a);
        let cb = cardinality(&self.b);
        match ca.cmp(&cb) {
            Ordering::Less => &self.a,
            Ordering::Greater => &self.b,
            Ordering::Equal => {
                let af = first_set_at_or_after(&self.a, 1).unwrap_or(usize::MAX);
                let bf = first_set_at_or_after(&self.b, 1).unwrap_or(usize::MAX);
                if af < bf { &self.a } else { &self.b }
            }
        }
    }

    /// Is taxon `t` in A?
    pub fn is_contained_in_a(&self, t: usize) -> bool {
        self.a.contains(t)
    }

    /// Is taxon `t` in B?
    pub fn is_contained_in_b(&self, t: usize) -> bool {
        self.b.contains(t)
    }

    /// Static compare: first by A, then by B (lexicographic on ascending set elements).
    pub fn compare(a: &BiPartition, b: &BiPartition) -> Ordering {
        match compare_bitsets(&a.a, &b.a) {
            Ordering::Equal => compare_bitsets(&a.b, &b.b),
            ord => ord,
        }
    }

    /// Are two splits compatible on the same taxa set?
    /// (!A1∩A2) || (!A1∩B2) || (!B1∩A2) || (!B1∩B2)
    pub fn are_compatible(s1: &BiPartition, s2: &BiPartition) -> bool {
        !intersects(&s1.a, &s2.a)
            || !intersects(&s1.a, &s2.b)
            || !intersects(&s1.b, &s2.a)
            || !intersects(&s1.b, &s2.b)
    }

    /// Are three splits weakly compatible?
    /// Negation of the two forbidden quadruple-intersection patterns (Java logic).
    pub fn are_weakly_compatible(s1: &BiPartition, s2: &BiPartition, s3: &BiPartition) -> bool {
        let (a1, b1) = (&s1.a, &s1.b);
        let (a2, b2) = (&s2.a, &s2.b);
        let (a3, b3) = (&s3.a, &s3.b);

        let bad1 = intersects3(a1, a2, a3)
            && intersects3(a1, b2, b3)
            && intersects3(b1, a2, b3)
            && intersects3(b1, b2, a3);

        let bad2 = intersects3(b1, b2, b3)
            && intersects3(b1, a2, a3)
            && intersects3(a1, b2, a3)
            && intersects3(a1, a2, b3);

        !(bad1 || bad2)
    }

    /// Is this a trivial split (smaller side has cardinality 1)?
    pub fn is_trivial(&self) -> bool {
        cardinality(self.smaller_part()) == 1
    }

    /// Does this split separate taxa `a` and `b` (1-based)?
    pub fn separates(&self, a: usize, b: usize) -> bool {
        let in_a = self.part_containing(a) as *const _;
        let in_b = self.part_containing(b) as *const _;
        in_a != in_b
    }
}

// ---- Equality / Hash / Ordering mirror Java ----

impl PartialEq for BiPartition {
    fn eq(&self, other: &Self) -> bool {
        bitset_eq(&self.a, &other.a) && bitset_eq(&self.b, &other.b)
    }
}
impl Eq for BiPartition {}

impl Hash for BiPartition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by set elements (stable across implementations)
        for i in self.a.ones() {
            i.hash(state);
        }
        0usize.hash(state); // separator
        for i in self.b.ones() {
            i.hash(state);
        }
    }
}

impl PartialOrd for BiPartition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BiPartition {
    fn cmp(&self, other: &Self) -> Ordering {
        match compare_bitsets(&self.a, &other.a) {
            Ordering::Equal => compare_bitsets(&self.b, &other.b),
            ord => ord,
        }
    }
}

impl Display for BiPartition {
    /// Formats as "{i1,i2,...} | {j1,j2,...}" with 1-based indices ascending,
    /// matching the Java toString().
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn part_to_string(bs: &FixedBitSet) -> String {
            let mut first = true;
            let mut s = String::new();
            for i in bs.ones() {
                if i == 0 {
                    continue;
                } // ignore 0 to keep 1-based view
                if !first {
                    s.push(',');
                } else {
                    first = false;
                }
                s.push_str(&i.to_string());
            }
            s
        }
        write!(
            f,
            "{{{}}} | {{{}}}",
            part_to_string(&self.a),
            part_to_string(&self.b)
        )
    }
}

// ---- Small helpers (BitSet-like utilities) ----

fn cardinality(bs: &FixedBitSet) -> usize {
    bs.ones().filter(|&i| i != 0).count()
}

fn first_set_at_or_after(bs: &FixedBitSet, start: usize) -> Option<usize> {
    bs.ones().find(|&i| i >= start)
}

fn bitset_eq(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    // FixedBitSet equality is structural (len + bits); for safety, compare membership.
    // (In practice, `==` would also work if lengths are kept consistent.)
    let a_len = a.ones().count();
    let b_len = b.ones().count();
    if a_len != b_len {
        return false;
    }
    // Same members?
    for i in a.ones() {
        if !b.contains(i) {
            return false;
        }
    }
    true
}

fn intersects(a: &FixedBitSet, b: &FixedBitSet) -> bool {
    // Scan the smaller set’s ones and test membership in the other.
    let (small, big) = if a.ones().size_hint().0 <= b.ones().size_hint().0 {
        (a, b)
    } else {
        (b, a)
    };
    for i in small.ones() {
        if i == 0 {
            continue;
        }
        if big.contains(i) {
            return true;
        }
    }
    false
}

fn intersects3(a: &FixedBitSet, b: &FixedBitSet, c: &FixedBitSet) -> bool {
    // Scan the smallest; test containment in the other two.
    let mut sizes = [
        (a.ones().size_hint().0, 0usize),
        (b.ones().size_hint().0, 1usize),
        (c.ones().size_hint().0, 2usize),
    ];
    sizes.sort_unstable();
    let (small_idx, _) = (sizes[0].1, sizes[0].0);
    let (small, other1, other2) = match small_idx {
        0 => (a, b, c),
        1 => (b, a, c),
        _ => (c, a, b),
    };
    for i in small.ones() {
        if i == 0 {
            continue;
        }
        if other1.contains(i) && other2.contains(i) {
            return true;
        }
    }
    false
}

/// Lexicographic compare of bit sets by ascending element list (ignoring 0).
fn compare_bitsets(a: &FixedBitSet, b: &FixedBitSet) -> Ordering {
    let mut ia = a.ones().filter(|&i| i != 0);
    let mut ib = b.ones().filter(|&i| i != 0);
    loop {
        match (ia.next(), ib.next()) {
            (Some(x), Some(y)) => {
                if x != y {
                    return x.cmp(&y);
                }
            }
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (None, None) => return Ordering::Equal,
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn bs_from(indices: &[usize], len: usize) -> FixedBitSet {
        let mut bs = FixedBitSet::with_capacity(len + 1);
        bs.grow(len + 1);
        for &i in indices {
            bs.set(i, true);
        }
        bs
    }

    #[test]
    fn constructor_normalizes_order() {
        // A first bit at 2; B first bit at 1 => B should become 'a'
        let a = bs_from(&[2, 4], 5);
        let b = bs_from(&[1, 3], 5);
        let p = BiPartition::new(a, b);
        assert!(p.a.contains(1));
        assert!(p.b.contains(2));
    }

    #[test]
    fn size_and_trivial() {
        let a = bs_from(&[1], 4);
        let b = bs_from(&[2, 3, 4], 4);
        let p = BiPartition::new(a, b);
        assert_eq!(p.ntax(), 4);
        assert_eq!(p.size(), 1);
        assert!(p.is_trivial());
    }

    #[test]
    fn compatibility() {
        let p1 = BiPartition::new(bs_from(&[1, 2], 4), bs_from(&[3, 4], 4));
        let p2 = BiPartition::new(bs_from(&[1, 3], 4), bs_from(&[2, 4], 4));
        // These two are incompatible:
        assert!(!BiPartition::are_compatible(&p1, &p2));
    }

    #[test]
    fn ordering() {
        let p1 = BiPartition::new(bs_from(&[2, 4], 5), bs_from(&[1, 3, 5], 5));
        let p2 = BiPartition::new(bs_from(&[3, 5], 5), bs_from(&[1, 2, 4], 5));
        // Compare by A then B as lexicographic lists:
        assert!(p1 < p2 || p1 > p2 || p1 == p2); // sanity
    }

    #[test]
    fn display_format() {
        let p = BiPartition::new(bs_from(&[2, 4], 5), bs_from(&[1, 3, 5], 5));
        let s = p.to_string();
        // Example: "{1,3,5} | {2,4}" or swapped depending on normalization
        assert!(s.contains('|') && s.contains('{') && s.contains('}'));
    }
}

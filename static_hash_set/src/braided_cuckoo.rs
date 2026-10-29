//! Static, cache-line bucketed braided cuckoo hashing.
//!
//! Five power-of-two layers have relative sizes 16:8:4:2:1. Capacity
//! is rounded up to leave at least 10% empty slots. The paper's cumulative
//! primary masses (20, 26, 29, 30.5) are normalized to probabilities,
//! giving weights 40:12:6:3; the last layer receives only secondary keys.
//! With eight-slot buckets placement is not guaranteed: construction retries
//! fresh hashes, then reports failure rather than growing or losing keys.
use std::hash::{BuildHasher, BuildHasherDefault};

pub use crate::cuckoo::Mode;
use crate::{BIN_SIZE, Bin, HashSet, S, T};

type Hasher = BuildHasherDefault<gxhash::GxHasher>;
const LAYERS: usize = 5;
const ATTEMPTS: u64 = 64;

pub struct BraidedCuckooSet<const MODE: u8> {
    table: Box<[Bin]>,
    offsets: [usize; LAYERS],
    smallest_bits: u32,
    seed: u64,
    len: usize,
    has_zero: bool,
}

impl<const MODE: u8> BraidedCuckooSet<MODE> {
    pub fn new(keys: &[T]) -> Self {
        Self::try_new(keys).expect("braided cuckoo placement failed after 64 attempts")
    }

    pub fn try_new(keys: &[T]) -> Option<Self> {
        let _ = Mode::from(MODE);
        let mut keys = keys.to_vec();
        keys.sort_unstable();
        keys.dedup();
        let has_zero = keys.first() == Some(&0);
        let len = keys.len();
        // Eight address bits in the smallest layer are required by the
        // fourth braid (bit positions 7 and 8, counted from the MSB).
        let smallest = len
            .checked_mul(10)?
            .div_ceil(9 * BIN_SIZE * 31)
            .max(256)
            .checked_next_power_of_two()?;
        let num_bins = smallest.checked_mul(31)?;
        let offsets = std::array::from_fn(|i| smallest * (32 - (32 >> i)));
        let mut this = Self {
            table: vec![Bin([0; BIN_SIZE]); num_bins].into_boxed_slice(),
            offsets,
            smallest_bits: smallest.ilog2(),
            seed: 0,
            len,
            has_zero,
        };
        for attempt in 0..ATTEMPTS {
            this.seed = attempt.wrapping_mul(0x9e3779b97f4a7c15);
            this.table.fill(Bin([0; BIN_SIZE]));
            if this.place(&keys) {
                return Some(this);
            }
        }
        None
    }

    // Incoming secondary keys are placed before a layer's primary keys.
    // Only primary keys may be forwarded, and their choice does not depend
    // on the secondary hash. No primary bin is changed after its phase.
    fn place(&mut self, keys: &[T]) -> bool {
        let mut primary: [Vec<(T, usize)>; LAYERS - 1] = std::array::from_fn(|_| Vec::new());
        for &key in keys.iter().filter(|&&key| key != 0) {
            let (layer, idx) = self.primary(key);
            primary[layer].push((key, idx));
        }
        for (layer, keys) in primary.iter().enumerate() {
            for &(key, idx) in keys {
                let idx = if self.table[idx].has_zero() {
                    idx
                } else {
                    self.secondary(key, layer, idx)
                };
                if !self.table[idx].has_zero() {
                    return false;
                }
                self.table[idx].insert(key);
            }
        }
        true
    }

    #[inline(always)]
    fn hash(&self, key: T, domain: u64) -> u64 {
        Hasher::default().hash_one((key, self.seed, domain))
    }

    #[inline(always)]
    fn primary(&self, key: T) -> (usize, usize) {
        let draw = ((self.hash(key, 0) as u128 * 61) >> 64) as u64;
        let layer = if draw < 40 {
            0
        } else if draw < 52 {
            1
        } else if draw < 58 {
            2
        } else {
            3
        };
        let bits = self.smallest_bits + 4 - layer as u32;
        let local = self.hash(key, 1) as usize & ((1usize << bits) - 1);
        (layer, self.offsets[layer] + local)
    }

    #[inline(always)]
    fn secondary(&self, key: T, layer: usize, primary: usize) -> usize {
        let local = (primary - self.offsets[layer]) >> 1;
        let remaining_bits = self.smallest_bits + 3 - layer as u32;
        // Delete the LSB, then XOR a nonzero two-bit mask into positions
        // 2i-1 and 2i in the paper's one-based, MSB-first notation.
        let shift = remaining_bits - 2 * (layer as u32 + 1);
        let change = 1 + ((self.hash(key, 2) as u128 * 3) >> 64) as usize;
        self.offsets[layer + 1] + (local ^ (change << shift))
    }

    #[inline(always)]
    fn pair(&self, key: T) -> (usize, usize) {
        let (layer, first) = self.primary(key);
        (first, self.secondary(key, layer, first))
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table)
    }

    #[inline(always)]
    pub fn contains_lazy(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let (layer, first) = self.primary(key);
        let bin = &self.table[first];
        let keys = S::splat(key as _);
        if bin.contains(keys) {
            return true;
        }
        if bin.has_zero() {
            return false;
        }
        self.table[self.secondary(key, layer, first)].contains(keys)
    }

    #[inline(always)]
    pub fn contains_eager(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let (first, second) = self.pair(key);
        let keys = S::splat(key as _);
        self.table[first].contains(keys) | self.table[second].contains(keys)
    }

    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        match Mode::from(MODE) {
            Mode::Lazy => self.contains_lazy(key),
            Mode::Eager => self.contains_eager(key),
        }
    }

    /// Tokens are primary bucket indices; eager lookups derive the braid
    /// from the key, avoiding a 32-bit limit on packed bucket indices.
    #[inline(always)]
    pub fn prefetch(&self, key: T) -> usize {
        let (layer, first) = self.primary(key);
        prefetch_index::prefetch_index(&self.table, first);
        if Mode::from(MODE) == Mode::Eager {
            prefetch_index::prefetch_index(&self.table, self.secondary(key, layer, first));
        }
        first
    }

    #[inline(always)]
    pub fn contains_with_token(&self, key: T, first: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let keys = S::splat(key as _);
        let bin = &self.table[first];
        match Mode::from(MODE) {
            Mode::Lazy if bin.contains(keys) => true,
            Mode::Lazy if bin.has_zero() => false,
            mode => {
                let layer = self.offsets.partition_point(|&offset| offset <= first) - 1;
                let found = self.table[self.secondary(key, layer, first)].contains(keys);
                if mode == Mode::Eager {
                    bin.contains(keys) | found
                } else {
                    found
                }
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        std::iter::repeat_n(0, self.has_zero as usize).chain(
            self.table
                .iter()
                .flat_map(|bin| bin.0)
                .filter(|&key| key != 0),
        )
    }

    pub fn test(&self) {
        assert_eq!(self.iter().count(), self.len());
        for key in self.iter() {
            assert!(self.contains(key));
        }
    }
}

impl<const MODE: u8> HashSet for BraidedCuckooSet<MODE> {
    fn name(&self) -> &'static str {
        match Mode::from(MODE) {
            Mode::Lazy => "BraidedCuckooSet<Lazy>",
            Mode::Eager => "BraidedCuckooSet<Eager>",
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(Self::new(keys))
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len as f32 / (self.table.len() * BIN_SIZE) as f32
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    fn prefetch(&self, key: T) -> usize {
        self.prefetch(key)
    }
    fn contains(&self, key: T) -> bool {
        self.contains(key)
    }
    fn contains_with_token(&self, key: T, token: usize) -> bool {
        self.contains_with_token(key, token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<const MODE: u8>(keys: &[T]) {
        let set = BraidedCuckooSet::<MODE>::new(keys);
        let expected: std::collections::HashSet<_> = keys.iter().copied().collect();
        assert_eq!(set.len(), expected.len());
        assert_eq!(
            set.iter().collect::<std::collections::HashSet<_>>(),
            expected
        );
        assert!(set.load_factor() <= 0.9);
        for key in keys.iter().copied().chain(1_000_000..1_010_000) {
            let found = expected.contains(&key);
            assert_eq!(set.contains_lazy(key), found);
            assert_eq!(set.contains_eager(key), found);
            assert_eq!(set.contains_with_token(key, set.prefetch(key)), found);
        }
        for layer in 0..4 {
            let size = set.offsets[layer + 1] - set.offsets[layer];
            assert!(size.is_power_of_two());
            assert_eq!(size, (set.table.len() - set.offsets[4]) << (4 - layer));
        }
        for key in keys.iter().copied().filter(|&key| key != 0) {
            let (layer, first) = set.primary(key);
            let second = set.secondary(key, layer, first);
            let before = (first - set.offsets[layer]) >> 1;
            let after = second - set.offsets[layer + 1];
            let shift = set.smallest_bits + 3 - layer as u32 - 2 * (layer as u32 + 1);
            assert!((1..=3).contains(&((before ^ after) >> shift)));
            assert_eq!((before ^ after) & !(3 << shift), 0);
            assert!(
                second
                    < set
                        .offsets
                        .get(layer + 2)
                        .copied()
                        .unwrap_or(set.table.len())
            );
        }
        set.test();
    }

    #[test]
    fn overloaded_secondary_reports_failure() {
        let mut set = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::new(&[]);
        let mut groups = std::collections::HashMap::<_, Vec<T>>::new();
        let colliding = (1..)
            .find_map(|key| {
                let pair = set.pair(key);
                let group = groups.entry(pair).or_default();
                group.push(key);
                (group.len() == 2 * BIN_SIZE + 1).then(|| group.clone())
            })
            .unwrap();
        assert!(!set.place(&colliding));
    }

    #[test]
    fn membership_and_braid() {
        for keys in [
            vec![],
            vec![0, 0, 1, 1, u64::MAX],
            (0..20_000).collect(),
            (0..25_000u64)
                .map(|x| x.wrapping_mul(0x9e3779b97f4a7c15))
                .collect(),
        ] {
            check::<{ Mode::Lazy as u8 }>(&keys);
            check::<{ Mode::Eager as u8 }>(&keys);
        }
    }
}

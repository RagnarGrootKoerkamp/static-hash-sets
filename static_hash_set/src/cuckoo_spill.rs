//! A dense_hash_set for u64 keys.
//!
//! Compared to std::collections::HashSet<u64>, this uses a different layout: no metadata table, just plain data.
//! This is similar to Google's dense_hash_map, which predates the SwissTable design. By avoiding a metadata table,
//! we may need to do longer probe sequences (each probe is 8 bytes, not 1 byte), but on the other hand we only take
//! 1 cache miss per access, not 2.

use std::hash::{BuildHasher, BuildHasherDefault};

use super::BIN_SIZE;
use crate::Bin;
use crate::S;
use crate::T;
use crate::traits::HashSet;

type Hasher = BuildHasherDefault<gxhash::GxHasher>;

#[derive(PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Mode {
    Eager = 0,
    Lazy = 1,
}

impl Mode {
    pub const fn from(value: u8) -> Self {
        match value {
            0 => Mode::Eager,
            1 => Mode::Lazy,
            _ => panic!(),
        }
    }
}

enum SpillWords {
    U8(Box<[u8]>),
    U16(Box<[u16]>),
}

impl SpillWords {
    fn new(width: usize, len: usize) -> Self {
        match width {
            8 => Self::U8(vec![0; len].into_boxed_slice()),
            16 => Self::U16(vec![0; len].into_boxed_slice()),
            _ => panic!("spill words must be 8 or 16 bits"),
        }
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> bool {
        match self {
            Self::U8(words) => words[idx] & 1 != 0,
            Self::U16(words) => words[idx] & 1 != 0,
        }
    }
    fn set(&mut self, idx: usize) {
        match self {
            Self::U8(words) => words[idx] |= 1,
            Self::U16(words) => words[idx] |= 1,
        }
    }
    fn fill(&mut self, value: u16) {
        match self {
            Self::U8(words) => words.fill(value as u8),
            Self::U16(words) => words.fill(value),
        }
    }
    fn allocation_size(&self) -> usize {
        match self {
            Self::U8(words) => std::mem::size_of_val(&**words),
            Self::U16(words) => std::mem::size_of_val(&**words),
        }
    }
}

/// Cuckoo set with one separate spill bit per primary bucket.
/// `BEFORE_PRIMARY` reads that bit before searching the primary bucket;
/// otherwise it reads it only after a primary miss.
pub struct SpillCuckooSet<const MODE: u8, const BEFORE_PRIMARY: bool, const WORD_BITS: usize = 8> {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[Bin]>,
    len: usize,
    has_zero: bool,
    spill: SpillWords,
    summary_ready: bool,
}

impl<const MODE: u8, const BEFORE_PRIMARY: bool, const WORD_BITS: usize>
    SpillCuckooSet<MODE, BEFORE_PRIMARY, WORD_BITS>
{
    pub fn new(slot_ratio: f32, keys: &[T]) -> Self {
        let mut this = Self::with_capacity(slot_ratio, keys.len());
        for &k in keys {
            this.insert(k);
        }
        this.rebuild_spill();
        this.summary_ready = true;
        this
    }
    fn with_capacity(slot_ratio: f32, n: usize) -> Self {
        assert!(slot_ratio >= 1.0);
        let capacity = (n as f32 * slot_ratio).ceil() as usize;
        let num_bins = capacity.div_ceil(BIN_SIZE);
        let table = vec![Bin([0 as T; BIN_SIZE]); num_bins].into_boxed_slice();
        let spill = SpillWords::new(WORD_BITS, num_bins);
        Self {
            slot_ratio,
            num_bins,
            table,
            len: 0,
            has_zero: false,
            spill,
            summary_ready: false,
        }
    }

    fn rebuild_spill(&mut self) {
        self.spill.fill(0);
        for bucket in 0..self.num_bins {
            for &key in &self.table[bucket].0 {
                if key != 0 && self.bin_idx_1(key) != bucket {
                    self.spill.set(self.bin_idx_1(key));
                }
            }
        }
    }

    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table) + self.spill.allocation_size()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len + self.has_zero as usize
    }

    #[inline(always)]
    fn bin_idx_1(&self, key: T) -> usize {
        let hash64 = Hasher::default().hash_one(key);
        (((hash64 as u128) * (self.num_bins as u128)) >> 64) as usize
    }

    #[inline(always)]
    fn bin_idx_2(&self, key: T) -> usize {
        const XOR: u64 = 0x9e3779b97f4a7c15;
        let hash64 = Hasher::default().hash_one(key ^ XOR);
        (((hash64 as u128) * (self.num_bins as u128)) >> 64) as usize
    }

    #[inline(always)]
    fn get_bin(&self, idx: usize) -> &Bin {
        unsafe { self.table.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn prefetch_first(&self, key: T) -> usize {
        let bin_idx = self.bin_idx_1(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
        if BEFORE_PRIMARY {
            match &self.spill {
                SpillWords::U8(words) => prefetch_index::prefetch_index(words, bin_idx),
                SpillWords::U16(words) => prefetch_index::prefetch_index(words, bin_idx),
            }
        }
        bin_idx
    }
    #[inline(always)]
    pub fn prefetch_second(&self, key: T) -> usize {
        let bin_idx = self.bin_idx_2(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
        bin_idx
    }
    #[inline(always)]
    pub fn prefetch_both(&self, key: T) -> usize {
        let b1 = self.prefetch_first(key);
        let b2 = self.prefetch_second(key);
        (b1 << 32) | (b2 as u32) as usize
    }
    #[inline(always)]
    pub fn prefetch(&self, key: T) -> usize {
        match Mode::from(MODE) {
            Mode::Eager => self.prefetch_both(key),
            Mode::Lazy => self.prefetch_first(key),
        }
    }

    #[inline(always)]
    pub fn contains_lazy(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let keys = S::splat(key as _);

        let bin_idx = self.bin_idx_1(key);
        let may_spill = if BEFORE_PRIMARY {
            self.spill.get(bin_idx)
        } else {
            false
        };
        let bin1 = self.get_bin(bin_idx);
        if bin1.contains(keys) {
            return true;
        }
        let may_spill = if BEFORE_PRIMARY {
            may_spill
        } else {
            self.spill.get(bin_idx)
        };
        if !may_spill {
            return false;
        }
        self.get_bin(self.bin_idx_2(key)).contains(keys)
    }

    #[inline(always)]
    pub fn contains_eager(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let keys = S::splat(key as _);

        self.get_bin(self.bin_idx_1(key)).contains(keys)
            | self.get_bin(self.bin_idx_2(key)).contains(keys)
    }

    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        match Mode::from(MODE) {
            Mode::Eager => self.contains_eager(key),
            Mode::Lazy => self.contains_lazy(key),
        }
    }

    #[inline(always)]
    pub fn contains_lazy_with_token(&self, key: T, bin1: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let keys = S::splat(key as _);

        let bin_idx = bin1;
        let may_spill = if BEFORE_PRIMARY {
            self.spill.get(bin_idx)
        } else {
            false
        };
        let bin = self.get_bin(bin_idx);
        if bin.contains(keys) {
            return true;
        }
        let may_spill = if BEFORE_PRIMARY {
            may_spill
        } else {
            self.spill.get(bin_idx)
        };
        if !may_spill {
            return false;
        }
        self.get_bin(self.bin_idx_2(key)).contains(keys)
    }

    #[inline(always)]
    pub fn contains_eager_with_token(&self, key: T, bin: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let keys = S::splat(key as _);

        let bin1 = bin >> 32;
        let bin2 = bin as u32 as usize;

        self.get_bin(bin1).contains(keys) | self.get_bin(bin2).contains(keys)
    }

    #[inline(always)]
    pub fn contains_with_token(&self, key: T, token: usize) -> bool {
        match Mode::from(MODE) {
            Mode::Eager => self.contains_eager_with_token(key, token),
            Mode::Lazy => self.contains_lazy_with_token(key, token),
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: T) {
        if self.summary_ready {
            self.spill.fill(1);
            self.summary_ready = false;
        }
        if self.contains_eager(key) {
            return;
        }
        if key == 0 {
            self.has_zero = true;
            self.len += 1;
            return;
        }

        for idx in [self.bin_idx_1(key), self.bin_idx_2(key)] {
            let bin = &mut self.table[idx];
            if bin.has_zero() {
                bin.insert(key);
                self.len += 1;
                return;
            }
        }
        // Start a displacement chain.
        self.insert_to_bin(key, self.bin_idx_1(key));
    }

    /// Internal version that forces the key into a specific option.
    #[inline(always)]
    fn insert_to_bin(&mut self, key: T, idx: usize) {
        let bin = &mut self.table[idx];
        if bin.has_zero() {
            bin.insert(key);
            self.len += 1;
            return;
        }
        let bump_idx = rand::random_range(0..BIN_SIZE);
        let key = std::mem::replace(&mut bin.0[bump_idx], key);
        let idx = self.bin_idx_1(key) ^ self.bin_idx_2(key) ^ idx;
        // Fingers crossed that this becomes a non-recursive tail call.
        self.insert_to_bin(key, idx);
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        std::iter::repeat_n(0, self.has_zero as usize).chain(
            self.table
                .iter()
                .flat_map(|b| b.0.iter().copied())
                .filter(|x| *x != 0),
        )
    }

    pub fn test(&self) {
        for x in self.iter() {
            assert!(self.contains_eager(x));
        }
    }
}

impl<const MODE: u8, const BEFORE_PRIMARY: bool, const WORD_BITS: usize> HashSet
    for SpillCuckooSet<MODE, BEFORE_PRIMARY, WORD_BITS>
{
    fn name(&self) -> &'static str {
        match Mode::from(MODE) {
            Mode::Eager => "SpillCuckooSet<Eager>",
            Mode::Lazy if BEFORE_PRIMARY && WORD_BITS == 8 => {
                "SpillCuckooSet<Lazy,BeforePrimary,u8>"
            }
            Mode::Lazy if !BEFORE_PRIMARY && WORD_BITS == 8 => "SpillCuckooSet<Lazy,AfterMiss,u8>",
            Mode::Lazy if BEFORE_PRIMARY => "SpillCuckooSet<Lazy,BeforePrimary,u16>",
            Mode::Lazy => "SpillCuckooSet<Lazy,AfterMiss,u16>",
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = SpillCuckooSet::<MODE, BEFORE_PRIMARY, WORD_BITS>::new(self.slot_ratio, keys);
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len() as f32 / (self.num_bins * BIN_SIZE) as f32
    }

    fn primary_hit_ratio(&self) -> Option<f32> {
        let mut total = 0usize;
        let mut primary = 0usize;
        for (idx, bin) in self.table.iter().enumerate() {
            for &key in &bin.0 {
                if key != 0 {
                    total += 1;
                    primary += (self.bin_idx_1(key) == idx) as usize;
                }
            }
        }
        (total != 0).then(|| primary as f32 / total as f32)
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    #[inline(always)]
    fn prefetch(&self, key: T) -> usize {
        self.prefetch(key)
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.contains(key)
    }
    #[inline(always)]
    fn contains_with_token(&self, key: T, token: usize) -> bool {
        self.contains_with_token(key, token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_ratio_matches_first_bucket_hits_and_excludes_zero() {
        let keys: Vec<_> = (0..20_000).collect();
        let set = SpillCuckooSet::<0, false>::new(1.05, &keys);
        let primary = set
            .iter()
            .filter(|&key| key != 0)
            .filter(|&key| set.get_bin(set.bin_idx_1(key)).contains(S::splat(key as _)))
            .count();
        assert_eq!(set.primary_hit_ratio(), Some(primary as f32 / 19_999.0));
        assert!(primary < 19_999);
        assert_eq!(
            SpillCuckooSet::<0, false>::new(1.1, &[]).primary_hit_ratio(),
            None
        );
        assert_eq!(
            SpillCuckooSet::<0, false>::new(1.1, &[0]).primary_hit_ratio(),
            None
        );
    }

    #[test]
    fn spill_metadata_and_both_lookup_orders_are_correct() {
        for &before in &[false, true] {
            if before {
                check::<true, 8>();
                check::<true, 16>();
            } else {
                check::<false, 8>();
                check::<false, 16>();
            }
        }
    }

    fn check<const BEFORE: bool, const WORD_BITS: usize>() {
        let keys: Vec<_> = (0..20_000).collect();
        let mut set = SpillCuckooSet::<{ Mode::Lazy as u8 }, BEFORE, WORD_BITS>::new(1.05, &keys);
        assert!(set.contains(0));
        assert!(!set.contains(u64::MAX));
        for key in 1..20_000 {
            assert!(set.contains(key));
            assert!(set.contains_with_token(key, set.prefetch_first(key)));
        }
        let mut expected_spill = vec![false; set.num_bins];
        for (stored_at, bin) in set.table.iter().enumerate() {
            for &key in &bin.0 {
                if key != 0 {
                    let primary = set.bin_idx_1(key);
                    if primary != stored_at {
                        expected_spill[primary] = true;
                    }
                }
            }
        }
        for (bucket, expected) in expected_spill.into_iter().enumerate() {
            assert_eq!(set.spill.get(bucket), expected);
        }
        set.insert(42_001);
        assert!(set.contains(42_001));
        assert!(set.contains_with_token(42_001, set.prefetch_first(42_001)));
        assert!(!set.contains(u64::MAX));
    }
}

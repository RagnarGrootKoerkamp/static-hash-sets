//! A dense_hash_set for u64 keys.
//!
//! Compared to std::collections::HashSet<u64>, this uses a different layout: no metadata table, just plain data.
//! This is similar to Google's dense_hash_map, which predates the SwissTable design. By avoiding a metadata table,
//! we may need to do longer probe sequences (each probe is 8 bytes, not 1 byte), but on the other hand we only take
//! 1 cache miss per access, not 2.

use std::hash::{BuildHasher, BuildHasherDefault};

const BIN_SIZE: usize = 7;
use crate::S;
use crate::T;
use crate::traits::HashSet;

#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Eq)]
struct FilteredBin {
    slots: [T; BIN_SIZE],
    spill: u64,
}

impl FilteredBin {
    #[inline(always)]
    fn contains(&self, keys: S) -> bool {
        let first: S = unsafe {
            std::mem::transmute([self.slots[0], self.slots[1], self.slots[2], self.slots[3]])
        };
        let tail: S =
            unsafe { std::mem::transmute([self.slots[4], self.slots[5], self.slots[6], 0]) };
        (first.simd_eq(keys) | tail.simd_eq(keys)).to_bitmask() > 0
    }
    #[inline(always)]
    fn has_zero(&self) -> bool {
        self.slots.contains(&0)
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.slots.iter().filter(|&&key| key != 0).count()
    }
    #[inline(always)]
    fn insert(&mut self, key: T) {
        let idx = self.len();
        assert!(idx < BIN_SIZE, "bucket is full");
        self.slots[idx] = key;
    }
}

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

pub struct InlineFilterCuckooSet<
    const MODE: u8,
    const ONE_HOT: bool,
    const INCLUDE_PRIMARY: bool = false,
    const BRANCHLESS_SECONDARY: bool = false,
> {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[FilteredBin]>,
    len: usize,
    has_zero: bool,
    summary_ready: bool,
}

impl<
        const MODE: u8,
        const ONE_HOT: bool,
        const INCLUDE_PRIMARY: bool,
        const BRANCHLESS_SECONDARY: bool,
    > InlineFilterCuckooSet<MODE, ONE_HOT, INCLUDE_PRIMARY, BRANCHLESS_SECONDARY>
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
        let table = vec![
            FilteredBin {
                slots: [0 as T; BIN_SIZE],
                spill: 0
            };
            num_bins
        ]
        .into_boxed_slice();
        Self {
            slot_ratio,
            num_bins,
            table,
            len: 0,
            has_zero: false,
            summary_ready: false,
        }
    }

    #[inline(always)]
    fn bloom_bits(key: T) -> u64 {
        let h1 = key.wrapping_mul(0x9e3779b97f4a7c15) ^ key.rotate_left(23);
        let h2 = key.wrapping_mul(0xbf58476d1ce4e5b9) ^ key.rotate_right(17);
        (1u64 << ((h1 ^ (h1 >> 32)) & 63)) | (1u64 << ((h2 ^ (h2 >> 29)) & 63))
    }

    #[inline(always)]
    fn one_hot_bit(key: T) -> u64 {
        let hash = key.wrapping_mul(0x9e3779b97f4a7c15) ^ key.rotate_left(23);
        1u64 << ((hash ^ (hash >> 32)) & 63)
    }

    #[inline(always)]
    fn add_summary(summary: &mut u64, key: T) {
        let bits = if ONE_HOT {
            Self::one_hot_bit(key)
        } else {
            Self::bloom_bits(key)
        };
        *summary |= bits;
    }

    #[inline(always)]
    fn might_spill(summary: u64, key: T) -> bool {
        let bits = if ONE_HOT {
            Self::one_hot_bit(key)
        } else {
            Self::bloom_bits(key)
        };
        summary & bits == bits
    }

    #[inline(always)]
    fn choose_secondary(primary: usize, secondary: usize, use_secondary: bool) -> usize {
        let mask = 0usize.wrapping_sub(use_secondary as usize);
        (primary & !mask) | (secondary & mask)
    }

    fn rebuild_spill(&mut self) {
        for bin in self.table.iter_mut() {
            bin.spill = 0;
        }
        for bucket in 0..self.num_bins {
            for slot in 0..BIN_SIZE {
                let key = self.table[bucket].slots[slot];
                if key != 0 {
                    let primary = self.bin_idx_1(key);
                    if INCLUDE_PRIMARY || primary != bucket {
                        Self::add_summary(&mut self.table[primary].spill, key);
                    }
                }
            }
        }
    }

    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table)
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
    fn get_bin(&self, idx: usize) -> &FilteredBin {
        unsafe { self.table.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn prefetch_first(&self, key: T) -> usize {
        let bin_idx = self.bin_idx_1(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
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
        let bin1 = self.get_bin(bin_idx);
        if BRANCHLESS_SECONDARY {
            let primary_match = bin1.contains(keys);
            let may_spill = Self::might_spill(bin1.spill, key);
            let bin2 = self.bin_idx_2(key);
            let secondary_idx = Self::choose_secondary(
                bin_idx,
                bin2,
                (!primary_match) & may_spill,
            );
            return primary_match | self.get_bin(secondary_idx).contains(keys);
        }
        if INCLUDE_PRIMARY && !Self::might_spill(bin1.spill, key) {
            return false;
        }
        if bin1.contains(keys) {
            return true;
        }
        if !INCLUDE_PRIMARY && !Self::might_spill(bin1.spill, key) {
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

        let primary_idx = bin1;
        let bin1 = self.get_bin(primary_idx);
        if BRANCHLESS_SECONDARY {
            let primary_match = bin1.contains(keys);
            let may_spill = Self::might_spill(bin1.spill, key);
            let bin2 = self.bin_idx_2(key);
            let secondary_idx = Self::choose_secondary(
                primary_idx,
                bin2,
                (!primary_match) & may_spill,
            );
            return primary_match | self.get_bin(secondary_idx).contains(keys);
        }
        if INCLUDE_PRIMARY && !Self::might_spill(bin1.spill, key) {
            return false;
        }
        if bin1.contains(keys) {
            return true;
        }
        if !INCLUDE_PRIMARY && !Self::might_spill(bin1.spill, key) {
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
            for bin in self.table.iter_mut() {
                bin.spill = u64::MAX;
            }
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
        let key = std::mem::replace(&mut bin.slots[bump_idx], key);
        let idx = self.bin_idx_1(key) ^ self.bin_idx_2(key) ^ idx;
        // Fingers crossed that this becomes a non-recursive tail call.
        self.insert_to_bin(key, idx);
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        std::iter::repeat_n(0, self.has_zero as usize).chain(
            self.table
                .iter()
                .flat_map(|b| b.slots.iter().copied())
                .filter(|x| *x != 0),
        )
    }

    pub fn test(&self) {
        for x in self.iter() {
            assert!(self.contains_eager(x));
        }
    }
}

impl<
        const MODE: u8,
        const ONE_HOT: bool,
        const INCLUDE_PRIMARY: bool,
        const BRANCHLESS_SECONDARY: bool,
    > HashSet for InlineFilterCuckooSet<MODE, ONE_HOT, INCLUDE_PRIMARY, BRANCHLESS_SECONDARY>
{
    fn name(&self) -> &'static str {
        if BRANCHLESS_SECONDARY {
            return match Mode::from(MODE) {
                Mode::Eager => "InlineOneHotCuckooSet<Eager,BumpedOnly,Branchless>",
                Mode::Lazy => "InlineOneHotCuckooSet<Lazy,BumpedOnly,Branchless>",
            };
        }
        match Mode::from(MODE) {
            Mode::Eager if ONE_HOT && INCLUDE_PRIMARY => "InlineOneHotCuckooSet<Eager,AllKeys>",
            Mode::Lazy if ONE_HOT && INCLUDE_PRIMARY => "InlineOneHotCuckooSet<Lazy,AllKeys>",
            Mode::Eager if ONE_HOT => "InlineOneHotCuckooSet<Eager,BumpedOnly>",
            Mode::Lazy if ONE_HOT => "InlineOneHotCuckooSet<Lazy,BumpedOnly>",
            Mode::Eager if INCLUDE_PRIMARY => "InlineTwoHotCuckooSet<Eager,AllKeys>",
            Mode::Lazy if INCLUDE_PRIMARY => "InlineTwoHotCuckooSet<Lazy,AllKeys>",
            Mode::Eager => "InlineTwoHotCuckooSet<Eager,BumpedOnly>",
            Mode::Lazy => "InlineTwoHotCuckooSet<Lazy,BumpedOnly>",
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = InlineFilterCuckooSet::<
            MODE,
            ONE_HOT,
            INCLUDE_PRIMARY,
            BRANCHLESS_SECONDARY,
        >::new(self.slot_ratio, keys);
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
            for &key in &bin.slots {
                if key != 0 {
                    total += 1;
                    primary += (self.bin_idx_1(key) == idx) as usize;
                }
            }
        }
        (total != 0).then(|| primary as f32 / total as f32)
    }
    fn max_primary_bucket_size(&self) -> Option<usize> {
        let mut counts = vec![0usize; self.num_bins];
        for bin in &self.table {
            for &key in &bin.slots {
                if key != 0 {
                    counts[self.bin_idx_1(key)] += 1;
                }
            }
        }
        counts.into_iter().max()
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
    fn inline_filter_bin_keeps_one_cache_line() {
        assert_eq!(std::mem::size_of::<FilteredBin>(), 64);
        assert_eq!(std::mem::align_of::<FilteredBin>(), 64);
    }

    #[test]
    fn primary_ratio_matches_first_bucket_hits_and_excludes_zero() {
        let keys: Vec<_> = (0..20_000).collect();
        let set = InlineFilterCuckooSet::<0, false>::new(1.05, &keys);
        let primary = set
            .iter()
            .filter(|&key| key != 0)
            .filter(|&key| set.get_bin(set.bin_idx_1(key)).contains(S::splat(key as _)))
            .count();
        assert_eq!(set.primary_hit_ratio(), Some(primary as f32 / 19_999.0));
        assert!(primary < 19_999);
        assert_eq!(
            InlineFilterCuckooSet::<0, false>::new(1.1, &[]).primary_hit_ratio(),
            None
        );
        assert_eq!(
            InlineFilterCuckooSet::<0, false>::new(1.1, &[0]).primary_hit_ratio(),
            None
        );
    }

    #[test]
    fn both_inline_filter_variants_lookup_and_insert_correctly() {
        check::<false, false>();
        check::<true, false>();
        check::<true, true>();
        check::<false, true>();
    }

    fn check<const ONE_HOT: bool, const INCLUDE_PRIMARY: bool>() {
        let keys: Vec<_> = (0..20_000).collect();
        let mut set = InlineFilterCuckooSet::<{ Mode::Lazy as u8 }, ONE_HOT, INCLUDE_PRIMARY>::new(
            1.05, &keys,
        );
        for &key in &keys {
            assert!(set.contains_lazy(key), "missing {key}");
            assert!(
                set.contains_with_token(key, set.prefetch_first(key)),
                "token lookup missed {key}"
            );
        }
        for key in 20_000..21_000 {
            assert!(!set.contains_lazy(key), "unexpected {key}");
        }
        set.insert(30_000);
        assert!(set.contains_lazy(30_000));
    }

    #[test]
    fn one_hot_summary_sets_and_checks_the_same_bit() {
        let mut summary = 0;
        let keys = [17u64, 71, 125, 999];
        for key in keys {
            InlineFilterCuckooSet::<1, true>::add_summary(&mut summary, key);
            assert_ne!(
                summary & InlineFilterCuckooSet::<1, true>::one_hot_bit(key),
                0
            );
            assert!(InlineFilterCuckooSet::<1, true>::might_spill(summary, key));
        }
        assert_eq!(summary.count_ones() as usize, keys.len());
    }
}

//! A dense_hash_set for u64 keys.
//!
//! Compared to std::collections::HashSet<u64>, this uses a different layout: no metadata table, just plain data.
//! This is similar to Google's dense_hash_map, which predates the SwissTable design. By avoiding a metadata table,
//! we may need to do longer probe sequences (each probe is 8 bytes, not 1 byte), but on the other hand we only take
//! 1 cache miss per access, not 2.

use std::hash::{BuildHasher, BuildHasherDefault};

const BIN_SIZE: usize = 7;
use crate::T;
use crate::traits::HashSet;

#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Eq)]
struct FilteredBin {
    spill: u64,
    slots: [T; BIN_SIZE],
}

impl FilteredBin {
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        // Lane 0 is the spill summary; only lanes 1..7 hold keys.
        let lanes: wide::u64x8 = unsafe { std::mem::transmute(*self) };
        lanes.simd_eq(wide::u64x8::splat(key)).to_bitmask() > 1
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
    const BRANCHLESS_SECONDARY: bool = false,
    const BFS: bool = false,
> {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[FilteredBin]>,
    len: usize,
    has_zero: bool,
    summary_ready: bool,
}

impl<const MODE: u8, const ONE_HOT: bool, const BRANCHLESS_SECONDARY: bool, const BFS: bool>
    InlineFilterCuckooSet<MODE, ONE_HOT, BRANCHLESS_SECONDARY, BFS>
{
    pub fn new(slot_ratio: f32, keys: &[T]) -> Self {
        let mut this = Self::with_capacity(slot_ratio, keys.len());
        if BFS {
            let mut workspace = crate::cuckoo::BfsConstruction::new(this.num_bins);
            for &key in keys {
                this.insert_bfs(key, &mut workspace);
            }
        } else {
            for &k in keys {
                this.insert(k);
            }
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
    fn bloom_bits(hash: u64) -> u64 {
        Self::one_hot_bit(hash) | (1u64 << ((hash >> 32) & 63))
    }

    #[inline(always)]
    fn one_hot_bit(hash: u64) -> u64 {
        1u64 << (hash & 63)
    }

    #[inline(always)]
    fn add_summary(summary: &mut u64, hash: u64) {
        let bits = if ONE_HOT {
            Self::one_hot_bit(hash)
        } else {
            Self::bloom_bits(hash)
        };
        *summary |= bits;
    }

    #[inline(always)]
    fn might_spill(summary: u64, hash: u64) -> bool {
        let bits = if ONE_HOT {
            Self::one_hot_bit(hash)
        } else {
            Self::bloom_bits(hash)
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
                    let hash = Self::primary_hash(key);
                    let primary = self.bin_idx_from_hash(hash);
                    if primary != bucket {
                        Self::add_summary(&mut self.table[primary].spill, hash);
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
    fn primary_hash(key: T) -> u64 {
        Hasher::default().hash_one(key)
    }

    #[inline(always)]
    fn bin_idx_from_hash(&self, hash: u64) -> usize {
        (((hash as u128) * (self.num_bins as u128)) >> 64) as usize
    }

    #[inline(always)]
    fn bin_idx_1(&self, key: T) -> usize {
        self.bin_idx_from_hash(Self::primary_hash(key))
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
        let hash = Self::primary_hash(key);
        let bin_idx = self.bin_idx_from_hash(hash);
        prefetch_index::prefetch_index(&self.table, bin_idx);
        // Lazy lookup reuses this hash for both the bin index and spill bits.
        hash as usize
    }
    #[inline(always)]
    pub fn prefetch_second(&self, key: T) -> usize {
        let bin_idx = self.bin_idx_2(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
        bin_idx
    }
    #[inline(always)]
    pub fn prefetch_both(&self, key: T) -> usize {
        let b1 = self.bin_idx_1(key);
        prefetch_index::prefetch_index(&self.table, b1);
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

        let hash = Self::primary_hash(key);
        let bin_idx = self.bin_idx_from_hash(hash);
        let bin1 = self.get_bin(bin_idx);
        if BRANCHLESS_SECONDARY {
            let primary_match = bin1.contains(key);
            let may_spill = Self::might_spill(bin1.spill, hash);
            let bin2 = self.bin_idx_2(key);
            let secondary_idx = Self::choose_secondary(bin_idx, bin2, (!primary_match) & may_spill);
            return primary_match | self.get_bin(secondary_idx).contains(key);
        }
        if ONE_HOT {
            let primary_match = bin1.contains(key);
            let may_spill = Self::might_spill(bin1.spill, hash);
            if may_spill {
                return primary_match | self.get_bin(self.bin_idx_2(key)).contains(key);
            }
            return primary_match;
        }
        if bin1.contains(key) {
            return true;
        }
        if !Self::might_spill(bin1.spill, hash) {
            return false;
        }
        self.get_bin(self.bin_idx_2(key)).contains(key)
    }

    #[inline(always)]
    pub fn contains_eager(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        self.get_bin(self.bin_idx_1(key)).contains(key)
            | self.get_bin(self.bin_idx_2(key)).contains(key)
    }

    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        match Mode::from(MODE) {
            Mode::Eager => self.contains_eager(key),
            Mode::Lazy => self.contains_lazy(key),
        }
    }

    /// Look up 8, 16, or 32 keys together, staging primary and secondary accesses.
    #[inline]
    pub fn contains_batch<const B: usize>(&self, keys: &[T; B]) -> [bool; B] {
        assert!(matches!(B, 8 | 16 | 32), "batch size must be 8, 16, or 32");
        let mut found = [false; B];
        if self.num_bins == 0 {
            for i in 0..B {
                found[i] = keys[i] == 0 && self.has_zero;
            }
            return found;
        }

        let mut hashes = [0u64; B];
        let mut primary = [0usize; B];
        for i in 0..B {
            if keys[i] == 0 {
                found[i] = self.has_zero;
            } else {
                hashes[i] = Self::primary_hash(keys[i]);
                primary[i] = self.bin_idx_from_hash(hashes[i]);
            }
        }

        for i in 0..B {
            if keys[i] != 0 {
                prefetch_index::prefetch_index(&self.table, primary[i]);
            }
        }

        let mut pending = [0usize; B];
        let mut pending_len = 0;
        for i in 0..B {
            if keys[i] == 0 {
                continue;
            }
            let bin = self.get_bin(primary[i]);
            let may_spill = Self::might_spill(bin.spill, hashes[i]);
            if bin.contains(keys[i]) {
                found[i] = true;
            } else if may_spill {
                pending[pending_len] = i;
                pending_len += 1;
            }
        }

        let mut secondary = [0usize; B];
        for &i in &pending[..pending_len] {
            secondary[i] = self.bin_idx_2(keys[i]);
            prefetch_index::prefetch_index(&self.table, secondary[i]);
        }
        for &i in &pending[..pending_len] {
            found[i] = self.get_bin(secondary[i]).contains(keys[i]);
        }
        found
    }

    pub fn count_batches<const B: usize>(&self, keys: &[T]) -> usize {
        assert!(matches!(B, 8 | 16 | 32), "batch size must be 8, 16, or 32");
        let mut chunks = keys.chunks_exact(B);
        let mut count = 0;
        for chunk in &mut chunks {
            let batch: &[T; B] = chunk.try_into().unwrap();
            count += self
                .contains_batch(batch)
                .iter()
                .filter(|&&found| found)
                .count();
        }
        count
            + chunks
                .remainder()
                .iter()
                .filter(|&&key| self.contains(key))
                .count()
    }

    #[inline(always)]
    pub fn contains_lazy_with_token(&self, key: T, token: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let hash = token as u64;
        let primary_idx = self.bin_idx_from_hash(hash);
        let bin1 = self.get_bin(primary_idx);
        if BRANCHLESS_SECONDARY {
            let primary_match = bin1.contains(key);
            let may_spill = Self::might_spill(bin1.spill, hash);
            let bin2 = self.bin_idx_2(key);
            let secondary_idx =
                Self::choose_secondary(primary_idx, bin2, (!primary_match) & may_spill);
            return primary_match | self.get_bin(secondary_idx).contains(key);
        }
        if ONE_HOT {
            let primary_match = bin1.contains(key);
            let may_spill = Self::might_spill(bin1.spill, hash);
            if may_spill {
                return primary_match | self.get_bin(self.bin_idx_2(key)).contains(key);
            }
            return primary_match;
        }
        if bin1.contains(key) {
            return true;
        }
        if !Self::might_spill(bin1.spill, hash) {
            return false;
        }
        self.get_bin(self.bin_idx_2(key)).contains(key)
    }

    #[inline(always)]
    pub fn contains_eager_with_token(&self, key: T, bin: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let bin1 = bin >> 32;
        let bin2 = bin as u32 as usize;

        self.get_bin(bin1).contains(key) | self.get_bin(bin2).contains(key)
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

    fn insert_bfs(&mut self, key: T, workspace: &mut crate::cuckoo::BfsConstruction) {
        if self.contains_eager(key) {
            return;
        }
        if key == 0 {
            self.has_zero = true;
            self.len += 1;
            return;
        }
        let roots = [self.bin_idx_1(key), self.bin_idx_2(key)];
        let path = workspace.path(&roots, BIN_SIZE, |bin, slot| {
            let occupant = self.table[bin].slots[slot];
            (
                occupant == 0,
                (occupant != 0).then(|| self.bin_idx_1(occupant) ^ self.bin_idx_2(occupant) ^ bin),
            )
        });
        for pair in path.windows(2) {
            let (dst, ds) = pair[0];
            let (src, ss) = pair[1];
            self.table[dst].slots[ds] = self.table[src].slots[ss];
        }
        let &(root, slot) = path.last().unwrap();
        self.table[root].slots[slot] = key;
        self.len += 1;
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

impl<const MODE: u8, const ONE_HOT: bool, const BRANCHLESS_SECONDARY: bool, const BFS: bool> HashSet
    for InlineFilterCuckooSet<MODE, ONE_HOT, BRANCHLESS_SECONDARY, BFS>
{
    fn name(&self) -> &'static str {
        if BFS {
            return match (Mode::from(MODE), ONE_HOT) {
                (Mode::Lazy, true) => "InlineOneHotCuckooSet<Lazy,BumpedOnly,BFS>",
                (Mode::Eager, true) => "InlineOneHotCuckooSet<Eager,BumpedOnly,BFS>",
                (Mode::Lazy, false) => "InlineTwoHotCuckooSet<Lazy,BumpedOnly,BFS>",
                (Mode::Eager, false) => "InlineTwoHotCuckooSet<Eager,BumpedOnly,BFS>",
            };
        }
        if BRANCHLESS_SECONDARY {
            return match Mode::from(MODE) {
                Mode::Eager => "InlineOneHotCuckooSet<Eager,BumpedOnly,Branchless>",
                Mode::Lazy => "InlineOneHotCuckooSet<Lazy,BumpedOnly,Branchless>",
            };
        }
        match Mode::from(MODE) {
            Mode::Eager if ONE_HOT => "InlineOneHotCuckooSet<Eager,BumpedOnly>",
            Mode::Lazy if ONE_HOT => "InlineOneHotCuckooSet<Lazy,BumpedOnly>",
            Mode::Eager => "InlineTwoHotCuckooSet<Eager,BumpedOnly>",
            Mode::Lazy => "InlineTwoHotCuckooSet<Lazy,BumpedOnly>",
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = InlineFilterCuckooSet::<MODE, ONE_HOT, BRANCHLESS_SECONDARY, BFS>::new(
            self.slot_ratio,
            keys,
        );
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len() as f32 / (self.num_bins * BIN_SIZE) as f32
    }

    fn negative_passes_filter(&self, key: T) -> Option<bool> {
        if matches!(Mode::from(MODE), Mode::Eager) {
            return None;
        }
        let hash = Self::primary_hash(key);
        let bin = self.get_bin(self.bin_idx_from_hash(hash));
        Some(Self::might_spill(bin.spill, hash))
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
    fn has_batch(&self) -> bool {
        true
    }
    fn count_batch(&self, keys: &[T], batch_size: usize) -> usize {
        match batch_size {
            8 => self.count_batches::<8>(keys),
            16 => self.count_batches::<16>(keys),
            32 => self.count_batches::<32>(keys),
            _ => panic!("batch size must be 8, 16, or 32"),
        }
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
    fn spill_lane_does_not_count_as_a_key() {
        let mut bin = FilteredBin {
            spill: 42,
            slots: [0; BIN_SIZE],
        };
        assert!(!bin.contains(42));
        for slot in 0..BIN_SIZE {
            bin.slots[slot] = 42;
            assert!(bin.contains(42));
            bin.slots[slot] = 0;
        }
    }

    #[test]
    fn primary_ratio_matches_first_bucket_hits_and_excludes_zero() {
        let keys: Vec<_> = (0..20_000).collect();
        let set = InlineFilterCuckooSet::<0, false>::new(1.05, &keys);
        let primary = set
            .iter()
            .filter(|&key| key != 0)
            .filter(|&key| set.get_bin(set.bin_idx_1(key)).contains(key))
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
        check::<false>();
        check::<true>();
    }

    fn check<const ONE_HOT: bool>() {
        let keys: Vec<_> = (0..20_000).collect();
        let mut set = InlineFilterCuckooSet::<{ Mode::Lazy as u8 }, ONE_HOT>::new(1.05, &keys);
        for &key in &keys {
            assert!(set.contains_lazy(key), "missing {key}");
            assert!(
                set.contains_with_token(key, set.prefetch_first(key)),
                "token lookup missed {key}"
            );
        }
        for key in 20_000..21_000 {
            assert!(!set.contains_lazy(key), "unexpected {key}");
            assert!(
                !set.contains_with_token(key, set.prefetch_first(key)),
                "token lookup found absent {key}"
            );
        }
        set.insert(30_000);
        assert!(set.contains_lazy(30_000));
    }

    #[test]
    fn one_hot_summary_sets_and_checks_the_same_bit() {
        let mut summary = 0;
        let keys = [17u64, 71, 125, 999];
        for key in keys {
            let hash = InlineFilterCuckooSet::<1, true>::primary_hash(key);
            InlineFilterCuckooSet::<1, true>::add_summary(&mut summary, hash);
            assert_ne!(
                summary & InlineFilterCuckooSet::<1, true>::one_hot_bit(hash),
                0
            );
            assert!(InlineFilterCuckooSet::<1, true>::might_spill(summary, hash));
        }
        assert_eq!(summary.count_ones() as usize, keys.len());
    }

    fn check_batch<const B: usize, const ONE_HOT: bool>(
        set: &InlineFilterCuckooSet<{ Mode::Lazy as u8 }, ONE_HOT>,
    ) {
        let primary = (1..20_000)
            .find(|&key| set.get_bin(set.bin_idx_1(key)).contains(key))
            .unwrap();
        let secondary = (1..20_000)
            .find(|&key| !set.get_bin(set.bin_idx_1(key)).contains(key))
            .unwrap();
        let queries: [T; B] = std::array::from_fn(|i| match i % 4 {
            0 => 0,
            1 => primary,
            2 => secondary,
            _ => 20_000 + i as u64,
        });
        let expected = queries.map(|key| set.contains_lazy(key));
        assert_eq!(set.contains_batch(&queries), expected);
        for block in 0..16 {
            let queries: [T; B] = std::array::from_fn(|i| {
                let index = block * B + i;
                match index % 3 {
                    0 => (index * 7919 % 20_000) as T,
                    1 => 20_000 + index as T,
                    _ => 0,
                }
            });
            assert_eq!(
                set.contains_batch(&queries),
                queries.map(|key| set.contains_lazy(key))
            );
        }
    }

    #[test]
    fn batches_match_scalar_lookup_for_each_filter_and_width() {
        let keys: Vec<_> = (0..20_000).collect();
        let mut one_hot = InlineFilterCuckooSet::<1, true>::new(1.05, &keys);
        check_batch::<8, true>(&one_hot);
        check_batch::<16, true>(&one_hot);
        check_batch::<32, true>(&one_hot);

        let two_hot = InlineFilterCuckooSet::<1, false>::new(1.05, &keys);
        check_batch::<8, false>(&two_hot);
        check_batch::<16, false>(&two_hot);
        check_batch::<32, false>(&two_hot);

        one_hot.insert(30_000);
        let queries = [30_000; 8];
        assert_eq!(one_hot.contains_batch(&queries), [true; 8]);
        let queries: Vec<T> = (0..105)
            .map(|i| if i % 3 == 0 { i } else { 20_000 + i })
            .collect();
        let expected = queries.iter().filter(|&&key| one_hot.contains(key)).count();
        assert_eq!(one_hot.count_batches::<8>(&queries), expected);
        assert_eq!(one_hot.count_batches::<16>(&queries), expected);
        assert_eq!(one_hot.count_batches::<32>(&queries), expected);
    }

    #[test]
    fn batch_lookup_handles_empty_table_and_zero() {
        let mut set = InlineFilterCuckooSet::<1, true>::new(1.05, &[]);
        assert_eq!(set.contains_batch(&[1; 8]), [false; 8]);
        set.insert(0);
        assert_eq!(set.contains_batch(&[0; 16]), [true; 16]);
    }
}

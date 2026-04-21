//! A dense_hash_set for u64 keys.
//!
//! Compared to std::collections::HashSet<u64>, this uses a different layout: no metadata table, just plain data.
//! This is similar to Google's dense_hash_map, which predates the SwissTable design. By avoiding a metadata table,
//! we may need to do longer probe sequences (each probe is 8 bytes, not 1 byte), but on the other hand we only take
//! 1 cache miss per access, not 2.

use std::hash::{BuildHasher, BuildHasherDefault};

use super::BIN_SIZE;
use crate::traits::HashSet;
use crate::u64_hashset::Bin;
use crate::S;
use crate::T;

type Hasher = BuildHasherDefault<gxhash::GxHasher>;

#[derive(PartialEq, Eq, Debug, std::marker::ConstParamTy)]
pub enum Mode {
    Eager,
    Lazy,
}

pub struct CuckooSet<const MODE: Mode> {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[Bin]>,
    len: usize,
    has_zero: bool,
}

impl<const MODE: Mode> IntoIterator for &CuckooSet<MODE> {
    type Item = T;

    type IntoIter = impl Iterator<Item = T>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::repeat_n(0, self.has_zero as usize).chain(
            self.table
                .iter()
                .flat_map(|b| b.0.iter().copied())
                .filter(|x| *x != 0),
        )
    }
}

impl<const MODE: Mode> CuckooSet<MODE> {
    pub fn new(slot_ratio: f32, keys: &[T]) -> Self {
        let mut this = Self::with_capacity(slot_ratio, keys.len());
        for &k in keys {
            this.insert(k);
        }
        this
    }
    fn with_capacity(slot_ratio: f32, n: usize) -> Self {
        assert!(slot_ratio >= 1.0);
        let capacity = (n as f32 * slot_ratio).ceil() as usize;
        let num_bins = capacity.div_ceil(BIN_SIZE);
        let table = vec![Bin([0 as T; BIN_SIZE]); num_bins].into_boxed_slice();
        Self {
            slot_ratio,
            num_bins,
            table,
            len: 0,
            has_zero: false,
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
        (hash64 as usize).widening_mul(self.num_bins).1
    }

    #[inline(always)]
    fn bin_idx_2(&self, key: T) -> usize {
        const XOR: u64 = 0x9e3779b97f4a7c15;
        let hash64 = Hasher::default().hash_one(key ^ XOR);
        (hash64 as usize).widening_mul(self.num_bins).1
    }

    #[inline(always)]
    fn get_bin(&self, idx: usize) -> &Bin {
        unsafe { self.table.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn prefetch_first(&self, key: T) {
        let bin_idx = self.bin_idx_1(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
    }
    #[inline(always)]
    pub fn prefetch_second(&self, key: T) {
        let bin_idx = self.bin_idx_2(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
    }
    #[inline(always)]
    pub fn prefetch_both(&self, key: T) {
        self.prefetch_first(key);
        self.prefetch_second(key);
    }
    #[inline(always)]
    pub fn prefetch(&self, key: T) {
        match MODE {
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

        let bin1 = self.get_bin(self.bin_idx_1(key));
        if bin1.contains(keys) {
            return true;
        }
        if bin1.has_zero() {
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
        match MODE {
            Mode::Eager => self.contains_eager(key),
            Mode::Lazy => self.contains_lazy(key),
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: T) {
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
        become self.insert_to_bin(key, idx);
    }

    pub fn test(&self) {
        for x in self {
            assert!(self.contains_eager(x));
        }
    }
}

impl<const MODE: Mode> HashSet for CuckooSet<MODE> {
    fn name(&self) -> &'static str {
        match MODE {
            Mode::Eager => "CuckooSet<Eager>",
            Mode::Lazy => "CuckooSet<Lazy>",
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = CuckooSet::<MODE>::new(self.slot_ratio, keys);
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    #[inline(always)]
    fn prefetch(&self, key: T) {
        self.prefetch(key)
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.contains(key)
    }
}

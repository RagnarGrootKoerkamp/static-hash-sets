//! A fake hashset that can be used for memory throughput measurements.

use std::hash::{BuildHasher, BuildHasherDefault};
use wide::CmpEq;

use super::BIN_SIZE;
use crate::traits::HashSet;
use crate::S;
use crate::T;

type Hasher = BuildHasherDefault<gxhash::GxHasher>;

pub struct MockHashSet {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[Bin]>,
    len: usize,
}

const PADDING: usize = 1000;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(align(64))] // Cache line alignment
pub(crate) struct Bin(pub [T; BIN_SIZE]);

impl Bin {
    /// Check if SIMD-splatted key non-zero key is present in bin.
    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        self.0[0] == key
    }
    #[inline(always)]
    pub fn insert(&mut self, key: T) {
        self.0[0] = key;
    }
}

impl MockHashSet {
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
        let table = vec![Bin([0 as T; BIN_SIZE]); num_bins + PADDING].into_boxed_slice();
        Self {
            slot_ratio,
            num_bins,
            table,
            len: 0,
        }
    }

    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn bin_idx(&self, key: T) -> usize {
        (((key as u128) * (self.num_bins as u128)) >> 64) as usize
    }

    #[inline(always)]
    fn get_bin(&self, idx: usize) -> &Bin {
        unsafe { self.table.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn prefetch(&self, key: T) -> usize {
        let bin_idx = self.bin_idx(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
        bin_idx
    }

    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        let bin_idx = self.bin_idx(key);
        let bin = self.get_bin(bin_idx);
        bin.contains(key)
    }

    #[inline(always)]
    pub fn contains_with_token(&self, key: T, mut bin_idx: usize) -> bool {
        let bin = self.get_bin(bin_idx);
        bin.contains(key)
    }

    #[inline(always)]
    pub fn insert(&mut self, key: T) {
        let mut bin_idx = self.bin_idx(key);
        let bin = &mut self.table[bin_idx];
        bin.insert(key);
    }

    pub fn iter(&self) -> impl Iterator<Item = T> {
        self.table.iter().map(|b| b.0[0]).filter(|x| *x != 0)
    }

    pub fn test(&self) {
        for x in self.iter() {
            assert!(self.contains(x));
        }
    }
}

impl HashSet for MockHashSet {
    fn name(&self) -> &'static str {
        "MockHashSet"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = MockHashSet::new(self.slot_ratio, keys);
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len() as f32 / (self.num_bins * BIN_SIZE) as f32
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    fn prefetch(&self, key: T) -> usize {
        MockHashSet::prefetch(self, key)
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

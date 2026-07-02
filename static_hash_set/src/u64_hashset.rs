//! A dense_hash_set for u64 keys.
//!
//! Compared to std::collections::HashSet<u64>, this uses a different layout: no metadata table, just plain data.
//! This is similar to Google's dense_hash_map, which predates the SwissTable design. By avoiding a metadata table,
//! we may need to do longer probe sequences (each probe is 8 bytes, not 1 byte), but on the other hand we only take
//! 1 cache miss per access, not 2.

use std::hash::{BuildHasher, BuildHasherDefault};

use crate::{traits::HashSet, Bin, BIN_SIZE, S, T};

type Hasher = BuildHasherDefault<gxhash::GxHasher>;

pub struct U64HashSet {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[Bin]>,
    len: usize,
    has_zero: bool,
}

const PADDING: usize = 1000;

impl U64HashSet {
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
    fn bin_idx(&self, key: T) -> usize {
        let hash64 = Hasher::default().hash_one(key);
        (((hash64 as u128) * (self.num_bins as u128)) >> 64) as usize
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
        if key == 0 {
            return self.has_zero;
        }

        let keys = S::splat(key as _);

        let mut bin_idx = self.bin_idx(key);
        loop {
            let bin = self.get_bin(bin_idx);
            if bin.contains(keys) {
                return true;
            }
            if bin.has_zero() {
                return false;
            }

            bin_idx += 1;
        }
    }

    #[inline(always)]
    pub fn contains_with_token(&self, key: T, mut bin_idx: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let keys = S::splat(key as _);

        // let mut bin_idx = self.bin_idx(key);
        loop {
            let bin = self.get_bin(bin_idx);
            if bin.contains(keys) {
                return true;
            }
            if bin.has_zero() {
                return false;
            }

            bin_idx += 1;
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: T) {
        if key == 0 {
            self.len += !self.has_zero as usize;
            self.has_zero = true;
            return;
        }

        let keys = S::splat(key as _);

        let mut bin_idx = self.bin_idx(key);
        loop {
            let bin = &mut self.table[bin_idx];
            if bin.contains(keys) {
                return;
            }
            if bin.has_zero() {
                bin.insert(key);
                self.len += 1;
                return;
            } else {
                bin_idx += 1;
                continue;
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> {
        std::iter::repeat_n(0, self.has_zero as usize).chain(
            self.table
                .iter()
                .flat_map(|b| b.0.iter().copied())
                .filter(|x| *x != 0),
        )
    }

    pub fn test(&self) {
        for x in self.iter() {
            assert!(self.contains(x));
        }
    }
}

impl HashSet for U64HashSet {
    fn name(&self) -> &'static str {
        "U64HashSet"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = U64HashSet::new(self.slot_ratio, keys);
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
        U64HashSet::prefetch(self, key)
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

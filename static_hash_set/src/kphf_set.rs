//! A dense_hash_set for u64 keys.
//!
//! Compared to std::collections::HashSet<u64>, this uses a different layout: no metadata table, just plain data.
//! This is similar to Google's dense_hash_map, which predates the SwissTable design. By avoiding a metadata table,
//! we may need to do longer probe sequences (each probe is 8 bytes, not 1 byte), but on the other hand we only take
//! 1 cache miss per access, not 2.

use super::BIN_SIZE;
use crate::traits::HashSet;
use crate::u64_hashset::Bin;
use crate::S;
use crate::T;

pub struct KphfSet<const MODE: kphf::Mode, const K: usize> {
    pub alpha: f32,
    pub bits_per_key: f32,
    table: Box<[Bin]>,
    len: usize,
    has_zero: bool,
    kphf: kphf::KptrHash<MODE, K>,
}

impl<const MODE: kphf::Mode, const K: usize> IntoIterator for &KphfSet<MODE, K> {
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

impl<const MODE: kphf::Mode, const K: usize> KphfSet<MODE, K> {
    pub fn new(alpha: f32, bits_per_key: f32, keys: &[T]) -> Self {
        let kphf = kphf::KptrHash::<MODE, K>::new::<T>(alpha, bits_per_key, keys).unwrap();

        let n = keys.len();
        let capacity = (n as f32 / alpha).ceil() as usize;
        let num_bins = capacity.div_ceil(BIN_SIZE);
        let table = vec![Bin([0 as T; BIN_SIZE]); num_bins].into_boxed_slice();
        let mut this = Self {
            alpha,
            bits_per_key,
            table,
            len: 0,
            has_zero: false,
            kphf,
        };
        for &k in keys {
            this.insert(k);
        }
        this
    }

    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table) + self.kphf.bits_used() / 8
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len + self.has_zero as usize
    }

    #[inline(always)]
    fn bin_idx(&self, key: T) -> usize {
        self.kphf.get(key)
    }

    #[inline(always)]
    fn get_bin(&self, idx: usize) -> &Bin {
        unsafe { self.table.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn prefetch(&self, key: T) {
        let bin_idx = self.bin_idx(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
    }

    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let keys = S::splat(key as _);

        let bin_idx = self.bin_idx(key);
        let bin = self.get_bin(bin_idx);
        bin.contains(keys)
    }

    #[inline(always)]
    fn insert(&mut self, key: T) {
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

    pub fn test(&self) {
        for x in self {
            assert!(self.contains(x));
        }
    }
}

impl<const MODE: kphf::Mode, const K: usize> HashSet for KphfSet<MODE, K> {
    fn name(&self) -> &'static str {
        match MODE {
            kphf::Mode::Linear => "KphfSet<Linear>",
            kphf::Mode::LinearBump => "KphfSet<LinearBump>",
            kphf::Mode::LinearBumpGreedy => "KphfSet<LinearBumpGreedy>",
            kphf::Mode::Sort => "KphfSet<Sort>",
            kphf::Mode::SortBump => "KphfSet<SortBump>",
            kphf::Mode::SortBumpGreedy => "KphfSet<SortBumpGreedy>",
            kphf::Mode::Consensus => "KphfSet<Consensus>",
            kphf::Mode::ConsensusGreedy => "KphfSet<ConsensusGreedy>",
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = KphfSet::<MODE, K>::new(self.alpha, self.bits_per_key, keys);
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        KphfSet::allocation_size(self)
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    #[inline(always)]
    fn prefetch(&self, key: T) {
        KphfSet::prefetch(self, key)
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.contains(key)
    }
}

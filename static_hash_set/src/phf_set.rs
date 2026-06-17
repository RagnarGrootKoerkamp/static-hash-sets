//! Map each key to a slot using a 1-PHF, then check if its there.
use std::marker::Sync;

use crate::phf_trait::Phf;
use crate::traits::HashSet;
use crate::T;

pub struct PhfSet<PHF: Phf> {
    // pub alpha: f32,
    // pub bits_per_key: f32,
    table: Box<[u64]>,
    len: usize,
    has_zero: bool,
    phf: PHF,
}

impl<PHF: Phf> IntoIterator for &PhfSet<PHF> {
    type Item = T;

    type IntoIter = impl Iterator<Item = T>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::repeat_n(0, self.has_zero as usize)
            .chain(self.table.iter().copied().filter(|x| *x != 0))
    }
}

impl<PHF: Phf> PhfSet<PHF> {
    pub fn new(_alpha: f32, _bits_per_key: f32, keys: &[T]) -> Self {
        let phf = PHF::new(keys);

        // let mut seen = HashSet::new();
        // for k in keys {
        //     let idx = phf.
        //     assert!(seen.insert(k), "Duplicate key {k:?} in input.");
        // }

        let table = vec![0 as T; phf.num_bins()].into_boxed_slice();

        let mut this = Self {
            // alpha,
            // bits_per_key,
            table,
            len: 0,
            has_zero: false,
            phf: phf,
        };
        for &k in keys {
            this.insert(k);
        }
        this
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn bin_idx(&self, key: T) -> usize {
        self.phf.get(key)
    }

    #[inline(always)]
    fn get_bin(&self, idx: usize) -> T {
        unsafe { *self.table.get_unchecked(idx) }
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

        let bin_idx = self.bin_idx(key);
        let bin = self.get_bin(bin_idx);
        bin == key
    }

    #[inline(always)]
    pub fn contains_with_token(&self, key: T, bin_idx: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let bin = self.get_bin(bin_idx);
        bin == key
    }

    #[inline(always)]
    fn insert(&mut self, key: T) {
        if key == 0 {
            assert!(!self.has_zero);
            self.len += 1;
            self.has_zero = true;
            return;
        }

        let bin_idx = self.bin_idx(key);
        let bin = &mut self.table[bin_idx];
        assert!(
            *bin == 0,
            "Trying to insert {key:?} but slot {bin_idx} is already full."
        );
        *bin = key;
        self.len += 1;
    }

    pub fn test(&self) {
        for x in self {
            assert!(self.contains(x));
        }
    }
}

impl<PHF: Phf + 'static + Sync + Send> HashSet for PhfSet<PHF> {
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = PhfSet::<PHF>::new(0.0, 0.0, keys);
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table) + self.phf.bits_used() / 8
    }
    fn load_factor(&self) -> f32 {
        self.len() as f32 / self.phf.num_bins() as f32
    }
    fn kphf_target_bits_per_key(&self) -> f32 {
        self.phf.bits_used() as f32 / self.len() as f32 / 8.0
    }
    fn kphf_size(&self) -> usize {
        self.phf.bits_used() / 8
    }
    // fn alpha(&self) -> f32 {
    // self.len() as f32 / (self.table.len() as f32)
    // }
    fn bumped_frac(&self) -> f32 {
        0.0
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    #[inline(always)]
    fn prefetch(&self, key: T) -> usize {
        PhfSet::prefetch(self, key)
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

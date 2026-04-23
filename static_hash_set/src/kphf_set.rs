//! Map each key to a bucket (cache line) of size k=8 using a non-minimal k-PHF, then check if its there.
use super::BIN_SIZE;
use crate::kphf_trait::Kphf;
use crate::traits::HashSet;
use crate::u64_hashset::Bin;
use crate::S;
use crate::T;

pub struct KphfSet<KPHF, const K: usize> {
    pub alpha: f32,
    pub bits_per_key: f32,
    table: Box<[Bin]>,
    len: usize,
    has_zero: bool,
    kphf: KPHF,
}

impl<KPHF, const K: usize> IntoIterator for &KphfSet<KPHF, K> {
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

impl<KPHF: Kphf<K>, const K: usize> KphfSet<KPHF, K> {
    pub fn try_new(alpha: f32, bits_per_key: f32, keys: &[T]) -> Option<Self> {
        let kphf = KPHF::try_new(alpha, bits_per_key, keys)?;
        let num_bins = match kphf.num_bins() {
            usize::MAX => keys.iter().filter(|&&k| k != 0).count().div_ceil(K).max(1),
            x => x,
        };
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
        Some(this)
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

        let bin_idx = self.bin_idx(key);
        let bin = self.get_bin(bin_idx);
        bin.contains(keys)
    }

    #[inline(always)]
    pub fn contains_with_token(&self, key: T, bin_idx: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }

        let keys = S::splat(key as _);
        let bin = self.get_bin(bin_idx);
        bin.contains(keys)
    }

    #[inline(always)]
    fn insert(&mut self, key: T) {
        if key == 0 {
            assert!(!self.has_zero);
            self.len += 1;
            self.has_zero = true;
            return;
        }

        let keys = S::splat(key as _);

        let bin_idx = self.bin_idx(key);
        let bin = &mut self.table[bin_idx];
        assert!(!bin.contains(keys));
        assert!(
            bin.has_zero(),
            "Trying to insert {key:?} but bin {bin_idx} is already full."
        );
        bin.insert(key);
        self.len += 1;
    }

    pub fn test(&self) {
        for x in self {
            assert!(self.contains(x));
        }
    }
}

impl<KPHF: Kphf<K> + Send + Sync + 'static, const K: usize> HashSet for KphfSet<KPHF, K> {
    fn name(&self) -> &'static str {
        self.kphf.name()
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(KphfSet::<KPHF, K>::try_new(self.alpha, self.bits_per_key, keys).unwrap())
    }
    fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table) + self.kphf.bits_used() / 8
    }
    fn load_factor(&self) -> f32 {
        self.len() as f32 / (self.table.len() as f32 * BIN_SIZE as f32)
    }
    fn kphf_target_bits_per_key(&self) -> f32 {
        self.bits_per_key
    }
    fn kphf_size(&self) -> usize {
        self.kphf.bits_used() / 8
    }
    fn bumped_frac(&self) -> f32 {
        self.kphf.num_bumped() as f32 / self.len() as f32
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    #[inline(always)]
    fn prefetch(&self, key: T) -> usize {
        KphfSet::prefetch(self, key)
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

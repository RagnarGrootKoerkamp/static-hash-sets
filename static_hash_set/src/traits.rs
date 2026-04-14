use fxhash::{FxBuildHasher, FxHasher};

use crate::{static_hashset::StaticHashSet, u64_hashset::U64HashSet};

use super::T;

pub trait HashSet: Send + Sync {
    fn name(&self) -> &'static str;
    fn new(&self, keys: &[T]) -> Box<dyn HashSet>;
    fn allocation_size(&self) -> usize;
    fn has_prefetch(&self) -> bool {
        false
    }
    fn prefetch(&self, _key: T) {}
    fn get(&self, key: T) -> bool;
    fn count(&self, keys: &[T]) -> usize {
        let lookahead = 32;
        let mut c = 0;
        for i in 0..keys.len().saturating_sub(lookahead) {
            self.prefetch(keys[i + lookahead]);
            c += self.get(keys[i]) as usize;
        }
        std::hint::black_box(c);
        c
    }
}

impl HashSet for hashbrown::HashSet<T, FxBuildHasher> {
    fn name(&self) -> &'static str {
        "FxHashSet"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let mut h = hashbrown::HashSet::with_capacity_and_hasher(keys.len(), Default::default());
        for &k in keys {
            h.insert(k);
        }
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    #[inline(always)]
    fn get(&self, key: T) -> bool {
        self.contains(&key)
    }
}

impl HashSet for fastbloom::BloomFilter<FxBuildHasher> {
    fn name(&self) -> &'static str {
        "BloomFilter"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = fastbloom::BloomFilter::with_false_pos(0.01)
            .hasher(FxBuildHasher::default())
            .items(keys.iter().copied());
        // eprintln!("\nBits/elem:  {}", h.num_bits() as f32 / keys.len() as f32);
        // eprintln!("Num hashes: {}", h.num_hashes());
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        0
    }
    #[inline(always)]
    fn get(&self, key: T) -> bool {
        self.contains(&key)
    }
}

impl HashSet for cuckoofilter::CuckooFilter<FxHasher> {
    fn name(&self) -> &'static str {
        "CuckooFilter"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let mut h = Self::with_capacity(keys.len() + keys.len() / 10);
        for &k in keys {
            h.add(&k).unwrap();
        }
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        0
    }
    #[inline(always)]
    fn get(&self, key: T) -> bool {
        self.contains(&key)
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
    fn has_prefetch(&self) -> bool {
        true
    }
    fn prefetch(&self, key: T) {
        U64HashSet::prefetch(self, key)
    }
    #[inline(always)]
    fn get(&self, key: T) -> bool {
        self.contains(key)
    }
}

impl HashSet for StaticHashSet {
    fn name(&self) -> &'static str {
        "StaticHashSet"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = StaticHashSet::new(self.slot_ratio, self.meta_ratio, keys);
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
        StaticHashSet::prefetch(self, key)
    }
    #[inline(always)]
    fn get(&self, key: T) -> bool {
        self.contains(key)
    }
}

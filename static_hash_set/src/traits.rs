use super::T;

pub trait HashSet: Send + Sync {
    fn name(&self) -> &'static str;
    fn new(&self, keys: &[T]) -> Box<dyn HashSet>;
    /// Bytes of the entire data structure.
    fn allocation_size(&self) -> usize;
    fn load_factor(&self) -> f32;
    fn kphf_target_bits_per_key(&self) -> f32 {
        0.0
    }
    /// Bytes of the KPHF only.
    fn kphf_size(&self) -> usize {
        0
    }
    /// Fraction of keys bumped in the first KPHF level.
    fn bumped_frac(&self) -> f32 {
        0.0
    }
    fn has_prefetch(&self) -> bool {
        false
    }
    fn prefetch(&self, _key: T) {}
    fn contains(&self, key: T) -> bool;
    fn count_loop(&self, keys: &[T]) -> usize {
        let mut c = 0;
        for &key in keys {
            c += self.contains(key) as usize;
        }
        std::hint::black_box(c);
        c
    }
    fn count_prefetch(&self, keys: &[T]) -> usize {
        let lookahead = 32;
        let mut c = 0;
        for i in 0..keys.len().saturating_sub(lookahead) {
            self.prefetch(unsafe { *keys.get_unchecked(i + lookahead) });
            c += self.contains(keys[i]) as usize;
        }
        std::hint::black_box(c);
        c
    }
}

// FIXME: Prefetch
impl HashSet for hashbrown::HashSet<T, gxhash::GxBuildHasher> {
    fn name(&self) -> &'static str {
        "FxHashSet"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = hashbrown::HashSet::from_iter(keys.iter().cloned());
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len() as f32 / self.capacity() as f32
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.contains(&key)
    }
}

impl HashSet for fastbloom::BloomFilter<gxhash::GxBuildHasher> {
    fn name(&self) -> &'static str {
        "BloomFilter"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = fastbloom::BloomFilter::with_false_pos(0.01)
            .hasher(gxhash::GxBuildHasher::default())
            .items(keys.iter().copied());
        // eprintln!("\nBits/elem:  {}", h.num_bits() as f32 / keys.len() as f32);
        // eprintln!("Num hashes: {}", h.num_hashes());
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        0
    }
    fn load_factor(&self) -> f32 {
        0.0
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.contains(&key)
    }
}

impl HashSet for cuckoofilter::CuckooFilter<gxhash::GxHasher> {
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
    fn load_factor(&self) -> f32 {
        0.0
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.contains(&key)
    }
}

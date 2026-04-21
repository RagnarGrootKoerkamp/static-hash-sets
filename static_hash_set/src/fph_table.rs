use crate::traits::HashSet;
use crate::T;
use fph_table_sys::{DynFphSet, MetaFphSet};

/// Hash set backed by `fph::DynamicFphSet<uint64_t, SimpleSeedHash>`.
pub struct FphDynSet {
    n: usize,
    inner: DynFphSet,
    max_load_factor: f32,
}

impl FphDynSet {
    pub fn new(max_load_factor: f32, keys: &[T]) -> Option<Self> {
        Some(Self {
            n: keys.len(),
            inner: DynFphSet::new(keys, max_load_factor)?,
            max_load_factor,
        })
    }
}

impl HashSet for FphDynSet {
    fn name(&self) -> &'static str {
        "FphDynSet"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(FphDynSet::new(self.max_load_factor, keys).expect("FphDynSet build failed"))
    }
    fn allocation_size(&self) -> usize {
        // Slots only; bucket-param array adds ~c*n/(log2(n)+1)*4 bytes but is not exposed.
        self.inner.slot_count() * size_of::<T>()
    }
    fn load_factor(&self) -> f32 {
        self.n as f32 / self.inner.slot_count() as f32
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.inner.contains(key)
    }
}

/// Hash set backed by `fph::MetaFphSet<uint64_t, SimpleSeedHash>`.
///
/// Faster than [`FphDynSet`] for negative lookups at large sizes; slower for positive ones.
pub struct FphMetaSet {
    n: usize,
    inner: MetaFphSet,
    max_load_factor: f32,
}

impl FphMetaSet {
    pub fn new(max_load_factor: f32, keys: &[T]) -> Option<Self> {
        Some(Self {
            n: keys.len(),
            inner: MetaFphSet::new(keys, max_load_factor)?,
            max_load_factor,
        })
    }
}

impl HashSet for FphMetaSet {
    fn name(&self) -> &'static str {
        "FphMetaSet"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(FphMetaSet::new(self.max_load_factor, keys).expect("FphMetaSet build failed"))
    }
    fn allocation_size(&self) -> usize {
        // Slots + 1-byte metadata per element.
        self.inner.slot_count() * size_of::<T>() + self.inner.elem_count()
    }
    fn load_factor(&self) -> f32 {
        self.n as f32 / self.inner.slot_count() as f32
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.inner.contains(key)
    }
}

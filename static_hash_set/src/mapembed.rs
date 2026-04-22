use crate::traits::HashSet;
use crate::T;
use mapembed_sys::MapEmbedSet;

/// Hash set backed by `MapEmbed` (KEY_LEN=8, VAL_LEN=0, 3 layers, cell_bit=4).
pub struct MapEmbed {
    n: usize,
    inner: MapEmbedSet,
}

impl MapEmbed {
    pub fn new(keys: &[T]) -> Option<Self> {
        Some(Self {
            n: keys.len(),
            inner: MapEmbedSet::new(keys)?,
        })
    }
}

impl HashSet for MapEmbed {
    fn name(&self) -> &'static str {
        "MapEmbed"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(MapEmbed::new(keys).expect("MapEmbed build failed"))
    }
    fn allocation_size(&self) -> usize {
        self.inner.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        // N = 8 slots per bucket
        self.n as f32 / (self.inner.bucket_number() * 8) as f32
    }
    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        self.inner.contains(key)
    }
}

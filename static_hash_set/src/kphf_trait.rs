use super::T;

pub trait Kphf<const K: usize>: Sized {
    fn name(&self) -> &'static str;
    fn new(alpha: f32, bits_per_key: f32, keys: &[T]) -> Self {
        Self::try_new(alpha, bits_per_key, keys).unwrap()
    }
    fn try_new(alpha: f32, bits_per_key: f32, keys: &[T]) -> Option<Self>;
    fn num_bins(&self) -> usize;
    fn get(&self, key: T) -> usize;
    fn bits_used(&self) -> usize;
    fn num_bumped(&self) -> usize;
}

#[cfg(feature = "kphf")]
impl<const MODE: u8, const K: usize> Kphf<K> for kphf::KptrHash<MODE, K> {
    fn name(&self) -> &'static str {
        match kphf::Mode::from(MODE) {
            kphf::Mode::Linear => "KptrHash<Linear>",
            kphf::Mode::LinearBump => "KptrHash<LinearBump>",
            kphf::Mode::LinearBumpGreedy => "KptrHash<LinearBumpGreedy>",
            kphf::Mode::Sort => "KptrHash<Sort>",
            kphf::Mode::SortBump => "KptrHash<SortBump>",
            kphf::Mode::SortBumpGreedy => "KptrHash<SortBumpGreedy>",
        }
    }

    fn try_new(alpha: f32, bits_per_key: f32, keys: &[T]) -> Option<Self> {
        kphf::KptrHash::<MODE, K>::new::<T>(alpha, bits_per_key, keys)
    }

    fn num_bins(&self) -> usize {
        self.num_bins()
    }

    #[inline(always)]
    fn get(&self, key: T) -> usize {
        self.get(key)
    }

    fn bits_used(&self) -> usize {
        self.bits_used()
    }

    fn num_bumped(&self) -> usize {
        self.num_bumped()
    }
}

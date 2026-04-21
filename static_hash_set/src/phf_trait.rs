use ph::{
    phast::{self, compressed_array::CompactFast},
    GetSize,
};
use ptr_hash::DefaultPtrHash;

use super::T;

/// PtrHash with default params:
/// - alpha = 0.99
/// - Fast linear remapping
/// - lambda = 3.0
/// on top of that, for faster queries:
/// - single part
/// - no remapping
pub type PtrHash = ptr_hash::DefaultPtrHash<ptr_hash::hash::Gx>;

pub trait Phf {
    fn name(&self) -> &'static str;
    fn new(keys: &[T]) -> Self;
    fn num_bins(&self) -> usize;
    fn get(&self, key: T) -> usize;
    fn bits_used(&self) -> usize;
}

impl Phf for PtrHash {
    fn name(&self) -> &'static str {
        "PtrHash"
    }

    fn new(keys: &[T]) -> Self {
        let mut params = ptr_hash::PtrHashParams::<_>::default_fast();
        params.remap = false;
        params.single_part = true;
        DefaultPtrHash::new(keys, params)
    }

    fn num_bins(&self) -> usize {
        self.max_index()
    }

    #[inline(always)]
    fn get(&self, key: T) -> usize {
        self.index_single_part_no_remap(&key)
    }

    fn bits_used(&self) -> usize {
        let (a, b) = self.bits_per_element();
        assert_eq!(b, 0.0);
        (a * self.n() as f64) as usize
    }
}

/// Chosen non-minimal PHast configuration:
/// - Phast+ (3.3)
/// - delta = 2
/// - S=8
/// - lambda = 4.1
/// - No remapping (using Perfect instead of Function2)
pub type PHast = ph::phast::Perfect<ph::seeds::Bits8, ph::phast::ShiftOnlyWrapped<2>>;

impl Phf for PHast {
    fn name(&self) -> &'static str {
        "PHast"
    }

    fn new(keys: &[T]) -> Self {
        let params = phast::Params::new(ph::seeds::Bits8, 410);
        Self::with_slice_p_threads_hash_sc(
            keys,
            &params,
            6,
            seedable_hash::BuildDefaultSeededHasher::default(),
            phast::ShiftOnlyWrapped::<2>,
        )
    }

    fn num_bins(&self) -> usize {
        self.output_range()
    }

    #[inline(always)]
    fn get(&self, key: T) -> usize {
        self.get(&key)
    }

    fn bits_used(&self) -> usize {
        self.size_bytes_content_dyn() * 8
    }
}

/// Chosen minimal PHast configuration:
/// - Phast+ (3.3)
/// - delta = 2
/// - S=8
/// - lambda = 4.1
/// - With CompactFast remapping (using Function2 instead of Perfect)
pub type PHastMinimal =
    ph::phast::Function2<ph::seeds::Bits8, ph::phast::ShiftOnlyWrapped<2>, CompactFast>;

impl Phf for PHastMinimal {
    fn name(&self) -> &'static str {
        "PHast"
    }

    fn new(keys: &[T]) -> Self {
        let params = phast::Params::new(ph::seeds::Bits8, 410);
        Self::with_slice_p_threads_hash_sc(
            keys,
            &params,
            6,
            seedable_hash::BuildDefaultSeededHasher::default(),
            phast::ShiftOnlyWrapped::<2>,
        )
    }

    fn num_bins(&self) -> usize {
        self.output_range()
    }

    #[inline(always)]
    fn get(&self, key: T) -> usize {
        self.get(&key)
    }

    fn bits_used(&self) -> usize {
        self.size_bytes_content_dyn() * 8
    }
}

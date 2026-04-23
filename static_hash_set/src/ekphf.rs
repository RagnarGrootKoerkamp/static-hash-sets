//! Wrappers around engineering-k-perfect-hashing library (ThresholdBasedBumping and HashDisplace).
//!
//! These are k-perfect hash functions (k-PHFs): each key maps to a bucket that holds
//! at most k=8 keys. To use as a set, actual keys are stored in bins indexed by the PHF.
use crate::kphf_trait::Kphf;
use crate::traits::HashSet;
use crate::u64_hashset::Bin;
use crate::BIN_SIZE;
use crate::S;
use crate::T;
use engineering_kphf::{Hd8Set, Tbb84pSet, Tbb85Set};

impl Kphf<8> for Tbb85Set {
    fn name(&self) -> &'static str {
        "Tbb85Set"
    }
    fn try_new(_alpha: f32, _bits_per_key: f32, keys: &[T]) -> Option<Self> {
        let non_zero: Vec<T> = keys.iter().copied().filter(|&k| k != 0).collect();
        Some(Tbb85Set::new(&non_zero, 2.0))
    }
    fn num_bins(&self) -> usize {
        usize::MAX
    }
    #[inline(always)]
    fn get(&self, key: T) -> usize {
        self.query(key) as usize
    }
    fn bits_used(&self) -> usize {
        self.count_bits()
    }
    fn num_bumped(&self) -> usize {
        0
    }
}

impl Kphf<8> for Tbb84pSet {
    fn name(&self) -> &'static str {
        "Tbb84pSet"
    }
    fn try_new(_alpha: f32, _bits_per_key: f32, keys: &[T]) -> Option<Self> {
        let non_zero: Vec<T> = keys.iter().copied().filter(|&k| k != 0).collect();
        Some(Tbb84pSet::new(&non_zero, 2.0))
    }
    fn num_bins(&self) -> usize {
        usize::MAX
    }
    #[inline(always)]
    fn get(&self, key: T) -> usize {
        self.query(key) as usize
    }
    fn bits_used(&self) -> usize {
        self.count_bits()
    }
    fn num_bumped(&self) -> usize {
        0
    }
}

impl Kphf<8> for Hd8Set {
    fn name(&self) -> &'static str {
        "Hd8Set"
    }
    fn try_new(_alpha: f32, _bits_per_key: f32, keys: &[T]) -> Option<Self> {
        let non_zero: Vec<T> = keys.iter().copied().filter(|&k| k != 0).collect();
        Some(Hd8Set::new(&non_zero, 12))
    }
    fn num_bins(&self) -> usize {
        usize::MAX
    }
    #[inline(always)]
    fn get(&self, key: T) -> usize {
        self.query(key) as usize
    }
    fn bits_used(&self) -> usize {
        self.count_bits()
    }
    fn num_bumped(&self) -> usize {
        0
    }
}

use super::BIN_SIZE;
use super::S;
use super::T;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(align(64))] // Cache line alignment
pub(crate) struct Bin(pub [T; BIN_SIZE]);

impl Bin {
    /// Check if SIMD-splatted key non-zero key is present in bin.
    #[inline(always)]
    pub fn contains(&self, keys: S) -> bool {
        let [h1, h2]: [S; 2] = unsafe { std::mem::transmute(*self) };
        (h1.simd_eq(keys) | h2.simd_eq(keys)).to_bitmask() > 0
    }
    /// Check if the bin contains a 0 entry.
    #[inline(always)]
    pub fn has_zero(&self) -> bool {
        let [h1, h2]: [S; 2] = unsafe { std::mem::transmute(*self) };
        (h1.simd_eq(S::ZERO) | h2.simd_eq(S::ZERO)).to_bitmask() > 0
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        let [h1, h2]: [S; 2] = unsafe { std::mem::transmute(*self) };
        BIN_SIZE
            - (h1.simd_eq(S::ZERO).to_bitmask().count_ones()
                + h2.simd_eq(S::ZERO).to_bitmask().count_ones()) as usize
    }
    #[inline(always)]
    pub fn insert(&mut self, key: T) {
        let idx = self.len();
        assert_eq!(
            self.0[idx], 0,
            "inserting {key} at idx {idx} with bin size {BIN_SIZE}"
        );
        self.0[idx] = key;
    }
}

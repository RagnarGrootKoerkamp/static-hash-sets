//! Exact quotienting in 64-byte buckets, using fast-range multiplication.
//! The remaining hash bits and primary/secondary bit are split into contiguous
//! prefixes followed by bit-packed suffixes. Prefix filtering only selects
//! candidates: the complete entry is always checked, so membership is exact.
//!
//! Keys must be nonzero and fit in 62 bits. The reversible hash also stays in
//! the 62-bit domain, saving two bits per entry. Out-of-domain lookups return false.
//! Zero keys are rejected. Zero entries pad unused slots; a zero primary encoding
//! is routed to the secondary bucket, where the location bit makes it nonzero.
//! `PREFIX` selects 8-, 16-, or 32-bit filtering (0 disables it); `SIMD=false`
//! selects u128 SWAR lane comparisons. The default is 16-bit SIMD filtering.
//! Set the fourth const parameter `EF=true` for sorted Elias–Fano buckets:
//! `CuckooSet::<0, 16, true, true>::new(slot_ratio, keys)`. Prefixes and suffixes
//! hold the low parts; a unary high-part bitmap also records occupancy. EF
//! buckets support up to 16 entries and are re-sorted on insertion/replacement.
//! Bucket counts may be arbitrary: the high
//! half of hash*m selects the bucket, and the high bits of its low half form the
//! quotient. Lookup uses multiplication and shifts, with no integer division.
//!
//! NOTE: THIS FILE IS AI SLOP!!!

pub use crate::cuckoo::Mode;
use crate::{T, traits::HashSet};

/// Quotienting accepts nonzero keys in the 62-bit domain.
pub const KEY_BITS: u32 = 62;
pub const KEY_MASK: u64 = (1u64 << KEY_BITS) - 1;
const PADDING_BITS: u32 = 64 - KEY_BITS;

#[derive(Clone)]
#[repr(align(64))]
struct PackedBin([u64; 8]);

impl PackedBin {
    #[inline(always)]
    fn read(&self, offset: usize, bits: usize) -> u64 {
        if bits == 0 {
            return 0;
        }
        let word = offset / 64;
        let shift = offset % 64;
        let mut value = self.0[word] >> shift;
        if shift + bits > 64 {
            value |= self.0[word + 1] << (64 - shift);
        }
        value & (u64::MAX >> (64 - bits))
    }

    fn write(&mut self, offset: usize, bits: usize, value: u64) {
        if bits == 0 {
            return;
        }
        let word = offset / 64;
        let shift = offset % 64;
        let mask = u64::MAX >> (64 - bits);
        self.0[word] = (self.0[word] & !(mask << shift)) | (value << shift);
        if shift + bits > 64 {
            let high_bits = shift + bits - 64;
            let mask = u64::MAX >> (64 - high_bits);
            self.0[word + 1] = (self.0[word + 1] & !mask) | (value >> (64 - shift));
        }
    }

    #[inline(always)]
    fn get<const P: usize>(&self, slot: usize, bits: usize, slots: usize) -> u64 {
        if P == 0 {
            return self.read(slot * bits, bits);
        }
        let prefix = self.read(slot * P, P);
        prefix | (self.read(slots * P + slot * (bits - P), bits - P) << P)
    }

    fn set<const P: usize>(&mut self, slot: usize, bits: usize, slots: usize, value: u64) {
        assert_ne!(value, 0, "zero is the empty-entry sentinel");
        self.write_value::<P>(slot, bits, slots, value);
    }

    fn write_value<const P: usize>(&mut self, slot: usize, bits: usize, slots: usize, value: u64) {
        if P == 0 {
            self.write(slot * bits, bits, value);
            return;
        }
        self.write(slot * P, P, value & (u64::MAX >> (64 - P)));
        self.write(slots * P + slot * (bits - P), bits - P, value >> P);
    }

    #[inline(always)]
    fn full<const P: usize>(&self, bits: usize, slots: usize) -> bool {
        self.get::<P>(slots - 1, bits, slots) != 0
    }

    fn push<const P: usize>(&mut self, value: u64, bits: usize, slots: usize) {
        let slot = (0..slots)
            .find(|&i| self.get::<P>(i, bits, slots) == 0)
            .expect("full bucket");
        self.set::<P>(slot, bits, slots, value);
    }

    #[inline(always)]
    fn candidates<const P: usize, const SIMD: bool>(&self, value: u64, slots: usize) -> u32 {
        let mut mask = 0u32;
        if SIMD {
            // All loads stay inside the cacheline. Lanes beyond `slots` are masked out.
            // Prefixes occupy the first slots*P bits in native little-endian lane order.
            match P {
                8 => {
                    let lanes = unsafe { self.0.as_ptr().cast::<[u8; 16]>().read_unaligned() };
                    mask = wide::u8x16::from(lanes)
                        .simd_eq(wide::u8x16::splat(value as u8))
                        .to_bitmask() as u32;
                }
                16 => {
                    let lanes = unsafe { self.0.as_ptr().cast::<[u16; 16]>().read_unaligned() };
                    mask = wide::u16x16::from(lanes)
                        .simd_eq(wide::u16x16::splat(value as u16))
                        .to_bitmask() as u32;
                }
                32 => {
                    for block in 0..2 {
                        let lanes = unsafe {
                            self.0
                                .as_ptr()
                                .add(block * 4)
                                .cast::<[u32; 8]>()
                                .read_unaligned()
                        };
                        mask |= (wide::u32x8::from(lanes)
                            .simd_eq(wide::u32x8::splat(value as u32))
                            .to_bitmask() as u32)
                            << (block * 8);
                    }
                }
                _ => unreachable!(),
            }
        } else {
            // Exact zero-lane detection on u128 blocks (no borrow into adjacent lanes).
            let ones = u128::MAX / ((1u128 << P) - 1);
            let high = ones << (P - 1);
            let low = high - ones;
            let needle = (value as u128 & ((1u128 << P) - 1)) * ones;
            for block in 0..(slots * P).div_ceil(128) {
                let word = self.0[block * 2] as u128 | ((self.0[block * 2 + 1] as u128) << 64);
                let x = word ^ needle;
                let mut equal = !(((x & low).wrapping_add(low)) | x | low) & high;
                while equal != 0 {
                    let lane = equal.trailing_zeros() as usize / P;
                    mask |= 1 << (block * (128 / P) + lane);
                    equal &= equal - 1;
                }
            }
        }
        mask & ((1 << slots) - 1)
    }

    #[inline(always)]
    fn contains<const P: usize, const SIMD: bool>(
        &self,
        value: u64,
        bits: usize,
        slots: usize,
    ) -> bool {
        if value == 0 {
            return false;
        }
        if P == 0 {
            return (0..slots).any(|slot| self.get::<0>(slot, bits, slots) == value);
        }
        let mut mask = self.candidates::<P, SIMD>(value, slots);
        while mask != 0 {
            let slot = mask.trailing_zeros() as usize;
            if self.get::<P>(slot, bits, slots) == value {
                return true;
            }
            mask &= mask - 1;
        }
        false
    }
    #[inline(always)]
    fn bitmap(&self, bits: usize, slots: usize, low_bits: usize) -> u64 {
        self.read(slots * low_bits, (1 << (bits - low_bits)) + slots)
    }

    #[inline(always)]
    fn get_encoded<const P: usize, const EF: bool>(
        &self,
        slot: usize,
        bits: usize,
        slots: usize,
        low_bits: usize,
    ) -> u64 {
        if !EF {
            return self.get::<P>(slot, bits, slots);
        }
        let mut bitmap = self.bitmap(bits, slots, low_bits);
        if slot >= bitmap.count_ones() as usize {
            return 0;
        }
        for _ in 0..slot {
            bitmap &= bitmap - 1;
        }
        let high = bitmap.trailing_zeros() as usize - slot;
        ((high as u64) << low_bits) | self.get::<P>(slot, low_bits, slots)
    }

    fn set_encoded<const P: usize, const EF: bool>(
        &mut self,
        slot: usize,
        bits: usize,
        slots: usize,
        low_bits: usize,
        value: u64,
    ) {
        if !EF {
            self.set::<P>(slot, bits, slots, value);
            return;
        }
        assert_ne!(value, 0, "zero is the empty-entry sentinel");
        let mut bitmap = self.bitmap(bits, slots, low_bits);
        let count = bitmap.count_ones() as usize;
        assert!(slot < count || (slot == count && count < slots));
        let mut entries = [0u64; 16];
        for (i, entry) in entries[..count].iter_mut().enumerate() {
            let high = bitmap.trailing_zeros() as usize - i;
            *entry = ((high as u64) << low_bits) | self.get::<P>(i, low_bits, slots);
            bitmap &= bitmap - 1;
        }
        entries[slot] = value;
        let count = count + (slot == count) as usize;
        entries[..count].sort_unstable();
        self.0 = [0; 8];
        let low_mask = u64::MAX >> (64 - low_bits);
        for (i, &entry) in entries[..count].iter().enumerate() {
            self.write_value::<P>(i, low_bits, slots, entry & low_mask);
            bitmap |= 1 << ((entry >> low_bits) as usize + i);
        }
        self.write(slots * low_bits, (1 << (bits - low_bits)) + slots, bitmap);
    }

    #[inline(always)]
    fn full_encoded<const P: usize, const EF: bool>(
        &self,
        bits: usize,
        slots: usize,
        low_bits: usize,
    ) -> bool {
        if EF {
            self.bitmap(bits, slots, low_bits).count_ones() as usize == slots
        } else {
            self.full::<P>(bits, slots)
        }
    }

    fn push_encoded<const P: usize, const EF: bool>(
        &mut self,
        value: u64,
        bits: usize,
        slots: usize,
        low_bits: usize,
    ) {
        if !EF {
            self.push::<P>(value, bits, slots);
            return;
        }
        let count = self.bitmap(bits, slots, low_bits).count_ones() as usize;
        self.set_encoded::<P, true>(count, bits, slots, low_bits, value);
    }

    #[inline(always)]
    fn contains_encoded<const P: usize, const SIMD: bool, const EF: bool>(
        &self,
        value: u64,
        bits: usize,
        slots: usize,
        low_bits: usize,
    ) -> bool {
        if !EF {
            return self.contains::<P, SIMD>(value, bits, slots);
        }
        if value == 0 {
            return false;
        }
        let count = self.bitmap(bits, slots, low_bits).count_ones() as usize;
        let occupied = (1u32 << count) - 1;
        let mut candidates = if P == 0 {
            occupied
        } else {
            self.candidates::<P, SIMD>(value, slots) & occupied
        };
        while candidates != 0 {
            let slot = candidates.trailing_zeros() as usize;
            if self.get_encoded::<P, true>(slot, bits, slots, low_bits) == value {
                return true;
            }
            candidates &= candidates - 1;
        }
        false
    }
}

pub struct CuckooSet<
    const MODE: u8,
    const PREFIX: usize = 16,
    const SIMD: bool = true,
    const EF: bool = false,
> {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[PackedBin]>,
    entry_bits: usize,
    slots: usize,
    low_bits: usize,
    len: usize,
}

impl<const MODE: u8, const PREFIX: usize, const SIMD: bool, const EF: bool>
    CuckooSet<MODE, PREFIX, SIMD, EF>
{
    pub fn new(slot_ratio: f32, keys: &[T]) -> Self {
        let mut this = Self::with_capacity(slot_ratio, keys.len());
        for &k in keys {
            this.insert(k);
        }
        this
    }
    fn with_capacity(slot_ratio: f32, n: usize) -> Self {
        assert!(slot_ratio >= 1.0);
        assert!(matches!(PREFIX, 0 | 8 | 16 | 32));
        assert!(
            cfg!(target_endian = "little"),
            "prefix SIMD layout requires little endian"
        );
        let capacity = (n as f64 * slot_ratio as f64).ceil() as usize;
        let (num_bins, entry_bits, slots, low_bits) = Self::table_layout(capacity);
        assert!(
            num_bins <= u32::MAX as usize,
            "prefetch tokens require 32-bit bucket indices"
        );
        Self {
            slot_ratio,
            num_bins,
            table: vec![PackedBin([0; 8]); num_bins].into_boxed_slice(),
            entry_bits,
            slots,
            low_bits,
            len: 0,
        }
    }

    fn bin_layout(entry_bits: usize) -> (usize, usize) {
        if !EF {
            return (512 / entry_bits, entry_bits);
        }
        // At most 16 lanes, matching the existing SIMD prefix filters. Choose
        // floor(log2(2^entry_bits / slots)) low bits, with prefix lanes intact.
        for slots in (1usize..=16).rev() {
            let high_bits = slots.next_power_of_two().ilog2() as usize;
            let low_bits = entry_bits.saturating_sub(high_bits).max(PREFIX);
            let bitmap_bits = (1usize << (entry_bits - low_bits)) + slots;
            if slots * low_bits + bitmap_bits <= 512 {
                return (slots, low_bits);
            }
        }
        unreachable!("entry does not fit in a cacheline")
    }

    fn table_layout(capacity: usize) -> (usize, usize, usize, usize) {
        let mut low = 2usize;
        let mut high = capacity.div_ceil(8).max(2);
        while low < high {
            let mid = low + (high - low) / 2;
            let bits = KEY_BITS as usize + 1 - mid.ilog2() as usize;
            let (slots, _) = Self::bin_layout(bits);
            if mid * slots >= capacity {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        let bits = KEY_BITS as usize + 1 - low.ilog2() as usize;
        let (slots, low_bits) = Self::bin_layout(bits);
        (low, bits, slots, low_bits)
    }

    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    // SplitMix-style permutation modulo 2^62. Odd multipliers remain invertible
    // modulo 2^62; masking each product keeps the permutation in that domain.
    #[inline(always)]
    fn permute(key: T, secondary: bool) -> T {
        // FIXME: Let's drop the multiplications
        let mut x = (key ^ if secondary { 0x9e3779b97f4a7c15 } else { 0 }) & KEY_MASK;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9) & KEY_MASK;
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb) & KEY_MASK;
        x ^ (x >> 31)
    }

    fn unpermute(mut x: T, secondary: bool) -> T {
        x ^= (x >> 31) ^ (x >> 62);
        x = x.wrapping_mul(0x319642b2d24d8ec3) & KEY_MASK;
        x ^= (x >> 27) ^ (x >> 54);
        x = x.wrapping_mul(0x96de1b173f119089) & KEY_MASK;
        x ^= (x >> 30) ^ (x >> 60);
        (x ^ if secondary { 0x9e3779b97f4a7c15 } else { 0 }) & KEY_MASK
    }

    #[inline(always)]
    fn product(&self, key: T, secondary: bool) -> u128 {
        ((Self::permute(key, secondary) << PADDING_BITS) as u128) * self.num_bins as u128
    }

    #[inline(always)]
    fn bin_idx_1(&self, key: T) -> usize {
        // FIXME: We should make this much simpler somehow.
        (self.product(key, false) >> 64) as usize
    }

    #[inline(always)]
    fn bin_idx_2(&self, key: T) -> usize {
        // FIXME: We should reuse the first hash as much as possible.
        (self.product(key, true) >> 64) as usize
    }

    #[inline(always)]
    fn entry(&self, key: T, secondary: bool) -> u64 {
        // The scaled hash has two known zero low bits. Valid products are
        // 4*m apart, so discard floor(log2(m))+2 bits without losing identity.
        (((self.product(key, secondary) as u64) >> (65 - self.entry_bits)) << 1) | secondary as u64
    }

    fn decode(&self, entry: u64, idx: usize) -> T {
        let lower_product =
            ((idx as u128) << 64) | (((entry >> 1) as u128) << (65 - self.entry_bits));
        // Valid products are multiples of 4*m. The truncated interval is no
        // wider than 4*m, so round up to its unique valid 62-bit hash value.
        // Division is used only for iteration/displacement.
        let hash = lower_product.div_ceil((self.num_bins as u128) << PADDING_BITS) as u64;
        Self::unpermute(hash, entry & 1 != 0)
    }

    #[inline(always)]
    fn matches(&self, idx: usize, key: T, secondary: bool) -> bool {
        key <= KEY_MASK
            && self.table[idx].contains_encoded::<PREFIX, SIMD, EF>(
                self.entry(key, secondary),
                self.entry_bits,
                self.slots,
                self.low_bits,
            )
    }

    #[inline(always)]
    pub fn prefetch_first(&self, key: T) -> usize {
        let bin_idx = self.bin_idx_1(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
        bin_idx
    }
    #[inline(always)]
    pub fn prefetch_second(&self, key: T) -> usize {
        let bin_idx = self.bin_idx_2(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
        bin_idx
    }
    #[inline(always)]
    pub fn prefetch_both(&self, key: T) -> usize {
        let b1 = self.prefetch_first(key);
        let b2 = self.prefetch_second(key);
        (b1 << 32) | (b2 as u32) as usize
    }
    #[inline(always)]
    pub fn prefetch(&self, key: T) -> usize {
        match Mode::from(MODE) {
            Mode::Eager => self.prefetch_both(key),
            Mode::Lazy => self.prefetch_first(key),
        }
    }

    #[inline(always)]
    pub fn contains_lazy(&self, key: T) -> bool {
        self.contains_lazy_with_token(key, self.bin_idx_1(key))
    }

    #[inline(always)]
    pub fn contains_eager(&self, key: T) -> bool {
        self.matches(self.bin_idx_1(key), key, false) | self.matches(self.bin_idx_2(key), key, true)
    }

    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        match Mode::from(MODE) {
            Mode::Eager => self.contains_eager(key),
            Mode::Lazy => self.contains_lazy(key),
        }
    }

    #[inline(always)]
    pub fn contains_lazy_with_token(&self, key: T, bin1: usize) -> bool {
        if key > KEY_MASK {
            return false;
        }
        // A zero primary encoding is always stored in its secondary bucket.
        if self.entry(key, false) == 0 {
            return self.matches(self.bin_idx_2(key), key, true);
        }
        if self.matches(bin1, key, false) {
            return true;
        }
        if !self.table[bin1].full_encoded::<PREFIX, EF>(self.entry_bits, self.slots, self.low_bits)
        {
            return false;
        }
        self.matches(self.bin_idx_2(key), key, true)
    }

    #[inline(always)]
    pub fn contains_eager_with_token(&self, key: T, bin: usize) -> bool {
        self.matches(bin >> 32, key, false) | self.matches(bin as u32 as usize, key, true)
    }

    #[inline(always)]
    pub fn contains_with_token(&self, key: T, token: usize) -> bool {
        match Mode::from(MODE) {
            Mode::Eager => self.contains_eager_with_token(key, token),
            Mode::Lazy => self.contains_lazy_with_token(key, token),
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: T) {
        assert_ne!(key, 0, "zero keys are not supported");
        assert!(key <= KEY_MASK, "keys must fit in 62 bits");
        if self.contains_eager(key) {
            return;
        }
        for secondary in [false, true] {
            let idx = if secondary {
                self.bin_idx_2(key)
            } else {
                self.bin_idx_1(key)
            };
            let entry = self.entry(key, secondary);
            if entry == 0 {
                continue;
            }
            if !self.table[idx].full_encoded::<PREFIX, EF>(
                self.entry_bits,
                self.slots,
                self.low_bits,
            ) {
                self.table[idx].push_encoded::<PREFIX, EF>(
                    entry,
                    self.entry_bits,
                    self.slots,
                    self.low_bits,
                );
                self.len += 1;
                return;
            }
        }
        let mut key = key;
        let mut secondary = false;
        loop {
            let idx = if secondary {
                self.bin_idx_2(key)
            } else {
                self.bin_idx_1(key)
            };
            let entry = self.entry(key, secondary);
            // A nonzero key can still have a zero primary quotient. Route it to
            // its secondary bucket, whose location bit makes the entry nonzero.
            if entry == 0 {
                secondary = true;
                continue;
            }
            if !self.table[idx].full_encoded::<PREFIX, EF>(
                self.entry_bits,
                self.slots,
                self.low_bits,
            ) {
                self.table[idx].push_encoded::<PREFIX, EF>(
                    entry,
                    self.entry_bits,
                    self.slots,
                    self.low_bits,
                );
                self.len += 1;
                return;
            }
            let slot = rand::random_range(0..self.slots);
            let displaced = self.table[idx].get_encoded::<PREFIX, EF>(
                slot,
                self.entry_bits,
                self.slots,
                self.low_bits,
            );
            self.table[idx].set_encoded::<PREFIX, EF>(
                slot,
                self.entry_bits,
                self.slots,
                self.low_bits,
                entry,
            );
            key = self.decode(displaced, idx);
            secondary = displaced & 1 == 0;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.table.iter().enumerate().flat_map(move |(idx, bin)| {
            (0..self.slots).filter_map(move |slot| {
                let entry =
                    bin.get_encoded::<PREFIX, EF>(slot, self.entry_bits, self.slots, self.low_bits);
                (entry != 0).then(|| self.decode(entry, idx))
            })
        })
    }

    pub fn test(&self) {
        for x in self.iter() {
            assert!(self.contains_eager(x));
        }
    }
}

impl<const MODE: u8, const PREFIX: usize, const SIMD: bool, const EF: bool> HashSet
    for CuckooSet<MODE, PREFIX, SIMD, EF>
{
    fn name(&self) -> &'static str {
        match (Mode::from(MODE), PREFIX, SIMD, EF) {
            (Mode::Eager, 0, _, false) => "CuckooQuotientSet<Eager,0,SWAR>",
            (Mode::Eager, 0, _, true) => "CuckooQuotientSet<Eager,0,SWAR,EF>",
            (Mode::Eager, 8, false, false) => "CuckooQuotientSet<Eager,8,SWAR>",
            (Mode::Eager, 8, false, true) => "CuckooQuotientSet<Eager,8,SWAR,EF>",
            (Mode::Eager, 8, true, false) => "CuckooQuotientSet<Eager,8,SIMD>",
            (Mode::Eager, 8, true, true) => "CuckooQuotientSet<Eager,8,SIMD,EF>",
            (Mode::Eager, 16, false, false) => "CuckooQuotientSet<Eager,16,SWAR>",
            (Mode::Eager, 16, false, true) => "CuckooQuotientSet<Eager,16,SWAR,EF>",
            (Mode::Eager, 16, true, false) => "CuckooQuotientSet<Eager,16,SIMD>",
            (Mode::Eager, 16, true, true) => "CuckooQuotientSet<Eager,16,SIMD,EF>",
            (Mode::Eager, 32, false, false) => "CuckooQuotientSet<Eager,32,SWAR>",
            (Mode::Eager, 32, false, true) => "CuckooQuotientSet<Eager,32,SWAR,EF>",
            (Mode::Eager, 32, true, false) => "CuckooQuotientSet<Eager,32,SIMD>",
            (Mode::Eager, 32, true, true) => "CuckooQuotientSet<Eager,32,SIMD,EF>",
            (Mode::Lazy, 0, _, false) => "CuckooQuotientSet<Lazy,0,SWAR>",
            (Mode::Lazy, 0, _, true) => "CuckooQuotientSet<Lazy,0,SWAR,EF>",
            (Mode::Lazy, 8, false, false) => "CuckooQuotientSet<Lazy,8,SWAR>",
            (Mode::Lazy, 8, false, true) => "CuckooQuotientSet<Lazy,8,SWAR,EF>",
            (Mode::Lazy, 8, true, false) => "CuckooQuotientSet<Lazy,8,SIMD>",
            (Mode::Lazy, 8, true, true) => "CuckooQuotientSet<Lazy,8,SIMD,EF>",
            (Mode::Lazy, 16, false, false) => "CuckooQuotientSet<Lazy,16,SWAR>",
            (Mode::Lazy, 16, false, true) => "CuckooQuotientSet<Lazy,16,SWAR,EF>",
            (Mode::Lazy, 16, true, false) => "CuckooQuotientSet<Lazy,16,SIMD>",
            (Mode::Lazy, 16, true, true) => "CuckooQuotientSet<Lazy,16,SIMD,EF>",
            (Mode::Lazy, 32, false, false) => "CuckooQuotientSet<Lazy,32,SWAR>",
            (Mode::Lazy, 32, false, true) => "CuckooQuotientSet<Lazy,32,SWAR,EF>",
            (Mode::Lazy, 32, true, false) => "CuckooQuotientSet<Lazy,32,SIMD>",
            (Mode::Lazy, 32, true, true) => "CuckooQuotientSet<Lazy,32,SIMD,EF>",
            _ => unreachable!("unsupported prefix width"),
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = CuckooSet::<MODE, PREFIX, SIMD, EF>::new(self.slot_ratio, keys);
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len() as f32 / (self.num_bins * self.slots) as f32
    }

    fn primary_hit_ratio(&self) -> Option<f32> {
        let mut total = 0usize;
        let mut primary = 0usize;
        for bin in &self.table {
            for slot in 0..self.slots {
                let entry =
                    bin.get_encoded::<PREFIX, EF>(slot, self.entry_bits, self.slots, self.low_bits);
                if entry != 0 {
                    total += 1;
                    primary += (entry & 1 == 0) as usize;
                }
            }
        }
        (total != 0).then(|| primary as f32 / total as f32)
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    #[inline(always)]
    fn prefetch(&self, key: T) -> usize {
        self.prefetch(key)
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

#[cfg(test)]
mod tests {
    use super::*;
    fn layout<const P: usize>() {
        assert_eq!(std::mem::size_of::<PackedBin>(), 64);
        assert_eq!(std::mem::align_of::<PackedBin>(), 64);
        for bits in 32..=64 {
            let slots = 512 / bits;
            let mask = u64::MAX >> (64 - bits);
            let mut bin = PackedBin([0; 8]);
            let values: Vec<_> = (0..slots)
                .map(|i| {
                    // Include nonzero entries with an all-zero prefix.
                    if i == 0 && P < bits {
                        1u64 << P
                    } else {
                        (CuckooSet::<0>::permute(i as u64, false) & mask).max(1)
                    }
                })
                .collect();
            for (slot, &value) in values.iter().enumerate() {
                assert!(!bin.full::<P>(bits, slots));
                bin.push::<P>(value, bits, slots);
                for (i, &expected) in values[..=slot].iter().enumerate() {
                    assert_eq!(bin.get::<P>(i, bits, slots), expected);
                    assert!(bin.contains::<P, false>(expected, bits, slots));
                    assert!(bin.contains::<P, true>(expected, bits, slots));
                }
                assert!(!bin.contains::<P, true>(0, bits, slots));
            }
            assert!(bin.full::<P>(bits, slots));
            for query in (0..1000).map(|x| CuckooSet::<0>::permute(x, true) & mask) {
                let expected = query != 0 && values.contains(&query);
                assert_eq!(bin.contains::<P, false>(query, bits, slots), expected);
                assert_eq!(bin.contains::<P, true>(query, bits, slots), expected);
                if P != 0 {
                    assert_eq!(
                        bin.candidates::<P, false>(query, slots),
                        bin.candidates::<P, true>(query, slots)
                    );
                }
            }
            for i in 0..slots {
                bin.set::<P>(i, bits, slots, mask);
            }
            for i in 0..slots {
                assert_eq!(bin.get::<P>(i, bits, slots), mask);
            }
        }
    }
    #[test]
    fn permutation_is_invertible_within_62_bits() {
        for key in [0, 1, KEY_MASK, 1 << (KEY_BITS - 1)]
            .into_iter()
            .chain((0..10_000u64).map(|x| x.wrapping_mul(0x9e3779b97f4a7c15) & KEY_MASK))
        {
            for secondary in [false, true] {
                let hash = CuckooSet::<0>::permute(key, secondary);
                assert!(hash <= KEY_MASK);
                assert_eq!(CuckooSet::<0>::unpermute(hash, secondary), key);
                assert_eq!(
                    CuckooSet::<0>::permute(CuckooSet::<0>::unpermute(key, secondary), secondary),
                    key
                );
            }
        }
    }

    #[test]
    fn fast_range_roundtrips_and_boundaries() {
        for m in (2usize..=257).chain([
            1023,
            1024,
            1025,
            (1 << 22) - 1,
            1 << 22,
            (1 << 22) + 1,
            1 << 31,
        ]) {
            let mut set = CuckooSet::<0>::new(1.1, &[]);
            // Encoding/decoding does not access the table: test large bucket
            // counts and bucket boundaries without allocating giant tables.
            set.num_bins = m;
            set.entry_bits = KEY_BITS as usize + 1 - m.ilog2() as usize;
            set.slots = 512 / set.entry_bits;
            for secondary in [false, true] {
                let boundaries = [0usize, 1, m / 2, m - 1].into_iter().flat_map(|idx| {
                    let first = ((idx as u128) << KEY_BITS).div_ceil(m as u128) as u64;
                    [
                        first.saturating_sub(1),
                        first,
                        first.saturating_add(1).min(KEY_MASK),
                    ]
                });
                for hash in boundaries
                    .chain([0, 1, KEY_MASK])
                    .chain((1..1000).map(|x| CuckooSet::<0>::permute(x, false)))
                {
                    let key = CuckooSet::<0>::unpermute(hash, secondary);
                    let idx = (set.product(key, secondary) >> 64) as usize;
                    let entry = set.entry(key, secondary);
                    assert_eq!(set.decode(entry, idx), key, "m={m}, hash={hash}");
                    assert!((entry as u128) < (1u128 << set.entry_bits));
                }
            }
        }
    }

    #[test]
    fn packed_layouts_and_prefix_masks() {
        layout::<0>();
        layout::<8>();
        layout::<16>();
        layout::<32>();
    }
    fn membership<const P: usize, const SIMD: bool, const EF: bool>() {
        type Expected = std::collections::HashSet<u64>;
        for n in [0usize, 1, 20, 1000, 20000] {
            let mut keys: Vec<_> = (1..=n as u64)
                .map(|x| CuckooSet::<0>::permute(x, false))
                .collect();
            keys.extend([KEY_MASK, 1 << (KEY_BITS - 1), KEY_MASK]);
            let expected: Expected = keys.iter().copied().collect();
            let mut set = CuckooSet::<0, P, SIMD, EF>::new(1.05, &keys);
            assert!(set.num_bins >= 2);
            // Force a nonzero key whose primary quotient is zero.
            let special = (1..set.num_bins)
                .map(|idx| set.decode(0, idx))
                .find(|&key| key != 0 && set.entry(key, false) == 0)
                .expect("nonzero key with a zero primary quotient");
            assert_ne!(special, 0);
            assert_eq!(set.entry(special, false), 0);
            set.insert(special);
            let mut expected = expected;
            expected.insert(special);
            assert_eq!(set.len(), expected.len());
            assert_eq!(set.iter().collect::<Expected>(), expected);
            let primary = expected
                .iter()
                .filter(|&&key| set.matches(set.bin_idx_1(key), key, false))
                .count();
            assert_eq!(
                set.primary_hit_ratio(),
                Some(primary as f32 / expected.len() as f32)
            );
            for k in expected
                .iter()
                .copied()
                .chain((1..20000).map(|x| CuckooSet::<0>::permute(x, true)))
                .chain([0, 1 << KEY_BITS, u64::MAX])
                .chain(expected.iter().copied().map(|key| key | (1 << KEY_BITS)))
            {
                let present = expected.contains(&k);
                assert_eq!(set.contains_eager(k), present);
                assert_eq!(set.contains_lazy(k), present);
                assert_eq!(
                    set.contains_eager_with_token(k, set.prefetch_both(k)),
                    present
                );
                assert_eq!(
                    set.contains_lazy_with_token(k, set.prefetch_first(k)),
                    present
                );
            }
        }
        let empty = CuckooSet::<1, P, SIMD, EF>::new(1.1, &[]);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.primary_hit_ratio(), None);
        assert!(!empty.contains(0));
    }
    #[test]
    fn unfiltered() {
        membership::<0, false, false>();
    }
    #[test]
    fn prefix8_scalar() {
        membership::<8, false, false>();
    }
    #[test]
    fn prefix8_simd() {
        membership::<8, true, false>();
    }
    #[test]
    fn prefix16_scalar() {
        membership::<16, false, false>();
    }
    #[test]
    fn prefix16_simd() {
        membership::<16, true, false>();
    }
    #[test]
    fn prefix32_scalar() {
        membership::<32, false, false>();
    }
    #[test]
    fn prefix32_simd() {
        membership::<32, true, false>();
    }
    fn ef_layout<const P: usize>() {
        for bits in 32..=62 {
            let (slots, low_bits) = CuckooSet::<0, P, true, true>::bin_layout(bits);
            let mut bin = PackedBin([0; 8]);
            let mut expected = Vec::new();
            let mask = u64::MAX >> (64 - bits);
            assert!(!bin.full_encoded::<P, true>(bits, slots, low_bits));
            for i in 0..slots {
                // Includes zero low parts and entries sharing their high part.
                let value = if i == 0 && low_bits < bits {
                    1 << low_bits
                } else {
                    (CuckooSet::<0>::permute((slots - i) as u64, false) & mask).max(1)
                };
                bin.push_encoded::<P, true>(value, bits, slots, low_bits);
                expected.push(value);
                expected.sort_unstable();
                assert_eq!(
                    bin.bitmap(bits, slots, low_bits).count_ones() as usize,
                    expected.len()
                );
                for (j, &entry) in expected.iter().enumerate() {
                    assert_eq!(bin.get_encoded::<P, true>(j, bits, slots, low_bits), entry);
                    assert!(bin.contains_encoded::<P, true, true>(entry, bits, slots, low_bits));
                    assert!(bin.contains_encoded::<P, false, true>(entry, bits, slots, low_bits));
                }
                for j in expected.len()..slots {
                    assert_eq!(bin.get_encoded::<P, true>(j, bits, slots, low_bits), 0);
                }
                assert!(!bin.contains_encoded::<P, true, true>(0, bits, slots, low_bits));
            }
            assert!(bin.full_encoded::<P, true>(bits, slots, low_bits));
            for slot in 0..slots {
                // Replacing a value can move it to another sorted rank.
                expected[slot] = mask;
                expected.sort_unstable();
                bin.set_encoded::<P, true>(slot, bits, slots, low_bits, mask);
                for (j, &entry) in expected.iter().enumerate() {
                    assert_eq!(bin.get_encoded::<P, true>(j, bits, slots, low_bits), entry);
                }
                assert_eq!(
                    bin.bitmap(bits, slots, low_bits).count_ones() as usize,
                    slots
                );
            }
            for value in (1..1000).map(|x| (CuckooSet::<0>::permute(x, true) & mask).max(1)) {
                assert_eq!(
                    bin.contains_encoded::<P, true, true>(value, bits, slots, low_bits),
                    expected.contains(&value)
                );
                assert_eq!(
                    bin.contains_encoded::<P, false, true>(value, bits, slots, low_bits),
                    expected.contains(&value)
                );
            }
        }
    }

    #[test]
    fn elias_fano_bucket_roundtrips_and_replacements() {
        ef_layout::<0>();
        ef_layout::<8>();
        ef_layout::<16>();
        ef_layout::<32>();
    }

    #[test]
    fn elias_fano_fits_thirteen_41_bit_entries() {
        assert_eq!(CuckooSet::<0, 16, true, true>::bin_layout(41), (13, 37));
        assert_eq!(13 * 37 + 16 + 13, 510);
        let capacity = ((1usize << 26) as f64 / 0.99).ceil() as usize;
        let (plain_bins, plain_bits, plain_slots, _) = CuckooSet::<0>::table_layout(capacity);
        let (ef_bins, ef_bits, ef_slots, ef_low) =
            CuckooSet::<0, 16, true, true>::table_layout(capacity);
        assert_eq!((plain_bits, plain_slots), (41, 12));
        assert_eq!((ef_bits, ef_slots, ef_low), (41, 13, 37));
        assert!(ef_bins < plain_bins);
    }

    #[test]
    fn ef_unfiltered() {
        membership::<0, false, true>();
    }
    #[test]
    fn ef_prefix8_scalar() {
        membership::<8, false, true>();
    }
    #[test]
    fn ef_prefix8_simd() {
        membership::<8, true, true>();
    }
    #[test]
    fn ef_prefix16_scalar() {
        membership::<16, false, true>();
    }
    #[test]
    fn ef_prefix16_simd() {
        membership::<16, true, true>();
    }
    #[test]
    fn ef_prefix32_scalar() {
        membership::<32, false, true>();
    }
    #[test]
    fn ef_prefix32_simd() {
        membership::<32, true, true>();
    }

    #[test]
    #[should_panic(expected = "keys must fit in 62 bits")]
    fn rejects_oversized_keys() {
        CuckooSet::<0>::new(1.1, &[1 << KEY_BITS]);
    }

    #[test]
    fn trait_rebuild_preserves_configuration() {
        let template = CuckooSet::<1, 8, false>::new(1.1, &[]);
        let rebuilt = HashSet::new(&template, &[1, 2, 3]);
        assert_eq!(rebuilt.name(), template.name());
        for key in [1, 2, 3] {
            assert!(rebuilt.contains(key));
        }
        assert!(!rebuilt.contains(4));
        let ef = CuckooSet::<1, 16, true, true>::new(1.1, &[]);
        let rebuilt = HashSet::new(&ef, &[1, 2, 3]);
        assert_eq!(rebuilt.name(), ef.name());
        assert!(rebuilt.contains(2));
        assert!(!rebuilt.contains(4));
    }

    #[test]
    #[should_panic(expected = "zero keys are not supported")]
    fn rejects_zero_key() {
        CuckooSet::<0>::new(1.1, &[0]);
    }
}

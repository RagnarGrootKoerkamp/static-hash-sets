//! Exact quotienting in 64-byte buckets, using fast-range multiplication.
//! The remaining hash bits and primary/secondary bit are split into contiguous
//! prefixes followed by bit-packed suffixes. Prefix filtering only selects
//! candidates: the complete entry is always checked, so membership is exact.
//!
//! The reversible hash accepts full-width nonzero u64 keys.
//! Zero keys are rejected. Zero entries pad unused slots; a zero primary encoding
//! is routed to the secondary bucket, where the location bit makes it nonzero.
//! `PREFIX` selects 8-, 16-, or 32-bit SIMD filtering (0 disables it).
//! The default is 16-bit SIMD filtering.
//! Set the fourth const parameter `EF=true` for sorted Elias–Fano buckets:
//! `CuckooSet::<0, 16, true>::new(slot_ratio, keys)`. Prefixes and suffixes
//! hold the low parts; a unary high-part bitmap also records occupancy. EF
//! buckets support up to 16 entries and are re-sorted on insertion/replacement.
//! Set the fifth const parameter to `true` for a 1-hot summary of keys placed
//! in secondary buckets. The summary occupies the last word of each bucket.
//! Bucket counts may be arbitrary: the high
//! half of hash*m selects the bucket, and the high bits of its low half form the
//! quotient. Lookup uses multiplication and shifts, with no integer division.
//!
//! NOTE: THIS FILE IS AI SLOP!!!

#[path = "cuckoo_bfs.rs"]
mod bfs;

pub use crate::cuckoo::Mode;
use crate::{T, traits::HashSet};

/// Bit width of the reversible hash domain.
pub const KEY_BITS: u32 = 64;
pub const KEY_MASK: u64 = u64::MAX >> (64 - KEY_BITS);
const PADDING_BITS: u32 = 64 - KEY_BITS;
const SECONDARY_SEED: u64 = 0x9e3779b97f4a7c15;
const MIX_MULTIPLIER: u64 = 0xbf58476d1ce4e5b9;
const MIX_INVERSE: u64 = 0x96de1b173f119089;

#[derive(Clone)]
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )),
    allow(dead_code)
)]
struct ByteMasks {
    width: usize,
    slots: usize,
    shuffle: [i8; 64],
    starts: u64,
    boundaries: u64,
}

impl ByteMasks {
    fn new(width: usize, data_bytes: usize) -> Self {
        debug_assert!((1..=8).contains(&width));
        let shuffle = std::array::from_fn(|i| (i % width) as i8);
        let mut starts = 0;
        for slot in 0..data_bytes / width {
            starts |= 1u64 << (slot * width);
        }
        Self {
            width,
            slots: data_bytes / width,
            shuffle,
            starts,
            boundaries: starts << width,
        }
    }
}

#[inline(always)]
#[cfg_attr(
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )),
    allow(dead_code)
)]
fn has_equal_byte_entry(equal: u64, masks: &ByteMasks) -> bool {
    let sum = equal.wrapping_add(masks.starts);
    let carry_into = sum ^ equal ^ masks.starts;
    (carry_into & masks.boundaries) != 0
}

/// One 64-byte bucket. With P = 0, entries are stored consecutively. Otherwise
/// the first `slots * P` bits hold slot prefixes, followed by their packed
/// suffixes in slot order. Elias–Fano stores only each entry's low bits in
/// that layout and appends a unary bitmap for high bits and occupancy. SIMD
/// compares prefixes before the full quotient is checked. Non-EF 32-bit-prefix
/// buckets with a spill filter store byte-aligned entries before the summary.
#[derive(Clone)]
#[repr(align(64))]
struct PackedBin<const P: usize, const FILTER: bool = false>([u64; 8]);

impl<const P: usize, const FILTER: bool> PackedBin<P, FILTER> {
    fn storage_bits(bits: usize) -> usize {
        if FILTER && P == 32 && bits <= 56 {
            if bits <= 48 { 48 } else { 56 }
        } else {
            bits
        }
    }

    #[inline(always)]
    fn get_bytes(&self, slot: usize, bits: usize) -> u64 {
        let bytes = bits.div_ceil(8);
        self.read(slot * bytes * 8, bits)
    }

    #[inline(always)]
    fn set_bytes(&mut self, slot: usize, bits: usize, value: u64) {
        let bytes = bits.div_ceil(8);
        self.write(slot * bytes * 8, bytes * 8, value);
    }

    #[inline(always)]
    fn contains_bytes(&self, value: u64, bits: usize, masks: &ByteMasks) -> bool {
        debug_assert_eq!(masks.width, bits.div_ceil(8));
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        return unsafe { self.contains_bytes_avx512(value, masks) };

        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )))]
        let slots = masks.slots;
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        )))]
        (0..slots).any(|slot| self.get_bytes(slot, bits) == value)
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn contains_bytes_avx512(&self, value: u64, masks: &ByteMasks) -> bool {
        use std::arch::x86_64::*;
        let repeated_value = _mm512_set1_epi64(value as i64);
        let shuffle = unsafe { _mm512_loadu_si512(masks.shuffle.as_ptr().cast()) };
        let target = _mm512_shuffle_epi8(repeated_value, shuffle);
        let data = unsafe { _mm512_loadu_si512(self.0.as_ptr().cast()) };
        let equal = _mm512_cmpeq_epi8_mask(data, target) as u64;
        has_equal_byte_entry(equal, masks)
    }

    #[inline(always)]
    fn read(&self, offset: usize, bits: usize) -> u64 {
        if bits == 0 {
            return 0;
        }
        let word = offset / 64;
        let shift = offset % 64;
        if FILTER {
            // The final word is reserved for the spill summary, so a second
            // word is always available for reads of packed entry data.
            let pair = self.0[word] as u128 | ((self.0.get(word + 1).copied().unwrap_or(0) as u128) << 64);
            return ((pair >> shift) as u64) & (u64::MAX >> (64 - bits));
        }
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
    fn get(&self, slot: usize, bits: usize, slots: usize) -> u64 {
        if P == 0 {
            return self.read(slot * bits, bits);
        }
        if FILTER && P == 32 && bits <= 48 {
            let prefix = unsafe { self.0.as_ptr().cast::<u32>().add(slot).read() } as u64;
            let suffix = unsafe {
                self.0
                    .as_ptr()
                    .cast::<u8>()
                    .add(slots * 4 + slot * 2)
                    .cast::<u16>()
                    .read_unaligned()
            } as u64;
            return prefix | (suffix << 32);
        }
        if FILTER && P == 32 && bits <= 56 {
            let prefix = unsafe { self.0.as_ptr().cast::<u32>().add(slot).read() } as u64;
            let suffix = unsafe {
                self.0
                    .as_ptr()
                    .cast::<u8>()
                    .add(slots * 4 + slot * 3)
                    .cast::<u32>()
                    .read_unaligned()
            } & 0x00ff_ffff;
            return prefix | ((suffix as u64) << 32);
        }
        let prefix = self.read(slot * P, P);
        prefix | (self.read(slots * P + slot * (bits - P), bits - P) << P)
    }

    fn set(&mut self, slot: usize, bits: usize, slots: usize, value: u64) {
        assert_ne!(value, 0, "zero is the empty-entry sentinel");
        self.write_value(slot, bits, slots, value);
    }

    fn write_value(&mut self, slot: usize, bits: usize, slots: usize, value: u64) {
        if P == 0 {
            self.write(slot * bits, bits, value);
            return;
        }
        if FILTER && P == 32 && bits <= 48 {
            self.write(slot * 32, 32, value as u32 as u64);
            self.write(slots * 32 + slot * 16, 16, value >> 32);
            return;
        }
        if FILTER && P == 32 && bits <= 56 {
            self.write(slot * 32, 32, value as u32 as u64);
            self.write(slots * 32 + slot * 24, 24, value >> 32);
            return;
        }
        self.write(slot * P, P, value & (u64::MAX >> (64 - P)));
        self.write(slots * P + slot * (bits - P), bits - P, value >> P);
    }

    #[inline(always)]
    fn full(&self, bits: usize, slots: usize) -> bool {
        self.get(slots - 1, bits, slots) != 0
    }

    fn push(&mut self, value: u64, bits: usize, slots: usize) {
        let slot = (0..slots)
            .find(|&i| self.get(i, bits, slots) == 0)
            .expect("full bucket");
        self.set(slot, bits, slots, value);
    }

    #[inline(always)]
    fn candidates(&self, value: u64, slots: usize) -> u32 {
        let mut mask = 0u32;
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
                if FILTER && slots <= 9 {
                    let lanes = unsafe { self.0.as_ptr().cast::<[u32; 8]>().read_unaligned() };
                    mask = wide::u32x8::from(lanes)
                        .simd_eq(wide::u32x8::splat(value as u32))
                        .to_bitmask() as u32;
                    if slots == 9 {
                        mask |= ((self.0[4] as u32 == value as u32) as u32) << 8;
                    }
                    return mask & ((1 << slots) - 1);
                }
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
        mask & ((1 << slots) - 1)
    }

    #[inline(always)]
    fn contains(&self, value: u64, bits: usize, slots: usize) -> bool {
        if value == 0 {
            return false;
        }
        if P == 0 {
            return (0..slots).any(|slot| self.get(slot, bits, slots) == value);
        }
        let mut mask = self.candidates(value, slots);
        if FILTER && P == 32 && bits <= 48 && slots >= 8 {
            let suffixes = unsafe {
                self.0
                    .as_ptr()
                    .cast::<u8>()
                    .add(slots * 4)
                    .cast::<[u16; 8]>()
                    .read_unaligned()
            };
            let mut suffix_mask = wide::u16x8::from(suffixes)
                .simd_eq(wide::u16x8::splat((value >> 32) as u16))
                .to_bitmask() as u32;
            if slots == 9 {
                suffix_mask |= (((self.get(8, bits, slots) >> 32) == (value >> 32)) as u32) << 8;
            }
            return mask & suffix_mask != 0;
        }
        if FILTER && P == 32 && bits <= 56 {
            let slot = (mask.trailing_zeros() as usize).min(slots - 1);
            let found = (mask != 0) & (self.get(slot, bits, slots) == value);
            mask &= mask.wrapping_sub(1);
            if mask == 0 {
                return found;
            }
            if found {
                return true;
            }
        }
        while mask != 0 {
            let slot = mask.trailing_zeros() as usize;
            if self.get(slot, bits, slots) == value {
                return true;
            }
            mask &= mask - 1;
        }
        false
    }
    #[inline(always)]
    fn bitmap(&self, bits: usize, slots: usize, low_bits: usize) -> u64 {
        self.read(
            slots * Self::storage_bits(low_bits),
            (1 << (bits - low_bits)) + slots,
        )
    }

    #[inline(always)]
    fn get_encoded<const EF: bool>(
        &self,
        slot: usize,
        bits: usize,
        slots: usize,
        low_bits: usize,
    ) -> u64 {
        if !EF {
            return self.get(slot, bits, slots);
        }
        let mut bitmap = self.bitmap(bits, slots, low_bits);
        if slot >= bitmap.count_ones() as usize {
            return 0;
        }
        for _ in 0..slot {
            bitmap &= bitmap - 1;
        }
        let high = bitmap.trailing_zeros() as usize - slot;
        ((high as u64) << low_bits) | self.get(slot, low_bits, slots)
    }

    fn set_encoded<const EF: bool>(
        &mut self,
        slot: usize,
        bits: usize,
        slots: usize,
        low_bits: usize,
        value: u64,
    ) {
        if !EF {
            self.set(slot, bits, slots, value);
            return;
        }
        assert_ne!(value, 0, "zero is the empty-entry sentinel");
        let mut bitmap = self.bitmap(bits, slots, low_bits);
        let count = bitmap.count_ones() as usize;
        assert!(slot < count || (slot == count && count < slots));
        let mut entries = [0u64; 16];
        for (i, entry) in entries[..count].iter_mut().enumerate() {
            let high = bitmap.trailing_zeros() as usize - i;
            *entry = ((high as u64) << low_bits) | self.get(i, low_bits, slots);
            bitmap &= bitmap - 1;
        }
        entries[slot] = value;
        let count = count + (slot == count) as usize;
        entries[..count].sort_unstable();
        let summary = self.0[7];
        self.0 = [0; 8];
        if FILTER {
            self.0[7] = summary;
        }
        let low_mask = u64::MAX >> (64 - low_bits);
        for (i, &entry) in entries[..count].iter().enumerate() {
            self.write_value(i, low_bits, slots, entry & low_mask);
            bitmap |= 1 << ((entry >> low_bits) as usize + i);
        }
        self.write(
            slots * Self::storage_bits(low_bits),
            (1 << (bits - low_bits)) + slots,
            bitmap,
        );
    }

    #[inline(always)]
    fn full_encoded<const EF: bool>(&self, bits: usize, slots: usize, low_bits: usize) -> bool {
        if EF {
            self.bitmap(bits, slots, low_bits).count_ones() as usize == slots
        } else {
            self.full(bits, slots)
        }
    }

    fn push_encoded<const EF: bool>(
        &mut self,
        value: u64,
        bits: usize,
        slots: usize,
        low_bits: usize,
    ) {
        if !EF {
            self.push(value, bits, slots);
            return;
        }
        let count = self.bitmap(bits, slots, low_bits).count_ones() as usize;
        self.set_encoded::<true>(count, bits, slots, low_bits, value);
    }

    #[inline(always)]
    fn contains_encoded<const EF: bool>(
        &self,
        value: u64,
        bits: usize,
        slots: usize,
        low_bits: usize,
    ) -> bool {
        if !EF {
            return self.contains(value, bits, slots);
        }
        if value == 0 {
            return false;
        }
        let count = self.bitmap(bits, slots, low_bits).count_ones() as usize;
        let occupied = (1u32 << count) - 1;
        let mut candidates = if P == 0 {
            occupied
        } else {
            self.candidates(value, slots) & occupied
        };
        while candidates != 0 {
            let slot = candidates.trailing_zeros() as usize;
            if self.get_encoded::<true>(slot, bits, slots, low_bits) == value {
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
    const EF: bool = false,
    const FILTER: bool = false,
    const BFS: bool = false,
    const FILTER32: bool = false,
> {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[PackedBin<PREFIX, FILTER>]>,
    entry_bits: usize,
    slots: usize,
    low_bits: usize,
    byte_masks: ByteMasks,
    len: usize,
    summary_ready: bool,
}

impl<const MODE: u8, const PREFIX: usize, const EF: bool, const FILTER: bool, const BFS: bool, const FILTER32: bool>
    CuckooSet<MODE, PREFIX, EF, FILTER, BFS, FILTER32>
{
    pub fn new(slot_ratio: f32, keys: &[T]) -> Self {
        let mut this = Self::with_capacity(slot_ratio, keys.len());
        if BFS {
            assert!(!EF, "BFS construction requires unsorted buckets");
            let mut workspace = bfs::BfsConstruction::new(this.num_bins);
            for &key in keys {
                this.insert_bfs(key, &mut workspace);
            }
        } else {
            for &k in keys {
                this.insert(k);
            }
        }
        if FILTER {
            this.rebuild_spill();
            this.summary_ready = true;
        }
        this
    }
    fn with_capacity(slot_ratio: f32, n: usize) -> Self {
        assert!(slot_ratio >= 1.0);
        assert!(!FILTER32 || (FILTER && PREFIX == 32 && !EF), "32-bit spill filter requires the byte layout");
        assert!(matches!(PREFIX, 0 | 8 | 16 | 32));
        assert!(
            cfg!(target_endian = "little"),
            "prefix SIMD layout requires little endian"
        );
        let capacity = (n as f64 * slot_ratio as f64).ceil() as usize;
        let (num_bins, entry_bits, slots, low_bits) = Self::table_layout(capacity);
        let byte_masks = ByteMasks::new(entry_bits.div_ceil(8), if FILTER32 { 60 } else { 56 });
        assert!(
            num_bins <= u32::MAX as usize,
            "prefetch tokens require 32-bit bucket indices"
        );
        Self {
            slot_ratio,
            num_bins,
            table: vec![PackedBin::<PREFIX, FILTER>([0; 8]); num_bins].into_boxed_slice(),
            entry_bits,
            slots,
            low_bits,
            byte_masks,
            len: 0,
            summary_ready: false,
        }
    }

    fn bin_layout(entry_bits: usize) -> (usize, usize) {
        if !EF {
            if FILTER && PREFIX == 32 {
                let entry_bytes = entry_bits.div_ceil(8);
                return ((if FILTER32 { 60 } else { 56 }) / entry_bytes, entry_bits);
            }
            return (
                (512 - 64 * FILTER as usize)
                    / PackedBin::<PREFIX, FILTER>::storage_bits(entry_bits),
                entry_bits,
            );
        }
        // At most 16 lanes, matching the existing SIMD prefix filters. Choose
        // floor(log2(2^entry_bits / slots)) low bits, with prefix lanes intact.
        for slots in (1usize..=16).rev() {
            let high_bits = slots.next_power_of_two().ilog2() as usize;
            let low_bits = entry_bits.saturating_sub(high_bits).max(PREFIX);
            let bitmap_bits = (1usize << (entry_bits - low_bits)) + slots;
            if slots * PackedBin::<PREFIX, FILTER>::storage_bits(low_bits) + bitmap_bits
                <= 512 - 64 * FILTER as usize
            {
                return (slots, low_bits);
            }
        }
        unreachable!("entry does not fit in a cacheline")
    }

    fn table_layout(capacity: usize) -> (usize, usize, usize, usize) {
        let mut low = 2usize;
        let mut high = capacity.div_ceil(if FILTER { 7 } else { 8 }).max(2);
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

    // One odd multiplication and two reversible xor-shifts mix all key bits.
    #[inline(always)]
    fn permute(key: T, secondary: bool) -> T {
        let x = key ^ if secondary { SECONDARY_SEED } else { 0 };
        let x = (x ^ (x >> 30)).wrapping_mul(MIX_MULTIPLIER);
        x ^ (x >> 27)
    }

    fn unpermute(mut x: T, secondary: bool) -> T {
        x ^= (x >> 27) ^ (x >> 54);
        x = x.wrapping_mul(MIX_INVERSE);
        x ^= (x >> 30) ^ (x >> 60);
        x ^ if secondary { SECONDARY_SEED } else { 0 }
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
        // Drop floor(log2(m)) low bits. Valid products are at least m apart.
        (((self.product(key, secondary) as u64) >> (65 - self.entry_bits)) << 1) | secondary as u64
    }

    fn decode(&self, entry: u64, idx: usize) -> T {
        let lower_product =
            ((idx as u128) << 64) | (((entry >> 1) as u128) << (65 - self.entry_bits));
        // Round up to the unique valid product in the truncated interval.
        // Division is used only for iteration/displacement.
        let hash = lower_product.div_ceil((self.num_bins as u128) << PADDING_BITS) as u64;
        Self::unpermute(hash, entry & 1 != 0)
    }

    #[inline(always)]
    fn matches(&self, idx: usize, key: T, secondary: bool) -> bool {
        key != 0 && self.contains_entry(idx, self.entry(key, secondary))
    }

    #[inline(always)]
    fn byte_layout(&self) -> bool {
        FILTER && PREFIX == 32 && !EF
    }

    #[inline(always)]
    fn get_entry(&self, idx: usize, slot: usize) -> u64 {
        if self.byte_layout() {
            self.table[idx].get_bytes(slot, self.entry_bits)
        } else {
            self.table[idx].get_encoded::<EF>(slot, self.entry_bits, self.slots, self.low_bits)
        }
    }

    #[inline(always)]
    fn set_entry(&mut self, idx: usize, slot: usize, value: u64) {
        if self.byte_layout() {
            self.table[idx].set_bytes(slot, self.entry_bits, value);
        } else {
            self.table[idx].set_encoded::<EF>(
                slot,
                self.entry_bits,
                self.slots,
                self.low_bits,
                value,
            );
        }
    }

    #[inline(always)]
    fn full_entry(&self, idx: usize) -> bool {
        if self.byte_layout() {
            self.table[idx].get_bytes(self.slots - 1, self.entry_bits) != 0
        } else {
            self.table[idx].full_encoded::<EF>(self.entry_bits, self.slots, self.low_bits)
        }
    }

    #[inline(always)]
    fn push_entry(&mut self, idx: usize, value: u64) {
        if self.byte_layout() {
            let slot = (0..self.slots)
                .find(|&slot| self.table[idx].get_bytes(slot, self.entry_bits) == 0)
                .expect("full bucket");
            self.table[idx].set_bytes(slot, self.entry_bits, value);
        } else {
            self.table[idx].push_encoded::<EF>(value, self.entry_bits, self.slots, self.low_bits);
        }
    }

    #[inline(always)]
    fn contains_entry(&self, idx: usize, value: u64) -> bool {
        if value == 0 {
            return false;
        }
        if !self.byte_layout() {
            return self.table[idx].contains_encoded::<EF>(
                value,
                self.entry_bits,
                self.slots,
                self.low_bits,
            );
        }
        self.table[idx].contains_bytes(value, self.entry_bits, &self.byte_masks)
    }

    #[inline(always)]
    fn spill_bits(hash: u64) -> u64 {
        if FILTER32 {
            let first = hash & 31;
            let second = (first + 1 + (hash >> 32) % 31) & 31;
            ((1u64 << first) | (1u64 << second)) << 32
        } else { 1u64 << (hash & 63) }
    }

    #[inline(always)]
    fn might_spill(&self, bin: usize, hash: u64) -> bool {
        let bits = Self::spill_bits(hash);
        self.table[bin].0[7] & bits == bits
    }

    fn rebuild_spill(&mut self) {
        for bin in self.table.iter_mut() {
            if FILTER32 { bin.0[7] &= u32::MAX as u64; } else { bin.0[7] = 0; }
        }
        for idx in 0..self.num_bins {
            for slot in 0..self.slots {
                let entry = self.get_entry(idx, slot);
                if entry != 0 && entry & 1 != 0 {
                    let key = self.decode(entry, idx);
                    let hash = Self::permute(key, false);
                    let primary = self.bin_idx_1(key);
                    self.table[primary].0[7] |= Self::spill_bits(hash);
                }
            }
        }
    }

    #[inline(always)]
    pub fn prefetch_first(&self, key: T) -> usize {
        let hash = Self::permute(key, false);
        let bin_idx = (((hash as u128) * self.num_bins as u128) >> 64) as usize;
        prefetch_index::prefetch_index(&self.table, bin_idx);
        // Carry the hash so lazy lookup can reuse it for the quotient and spill bit.
        hash as usize
    }
    #[inline(always)]
    pub fn prefetch_second(&self, key: T) -> usize {
        let bin_idx = self.bin_idx_2(key);
        prefetch_index::prefetch_index(&self.table, bin_idx);
        bin_idx
    }
    #[inline(always)]
    pub fn prefetch_both(&self, key: T) -> usize {
        let b1 = self.bin_idx_1(key);
        prefetch_index::prefetch_index(&self.table, b1);
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
        self.contains_lazy_with_token(key, Self::permute(key, false) as usize)
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
    fn contains_lazy_byte_layout(&self, key: T, hash: u64, bin1: usize, entry: u64) -> bool {
        if !self.might_spill(bin1, hash) {
            return entry != 0
                && self.table[bin1].contains_bytes(entry, self.entry_bits, &self.byte_masks);
        }

        let secondary_product = (Self::permute(key, true) as u128) * self.num_bins as u128;
        let bin2 = (secondary_product >> 64) as usize;
        let secondary_entry = (((secondary_product as u64) >> (65 - self.entry_bits)) << 1) | 1;
        let primary_match =
            entry != 0 && self.table[bin1].contains_bytes(entry, self.entry_bits, &self.byte_masks);
        primary_match
            | self.table[bin2].contains_bytes(secondary_entry, self.entry_bits, &self.byte_masks)
    }

    #[inline(always)]
    pub fn contains_lazy_with_token(&self, key: T, token: usize) -> bool {
        if key == 0 {
            return false;
        }
        let hash = token as u64;
        let product = (hash as u128) * self.num_bins as u128;
        let bin1 = (product >> 64) as usize;
        let entry = ((product as u64) >> (65 - self.entry_bits)) << 1;
        if self.byte_layout() {
            return self.contains_lazy_byte_layout(key, hash, bin1, entry);
        }
        // A zero primary encoding is always stored in its secondary bucket.
        if entry == 0 {
            return self.matches(self.bin_idx_2(key), key, true);
        }
        let primary_match = self.table[bin1].contains_encoded::<EF>(
            entry,
            self.entry_bits,
            self.slots,
            self.low_bits,
        );
        if FILTER {
            return if self.might_spill(bin1, hash) {
                primary_match | self.matches(self.bin_idx_2(key), key, true)
            } else {
                primary_match
            };
        }
        if primary_match {
            return true;
        }
        if !self.full_entry(bin1) {
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
        if self.contains_eager(key) {
            return;
        }
        if FILTER && self.summary_ready {
            for bin in self.table.iter_mut() {
                bin.0[7] |= if FILTER32 { (u32::MAX as u64) << 32 } else { u64::MAX };
            }
            self.summary_ready = false;
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
            if !self.full_entry(idx) {
                self.push_entry(idx, entry);
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
            if !self.full_entry(idx) {
                self.push_entry(idx, entry);
                self.len += 1;
                return;
            }
            let slot = rand::random_range(0..self.slots);
            let displaced = self.get_entry(idx, slot);
            self.set_entry(idx, slot, entry);
            key = self.decode(displaced, idx);
            secondary = displaced & 1 == 0;
        }
    }

    fn insert_bfs(&mut self, key: T, workspace: &mut bfs::BfsConstruction) {
        assert_ne!(key, 0, "zero keys are not supported");
        if self.contains_eager(key) {
            return;
        }
        let primary = self.bin_idx_1(key);
        let secondary = self.bin_idx_2(key);
        let roots = if self.entry(key, false) == 0 {
            vec![secondary]
        } else {
            vec![primary, secondary]
        };
        let path = workspace.path(&roots, self.slots, |bin, slot| {
            let entry = self.get_entry(bin, slot);
            if entry == 0 {
                return (true, None);
            }
            let occupant = self.decode(entry, bin);
            let alternate_secondary = entry & 1 == 0;
            let alternate = if alternate_secondary {
                self.bin_idx_2(occupant)
            } else {
                self.bin_idx_1(occupant)
            };
            (
                false,
                (self.entry(occupant, alternate_secondary) != 0).then_some(alternate),
            )
        });
        for pair in path.windows(2) {
            let (dst, ds) = pair[0];
            let (src, ss) = pair[1];
            let displaced = self.get_entry(src, ss);
            let occupant = self.decode(displaced, src);
            self.set_entry(dst, ds, self.entry(occupant, displaced & 1 == 0));
        }
        let &(root, slot) = path.last().unwrap();
        self.set_entry(
            root,
            slot,
            self.entry(key, root != primary || self.entry(key, false) == 0),
        );
        self.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.table.iter().enumerate().flat_map(move |(idx, _)| {
            (0..self.slots).filter_map(move |slot| {
                let entry = self.get_entry(idx, slot);
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

impl<const MODE: u8, const PREFIX: usize, const EF: bool, const FILTER: bool, const BFS: bool, const FILTER32: bool>
    HashSet for CuckooSet<MODE, PREFIX, EF, FILTER, BFS, FILTER32>
{
    fn name(&self) -> &'static str {
        if FILTER32 {
            assert!(BFS && MODE == Mode::Lazy as u8);
            return "CuckooQuotientSet<Lazy,32,SIMD,TwoHot32,BFS>";
        }
        if BFS {
            assert!(FILTER && PREFIX == 32 && !EF && MODE == Mode::Lazy as u8);
            return "CuckooQuotientSet<Lazy,32,SIMD,OneHot,BFS>";
        }
        if FILTER {
            return match (Mode::from(MODE), PREFIX, EF) {
                (Mode::Eager, 0, false) => "CuckooQuotientSet<Eager,0,SWAR,OneHot>",
                (Mode::Eager, 0, true) => "CuckooQuotientSet<Eager,0,SWAR,EF,OneHot>",
                (Mode::Lazy, 0, false) => "CuckooQuotientSet<Lazy,0,SWAR,OneHot>",
                (Mode::Lazy, 0, true) => "CuckooQuotientSet<Lazy,0,SWAR,EF,OneHot>",
                (Mode::Eager, 8, false) => "CuckooQuotientSet<Eager,8,SIMD,OneHot>",
                (Mode::Eager, 8, true) => "CuckooQuotientSet<Eager,8,SIMD,EF,OneHot>",
                (Mode::Lazy, 8, false) => "CuckooQuotientSet<Lazy,8,SIMD,OneHot>",
                (Mode::Lazy, 8, true) => "CuckooQuotientSet<Lazy,8,SIMD,EF,OneHot>",
                (Mode::Eager, 16, false) => "CuckooQuotientSet<Eager,16,SIMD,OneHot>",
                (Mode::Eager, 16, true) => "CuckooQuotientSet<Eager,16,SIMD,EF,OneHot>",
                (Mode::Lazy, 16, false) => "CuckooQuotientSet<Lazy,16,SIMD,OneHot>",
                (Mode::Lazy, 16, true) => "CuckooQuotientSet<Lazy,16,SIMD,EF,OneHot>",
                (Mode::Eager, 32, false) => "CuckooQuotientSet<Eager,32,SIMD,OneHot>",
                (Mode::Eager, 32, true) => "CuckooQuotientSet<Eager,32,SIMD,EF,OneHot>",
                (Mode::Lazy, 32, false) => "CuckooQuotientSet<Lazy,32,SIMD,OneHot>",
                (Mode::Lazy, 32, true) => "CuckooQuotientSet<Lazy,32,SIMD,EF,OneHot>",
                _ => unreachable!("unsupported prefix width"),
            };
        }
        match (Mode::from(MODE), PREFIX, EF) {
            (Mode::Eager, 0, false) => "CuckooQuotientSet<Eager,0,SWAR>",
            (Mode::Eager, 0, true) => "CuckooQuotientSet<Eager,0,SWAR,EF>",
            (Mode::Eager, 8, false) => "CuckooQuotientSet<Eager,8,SIMD>",
            (Mode::Eager, 8, true) => "CuckooQuotientSet<Eager,8,SIMD,EF>",
            (Mode::Eager, 16, false) => "CuckooQuotientSet<Eager,16,SIMD>",
            (Mode::Eager, 16, true) => "CuckooQuotientSet<Eager,16,SIMD,EF>",
            (Mode::Eager, 32, false) => "CuckooQuotientSet<Eager,32,SIMD>",
            (Mode::Eager, 32, true) => "CuckooQuotientSet<Eager,32,SIMD,EF>",
            (Mode::Lazy, 0, false) => "CuckooQuotientSet<Lazy,0,SWAR>",
            (Mode::Lazy, 0, true) => "CuckooQuotientSet<Lazy,0,SWAR,EF>",
            (Mode::Lazy, 8, false) => "CuckooQuotientSet<Lazy,8,SIMD>",
            (Mode::Lazy, 8, true) => "CuckooQuotientSet<Lazy,8,SIMD,EF>",
            (Mode::Lazy, 16, false) => "CuckooQuotientSet<Lazy,16,SIMD>",
            (Mode::Lazy, 16, true) => "CuckooQuotientSet<Lazy,16,SIMD,EF>",
            (Mode::Lazy, 32, false) => "CuckooQuotientSet<Lazy,32,SIMD>",
            (Mode::Lazy, 32, true) => "CuckooQuotientSet<Lazy,32,SIMD,EF>",
            _ => unreachable!("unsupported prefix width"),
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        let h = CuckooSet::<MODE, PREFIX, EF, FILTER, BFS, FILTER32>::new(self.slot_ratio, keys);
        Box::new(h)
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len() as f32 / (self.num_bins * self.slots) as f32
    }

    fn negative_passes_filter(&self, key: T) -> Option<bool> {
        if !FILTER || Mode::from(MODE) != Mode::Lazy {
            return None;
        }
        let primary = self.bin_idx_1(key);
        Some(self.might_spill(primary, Self::permute(key, false)))
    }

    fn primary_hit_ratio(&self) -> Option<f32> {
        let mut total = 0usize;
        let mut primary = 0usize;
        for (idx, _) in self.table.iter().enumerate() {
            for slot in 0..self.slots {
                let entry = self.get_entry(idx, slot);
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

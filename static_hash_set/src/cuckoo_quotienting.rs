//! Exact quotienting in 64-byte buckets, using fast-range multiplication.
//! The remaining hash bits and primary/secondary bit are split into contiguous
//! prefixes followed by bit-packed suffixes. Prefix filtering only selects
//! candidates: the complete entry is always checked, so membership is exact.
//!
//! The reversible hash accepts full-width nonzero u64 keys.
//! Zero keys are rejected. Zero entries pad unused slots; a zero primary encoding
//! is routed to the secondary bucket, where the location bit makes it nonzero.
//! `PREFIX` selects 8-, 16-, or 32-bit filtering (0 disables it); `SIMD=false`
//! selects u128 SWAR lane comparisons. The default is 16-bit SIMD filtering.
//! Set the fourth const parameter `EF=true` for sorted Elias–Fano buckets:
//! `CuckooSet::<0, 16, true, true>::new(slot_ratio, keys)`. Prefixes and suffixes
//! hold the low parts; a unary high-part bitmap also records occupancy. EF
//! buckets support up to 16 entries and are re-sorted on insertion/replacement.
//! Set the fifth const parameter to `true` for a 1-hot summary of keys placed
//! in secondary buckets. The summary occupies the last word of each bucket.
//! Bucket counts may be arbitrary: the high
//! half of hash*m selects the bucket, and the high bits of its low half form the
//! quotient. Lookup uses multiplication and shifts, with no integer division.
//!
//! NOTE: THIS FILE IS AI SLOP!!!

pub use crate::cuckoo::Mode;
use crate::{T, traits::HashSet};

/// Bit width of the reversible hash domain.
pub const KEY_BITS: u32 = 64;
pub const KEY_MASK: u64 = u64::MAX >> (64 - KEY_BITS);
const PADDING_BITS: u32 = 64 - KEY_BITS;
const SECONDARY_SEED: u64 = 0x9e3779b97f4a7c15;
const MIX_MULTIPLIER: u64 = 0xbf58476d1ce4e5b9;
const MIX_INVERSE: u64 = 0x96de1b173f119089;

/// One 64-byte bucket. With P = 0, entries are stored consecutively. Otherwise
/// the first `slots * P` bits hold slot prefixes, followed by their packed
/// suffixes in slot order. Elias–Fano stores only each entry's low bits in
/// that layout and appends a unary bitmap for high bits and occupancy. SIMD
/// changes how prefixes are compared, not where bits are stored.
#[derive(Clone)]
#[repr(align(64))]
struct PackedBin<const P: usize, const SIMD: bool, const FILTER: bool = false>([u64; 8]);

impl<const P: usize, const SIMD: bool, const FILTER: bool> PackedBin<P, SIMD, FILTER> {
    #[inline(always)]
    fn might_spill(&self, hash: u64) -> bool {
        self.0[7] & (1u64 << (hash & 63)) != 0
    }

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
    fn get(&self, slot: usize, bits: usize, slots: usize) -> u64 {
        if P == 0 {
            return self.read(slot * bits, bits);
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
    fn contains(&self, value: u64, bits: usize, slots: usize) -> bool {
        if value == 0 {
            return false;
        }
        if P == 0 {
            return (0..slots).any(|slot| self.get(slot, bits, slots) == value);
        }
        let mut mask = self.candidates(value, slots);
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
        self.read(slots * low_bits, (1 << (bits - low_bits)) + slots)
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
        self.write(slots * low_bits, (1 << (bits - low_bits)) + slots, bitmap);
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
    const SIMD: bool = true,
    const EF: bool = false,
    const FILTER: bool = false,
> {
    pub slot_ratio: f32,
    num_bins: usize,
    table: Box<[PackedBin<PREFIX, SIMD, FILTER>]>,
    entry_bits: usize,
    slots: usize,
    low_bits: usize,
    len: usize,
    summary_ready: bool,
}

impl<const MODE: u8, const PREFIX: usize, const SIMD: bool, const EF: bool, const FILTER: bool>
    CuckooSet<MODE, PREFIX, SIMD, EF, FILTER>
{
    pub fn new(slot_ratio: f32, keys: &[T]) -> Self {
        let mut this = Self::with_capacity(slot_ratio, keys.len());
        for &k in keys {
            this.insert(k);
        }
        if FILTER {
            this.rebuild_spill();
            this.summary_ready = true;
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
            table: vec![PackedBin::<PREFIX, SIMD, FILTER>([0; 8]); num_bins].into_boxed_slice(),
            entry_bits,
            slots,
            low_bits,
            len: 0,
            summary_ready: false,
        }
    }

    fn bin_layout(entry_bits: usize) -> (usize, usize) {
        if !EF {
            return ((512 - 64 * FILTER as usize) / entry_bits, entry_bits);
        }
        // At most 16 lanes, matching the existing SIMD prefix filters. Choose
        // floor(log2(2^entry_bits / slots)) low bits, with prefix lanes intact.
        for slots in (1usize..=16).rev() {
            let high_bits = slots.next_power_of_two().ilog2() as usize;
            let low_bits = entry_bits.saturating_sub(high_bits).max(PREFIX);
            let bitmap_bits = (1usize << (entry_bits - low_bits)) + slots;
            if slots * low_bits + bitmap_bits <= 512 - 64 * FILTER as usize {
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
        key != 0
            && self.table[idx].contains_encoded::<EF>(
                self.entry(key, secondary),
                self.entry_bits,
                self.slots,
                self.low_bits,
            )
    }

    fn rebuild_spill(&mut self) {
        for bin in self.table.iter_mut() {
            bin.0[7] = 0;
        }
        for idx in 0..self.num_bins {
            for slot in 0..self.slots {
                let entry = self.table[idx].get_encoded::<EF>(
                    slot,
                    self.entry_bits,
                    self.slots,
                    self.low_bits,
                );
                if entry != 0 && entry & 1 != 0 {
                    let key = self.decode(entry, idx);
                    let hash = Self::permute(key, false);
                    let primary = self.bin_idx_1(key);
                    self.table[primary].0[7] |= 1u64 << (hash & 63);
                }
            }
        }
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
        if key == 0 {
            return false;
        }
        // A zero primary encoding is always stored in its secondary bucket.
        if self.entry(key, false) == 0 {
            return self.matches(self.bin_idx_2(key), key, true);
        }
        if self.matches(bin1, key, false) {
            return true;
        }
        if FILTER && !self.table[bin1].might_spill(Self::permute(key, false)) {
            return false;
        }
        if !self.table[bin1].full_encoded::<EF>(self.entry_bits, self.slots, self.low_bits) {
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
                bin.0[7] = u64::MAX;
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
            if !self.table[idx].full_encoded::<EF>(self.entry_bits, self.slots, self.low_bits) {
                self.table[idx].push_encoded::<EF>(
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
            if !self.table[idx].full_encoded::<EF>(self.entry_bits, self.slots, self.low_bits) {
                self.table[idx].push_encoded::<EF>(
                    entry,
                    self.entry_bits,
                    self.slots,
                    self.low_bits,
                );
                self.len += 1;
                return;
            }
            let slot = rand::random_range(0..self.slots);
            let displaced =
                self.table[idx].get_encoded::<EF>(slot, self.entry_bits, self.slots, self.low_bits);
            self.table[idx].set_encoded::<EF>(
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
                let entry = bin.get_encoded::<EF>(slot, self.entry_bits, self.slots, self.low_bits);
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

impl<const MODE: u8, const PREFIX: usize, const SIMD: bool, const EF: bool, const FILTER: bool>
    HashSet for CuckooSet<MODE, PREFIX, SIMD, EF, FILTER>
{
    fn name(&self) -> &'static str {
        if FILTER {
            return match (Mode::from(MODE), PREFIX, SIMD, EF) {
                (Mode::Eager, 0, _, false) => "CuckooQuotientSet<Eager,0,SWAR,OneHot>",
                (Mode::Eager, 0, _, true) => "CuckooQuotientSet<Eager,0,SWAR,EF,OneHot>",
                (Mode::Lazy, 0, _, false) => "CuckooQuotientSet<Lazy,0,SWAR,OneHot>",
                (Mode::Lazy, 0, _, true) => "CuckooQuotientSet<Lazy,0,SWAR,EF,OneHot>",
                (Mode::Eager, 8, false, false) => "CuckooQuotientSet<Eager,8,SWAR,OneHot>",
                (Mode::Eager, 8, false, true) => "CuckooQuotientSet<Eager,8,SWAR,EF,OneHot>",
                (Mode::Lazy, 8, false, false) => "CuckooQuotientSet<Lazy,8,SWAR,OneHot>",
                (Mode::Lazy, 8, false, true) => "CuckooQuotientSet<Lazy,8,SWAR,EF,OneHot>",
                (Mode::Eager, 8, true, false) => "CuckooQuotientSet<Eager,8,SIMD,OneHot>",
                (Mode::Eager, 8, true, true) => "CuckooQuotientSet<Eager,8,SIMD,EF,OneHot>",
                (Mode::Lazy, 8, true, false) => "CuckooQuotientSet<Lazy,8,SIMD,OneHot>",
                (Mode::Lazy, 8, true, true) => "CuckooQuotientSet<Lazy,8,SIMD,EF,OneHot>",
                (Mode::Eager, 16, false, false) => "CuckooQuotientSet<Eager,16,SWAR,OneHot>",
                (Mode::Eager, 16, false, true) => "CuckooQuotientSet<Eager,16,SWAR,EF,OneHot>",
                (Mode::Lazy, 16, false, false) => "CuckooQuotientSet<Lazy,16,SWAR,OneHot>",
                (Mode::Lazy, 16, false, true) => "CuckooQuotientSet<Lazy,16,SWAR,EF,OneHot>",
                (Mode::Eager, 16, true, false) => "CuckooQuotientSet<Eager,16,SIMD,OneHot>",
                (Mode::Eager, 16, true, true) => "CuckooQuotientSet<Eager,16,SIMD,EF,OneHot>",
                (Mode::Lazy, 16, true, false) => "CuckooQuotientSet<Lazy,16,SIMD,OneHot>",
                (Mode::Lazy, 16, true, true) => "CuckooQuotientSet<Lazy,16,SIMD,EF,OneHot>",
                (Mode::Eager, 32, false, false) => "CuckooQuotientSet<Eager,32,SWAR,OneHot>",
                (Mode::Eager, 32, false, true) => "CuckooQuotientSet<Eager,32,SWAR,EF,OneHot>",
                (Mode::Lazy, 32, false, false) => "CuckooQuotientSet<Lazy,32,SWAR,OneHot>",
                (Mode::Lazy, 32, false, true) => "CuckooQuotientSet<Lazy,32,SWAR,EF,OneHot>",
                (Mode::Eager, 32, true, false) => "CuckooQuotientSet<Eager,32,SIMD,OneHot>",
                (Mode::Eager, 32, true, true) => "CuckooQuotientSet<Eager,32,SIMD,EF,OneHot>",
                (Mode::Lazy, 32, true, false) => "CuckooQuotientSet<Lazy,32,SIMD,OneHot>",
                (Mode::Lazy, 32, true, true) => "CuckooQuotientSet<Lazy,32,SIMD,EF,OneHot>",
                _ => unreachable!("unsupported prefix width"),
            };
        }
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
        let h = CuckooSet::<MODE, PREFIX, SIMD, EF, FILTER>::new(self.slot_ratio, keys);
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
        Some(self.table[primary].might_spill(Self::permute(key, false)))
    }

    fn primary_hit_ratio(&self) -> Option<f32> {
        let mut total = 0usize;
        let mut primary = 0usize;
        for bin in &self.table {
            for slot in 0..self.slots {
                let entry = bin.get_encoded::<EF>(slot, self.entry_bits, self.slots, self.low_bits);
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
        assert_eq!(std::mem::size_of::<PackedBin<P, false>>(), 64);
        assert_eq!(std::mem::align_of::<PackedBin<P, false>>(), 64);
        for bits in 32..=64 {
            let slots = 512 / bits;
            let mask = u64::MAX >> (64 - bits);
            let mut bin = PackedBin::<P, false>([0; 8]);
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
                assert!(!bin.full(bits, slots));
                bin.push(value, bits, slots);
                for (i, &expected) in values[..=slot].iter().enumerate() {
                    assert_eq!(bin.get(i, bits, slots), expected);
                    assert!(bin.contains(expected, bits, slots));
                    assert!(PackedBin::<P, true>(bin.0).contains(expected, bits, slots));
                }
                assert!(!PackedBin::<P, true>(bin.0).contains(0, bits, slots));
            }
            assert!(bin.full(bits, slots));
            for query in (0..1000).map(|x| CuckooSet::<0>::permute(x, true) & mask) {
                let expected = query != 0 && values.contains(&query);
                let simd_bin = PackedBin::<P, true>(bin.0);
                assert_eq!(bin.contains(query, bits, slots), expected);
                assert_eq!(simd_bin.contains(query, bits, slots), expected);
                if P != 0 {
                    assert_eq!(
                        bin.candidates(query, slots),
                        simd_bin.candidates(query, slots)
                    );
                }
            }
            for i in 0..slots {
                bin.set(i, bits, slots, mask);
            }
            for i in 0..slots {
                assert_eq!(bin.get(i, bits, slots), mask);
            }
        }
    }
    #[test]
    fn permutation_is_invertible_over_u64() {
        for key in [0, 1, KEY_MASK, 1 << (KEY_BITS - 1)]
            .into_iter()
            .chain((0..10_000u64).map(|x| x.wrapping_mul(0x9e3779b97f4a7c15) & KEY_MASK))
        {
            for secondary in [false, true] {
                let hash = CuckooSet::<0>::permute(key, secondary);
                assert_eq!(CuckooSet::<0>::unpermute(hash, secondary), key);
                assert_eq!(
                    CuckooSet::<0>::permute(CuckooSet::<0>::unpermute(key, secondary), secondary),
                    key
                );
            }
        }
    }

    #[test]
    fn sequential_keys_have_valid_placements() {
        let keys: Vec<_> = (1..=20_000).collect();
        let set = CuckooSet::<1, 32, true>::new(1.05, &keys);
        assert_eq!(set.len(), keys.len());
        assert!(keys.into_iter().all(|key| set.contains(key)));
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
        membership_with_filter::<P, SIMD, EF, false>();
    }

    fn membership_with_filter<
        const P: usize,
        const SIMD: bool,
        const EF: bool,
        const FILTER: bool,
    >() {
        type Expected = std::collections::HashSet<u64>;
        for n in [0usize, 1, 20, 1000, 20000] {
            let mut keys: Vec<_> = (1..=n as u64)
                .map(|x| CuckooSet::<0>::permute(x, false))
                .collect();
            keys.extend([KEY_MASK, 1 << (KEY_BITS - 1), KEY_MASK]);
            let expected: Expected = keys.iter().copied().collect();
            let mut set = CuckooSet::<0, P, SIMD, EF, FILTER>::new(1.05, &keys);
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
                .chain([0, 1 << (KEY_BITS - 1), u64::MAX])
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
        for bits in 32..=64 {
            let (slots, low_bits) = CuckooSet::<0, P, true, true>::bin_layout(bits);
            let mut bin = PackedBin::<P, false>([0; 8]);
            let mut expected = Vec::new();
            let mask = u64::MAX >> (64 - bits);
            assert!(!bin.full_encoded::<true>(bits, slots, low_bits));
            for i in 0..slots {
                // Includes zero low parts and entries sharing their high part.
                let value = if i == 0 && low_bits < bits {
                    1 << low_bits
                } else {
                    (CuckooSet::<0>::permute((slots - i) as u64, false) & mask).max(1)
                };
                bin.push_encoded::<true>(value, bits, slots, low_bits);
                expected.push(value);
                expected.sort_unstable();
                assert_eq!(
                    bin.bitmap(bits, slots, low_bits).count_ones() as usize,
                    expected.len()
                );
                for (j, &entry) in expected.iter().enumerate() {
                    assert_eq!(bin.get_encoded::<true>(j, bits, slots, low_bits), entry);
                    assert!(bin.contains_encoded::<true>(entry, bits, slots, low_bits));
                    assert!(
                        PackedBin::<P, true>(bin.0)
                            .contains_encoded::<true>(entry, bits, slots, low_bits)
                    );
                }
                for j in expected.len()..slots {
                    assert_eq!(bin.get_encoded::<true>(j, bits, slots, low_bits), 0);
                }
                assert!(
                    !PackedBin::<P, true>(bin.0).contains_encoded::<true>(0, bits, slots, low_bits)
                );
            }
            assert!(bin.full_encoded::<true>(bits, slots, low_bits));
            for slot in 0..slots {
                // Replacing a value can move it to another sorted rank.
                expected[slot] = mask;
                expected.sort_unstable();
                bin.set_encoded::<true>(slot, bits, slots, low_bits, mask);
                for (j, &entry) in expected.iter().enumerate() {
                    assert_eq!(bin.get_encoded::<true>(j, bits, slots, low_bits), entry);
                }
                assert_eq!(
                    bin.bitmap(bits, slots, low_bits).count_ones() as usize,
                    slots
                );
            }
            for value in (1..1000).map(|x| (CuckooSet::<0>::permute(x, true) & mask).max(1)) {
                assert_eq!(
                    PackedBin::<P, true>(bin.0)
                        .contains_encoded::<true>(value, bits, slots, low_bits),
                    expected.contains(&value)
                );
                assert_eq!(
                    bin.contains_encoded::<true>(value, bits, slots, low_bits),
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
    fn elias_fano_increases_bucket_capacity() {
        assert_eq!(CuckooSet::<0, 16, true, true>::bin_layout(41), (13, 37));
        assert_eq!(13 * 37 + 16 + 13, 510);
        let capacity = ((1usize << 26) as f64 / 0.99).ceil() as usize;
        let (plain_bins, plain_bits, plain_slots, _) = CuckooSet::<0>::table_layout(capacity);
        let (ef_bins, ef_bits, ef_slots, ef_low) =
            CuckooSet::<0, 16, true, true>::table_layout(capacity);
        assert_eq!((plain_bits, plain_slots), (43, 11));
        assert_eq!((ef_bits, ef_slots, ef_low), (43, 12, 39));
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
    fn one_hot_prefix16_simd() {
        membership_with_filter::<16, true, false, true>();
    }

    #[test]
    fn one_hot_ef_prefix16_simd() {
        membership_with_filter::<16, true, true, true>();
    }

    #[test]
    fn one_hot_prefix32_simd() {
        membership_with_filter::<32, true, false, true>();
    }

    #[test]
    fn one_hot_ef_prefix32_simd() {
        membership_with_filter::<32, true, true, true>();
    }

    #[test]
    fn one_hot_summary_tracks_secondary_entries() {
        let keys: Vec<_> = (1..=20_000u64)
            .map(|x| CuckooSet::<0>::permute(x, false))
            .collect();
        let mut set = CuckooSet::<1, 16, true, true, true>::new(1.05, &keys);
        assert!(set.summary_ready);
        let mut secondary = 0;
        for (idx, bin) in set.table.iter().enumerate() {
            for slot in 0..set.slots {
                let entry = bin.get_encoded::<true>(slot, set.entry_bits, set.slots, set.low_bits);
                if entry != 0 && entry & 1 != 0 {
                    secondary += 1;
                    let key = set.decode(entry, idx);
                    let primary = set.bin_idx_1(key);
                    assert!(set.table[primary].might_spill(CuckooSet::<0>::permute(key, false)));
                }
            }
        }
        assert!(secondary > 0);
        let rejected = (20_001..21_001u64)
            .map(|x| CuckooSet::<0>::permute(x, true))
            .filter(|&key| !set.contains(key))
            .filter(|&key| !set.negative_passes_filter(key).unwrap())
            .count();
        assert!(rejected > 0);
        let extra = u64::MAX - 1;
        set.insert(extra);
        assert!(set.contains_lazy(extra));
        assert!(set.table.iter().all(|bin| bin.0[7] == u64::MAX));
    }

    #[test]
    #[should_panic(expected = "zero keys are not supported")]
    fn rejects_zero_keys() {
        CuckooSet::<0>::new(1.1, &[0]);
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

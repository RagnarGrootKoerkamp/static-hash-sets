//! Static, cache-line bucketed braided cuckoo hashing.
//!
//! Five layers have relative sizes 16:8:4:2:1. Capacity is rounded only
//! to whole buckets in the smallest layer, targeting 90% occupancy.
//! The paper's cumulative
//! primary masses (20, 26, 29, 30.5) are normalized to probabilities,
//! giving weights 40:12:6:3; the last layer receives only secondary keys.
//! Greedy top-to-bottom placement is repaired with bounded augmenting paths
//! when forwarded keys collide. Repairs never leave a previously full bin
//! partially empty, which preserves lazy lookup termination.
//! Non-power-of-two layers use a mixed, lossy projection with four guard
//! address bits; this relaxes the paper's exact three-neighbor topology.
//!
//! NOTE: THIS FILE IS AI SLOP!!!

pub use crate::cuckoo::Mode;
use crate::{Bin, HashSet, BIN_SIZE, S, T};

const LAYERS: usize = 5;
const ATTEMPTS: u64 = 64;

/// One complete greedy phase; demand includes keys that could not be placed.
#[derive(Debug)]
pub struct LayerStats {
    pub layer: usize,
    pub primary_keys: usize,
    pub forwarded: usize,
    pub max_secondary_demand: usize,
    pub overloaded_secondary_bins: usize,
    pub unplaced: usize,
    pub max_primary_component_demand: Option<usize>,
    pub impossible_primary_components: Option<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct BuildStats {
    pub attempts: usize,
    pub repairs: usize,
    pub repair_visited: usize,
    pub max_repair_path: usize,
    pub failed_layer: Option<usize>,
    pub exhausted_component_bins: usize,
    pub search_limit_hit: bool,
}

#[derive(Debug)]
pub enum BuildError {
    CapacityOverflow,
    Placement(BuildStats),
}

pub struct BraidedCuckooSet<const MODE: u8> {
    table: Box<[Bin]>,
    offsets: [usize; LAYERS],
    smallest: usize,
    virtual_smallest_bits: u32,
    seed: u64,
    len: usize,
    has_zero: bool,
}

impl<const MODE: u8> BraidedCuckooSet<MODE> {
    pub fn new(keys: &[T]) -> Self {
        Self::try_new(keys).expect("braided cuckoo placement failed after 64 attempts")
    }

    pub fn try_new(keys: &[T]) -> Option<Self> {
        let _ = Mode::from(MODE);
        let mut keys = keys.to_vec();
        keys.sort_unstable();
        keys.dedup();
        Self::try_from_unique_keys(&keys, ATTEMPTS as usize, true)
            .ok()
            .map(|(set, _)| set)
    }

    /// Bucket counts at the requested 90% load, rounded only at the smallest
    /// layer. Small tables retain 256 buckets there to define all four braids.
    pub fn layer_bins_for_len(n: usize) -> Option<[usize; LAYERS]> {
        let smallest = n.checked_mul(10)?.div_ceil(9 * BIN_SIZE * 31).max(256);
        smallest.checked_mul(31)?.checked_mul(BIN_SIZE)?;
        Some(std::array::from_fn(|i| smallest << (4 - i)))
    }

    fn empty(n: usize, has_zero: bool) -> Option<Self> {
        let smallest = Self::layer_bins_for_len(n)?[4];
        let num_bins = smallest.checked_mul(31)?;
        Some(Self {
            table: vec![Bin([0; BIN_SIZE]); num_bins].into_boxed_slice(),
            offsets: std::array::from_fn(|i| smallest * (32 - (32 >> i))),
            smallest,
            virtual_smallest_bits: smallest.checked_next_power_of_two()?.ilog2(),
            seed: 0,
            len: n,
            has_zero,
        })
    }

    /// Fast construction for inputs known to contain distinct keys.
    /// Duplicates must be removed by the caller (use `try_new` otherwise).
    /// `repair=false` evaluates the paper's original greedy placement.
    pub fn try_from_unique_keys(
        keys: &[T],
        max_attempts: usize,
        repair: bool,
    ) -> Result<(Self, BuildStats), BuildError> {
        let _ = Mode::from(MODE);
        let mut this =
            Self::empty(keys.len(), keys.contains(&0)).ok_or(BuildError::CapacityOverflow)?;
        let mut stats = BuildStats::default();
        for attempt in 0..max_attempts {
            stats.attempts += 1;
            stats.failed_layer = None;
            this.seed = (attempt as u64).wrapping_mul(0x9e3779b97f4a7c15);
            if attempt != 0 {
                this.table.fill(Bin([0; BIN_SIZE]));
            }
            if this.place_with_repair(keys, repair, &mut stats) {
                return Ok((this, stats));
            }
        }
        Err(BuildError::Placement(stats))
    }

    /// Change lookup strategy without copying or rebuilding the table.
    pub fn into_mode<const NEW_MODE: u8>(self) -> BraidedCuckooSet<NEW_MODE> {
        let _ = Mode::from(NEW_MODE);
        BraidedCuckooSet {
            table: self.table,
            offsets: self.offsets,
            smallest: self.smallest,
            virtual_smallest_bits: self.virtual_smallest_bits,
            seed: self.seed,
            len: self.len,
            has_zero: self.has_zero,
        }
    }

    /// Diagnose one greedy attempt, stopping after the first failing layer.
    /// Unlike ordinary construction this finishes the phase to count every
    /// overflowing secondary bin, rather than just reporting the first one.
    pub fn diagnose(keys: &[T]) -> Vec<LayerStats> {
        let mut this = Self::empty(keys.len(), keys.contains(&0)).expect("capacity overflow");
        let smallest = this.smallest;
        let mut primary: [Vec<(T, usize)>; LAYERS - 1] = std::array::from_fn(|_| Vec::new());
        for &key in keys.iter().filter(|&&key| key != 0) {
            let (layer, idx) = this.primary(key);
            primary[layer].push((key, idx));
        }
        let mut report = Vec::new();
        for (layer, keys) in primary.iter().enumerate() {
            let next_size = smallest << (3 - layer);
            let mut demand = vec![0usize; next_size];
            let mut forwarded = 0;
            // Only the exact power-of-two braid has the small-component
            // capacity certificate. Projection can merge those components.
            let mut component_demand = if smallest.is_power_of_two() {
                vec![0usize; next_size / 4]
            } else {
                Vec::new()
            };
            for &(key, idx) in keys {
                if !component_demand.is_empty() {
                    let component = this.component((idx - this.offsets[layer]) >> 1, layer);
                    component_demand[component] += 1;
                }
                if this.table[idx].has_zero() {
                    this.table[idx].insert(key);
                } else {
                    let second = this.secondary(key, layer, idx);
                    demand[second - this.offsets[layer + 1]] += 1;
                    forwarded += 1;
                    if this.table[second].has_zero() {
                        this.table[second].insert(key);
                    }
                }
            }
            let unplaced = demand.iter().map(|&n| n.saturating_sub(BIN_SIZE)).sum();
            report.push(LayerStats {
                layer,
                primary_keys: keys.len(),
                forwarded,
                max_secondary_demand: *demand.iter().max().unwrap(),
                overloaded_secondary_bins: demand.iter().filter(|&&n| n > BIN_SIZE).count(),
                unplaced,
                max_primary_component_demand: component_demand.iter().copied().max(),
                impossible_primary_components: (!component_demand.is_empty()).then(|| {
                    component_demand
                        .iter()
                        .filter(|&&n| n > 12 * BIN_SIZE)
                        .count()
                }),
            });
            if unplaced != 0 {
                break;
            }
        }
        report
    }

    #[cfg(test)]
    fn place(&mut self, keys: &[T]) -> bool {
        self.place_with_repair(keys, false, &mut BuildStats::default())
    }

    fn place_with_repair(&mut self, keys: &[T], repair: bool, stats: &mut BuildStats) -> bool {
        // Group only keys, not (key, index) pairs: one extra AES hash is cheaper
        // than doubling the temporary memory traffic on large inputs.
        let weights = [40, 12, 6, 3];
        let mut primary: [Vec<T>; LAYERS - 1] = std::array::from_fn(|i| {
            let expected = keys.len().saturating_mul(weights[i]) / 61;
            Vec::with_capacity(expected + 8 * expected.isqrt() + 64)
        });
        for &key in keys.iter().filter(|&&key| key != 0) {
            primary[self.primary(key).0].push(key);
        }
        for (layer, keys) in primary.iter().enumerate() {
            // Pipeline the random cache-line fetches, reusing the computed
            // addresses when the key reaches the front of the pipeline.
            const AHEAD: usize = 32;
            let mut bins = [(0, 0); AHEAD];
            for (i, &key) in keys.iter().take(AHEAD).enumerate() {
                bins[i] = self.pair(key);
                prefetch_index::prefetch_index(&self.table, bins[i].0);
            }
            for (i, &key) in keys.iter().enumerate() {
                let (first, second) = bins[i % AHEAD];
                if i + AHEAD < keys.len() {
                    let next = self.pair(keys[i + AHEAD]);
                    prefetch_index::prefetch_index(&self.table, next.0);
                    bins[i % AHEAD] = next;
                }
                if self.push(first, key) || self.push(second, key) {
                    continue;
                }
                if !repair || !self.repair(key, first, second, stats) {
                    stats.failed_layer = Some(layer);
                    return false;
                }
            }
        }
        true
    }

    /// Packed bins let us test just the last slot and append without SIMD
    /// counting. Every insertion and repair preserves this packing.
    #[inline(always)]
    fn push(&mut self, idx: usize, key: T) -> bool {
        let bin = &mut self.table[idx];
        if bin.0[BIN_SIZE - 1] != 0 {
            return false;
        }
        let [lo, hi]: [S; 2] = unsafe { std::mem::transmute(*bin) };
        let empty = lo.simd_eq(S::ZERO).to_bitmask() | (hi.simd_eq(S::ZERO).to_bitmask() << 4);
        let slot = empty.trailing_zeros() as usize;
        bin.0[slot] = key;
        true
    }

    fn repair(&mut self, key: T, first: usize, second: usize, stats: &mut BuildStats) -> bool {
        // Breadth-first search for a free slot in the existing two-choice
        // graph. Full bins stay full: copy keys backward along the path,
        // overwriting their old slots, and insert the new key at the root.
        // The common repair needs only one or two edges.
        const MAX_VISITED: usize = 4096;
        let mut queue = Vec::with_capacity(32);
        queue.push((first, usize::MAX, 0usize));
        queue.push((second, usize::MAX, 0usize));
        let mut head = 0;
        while head < queue.len() {
            let bin = queue[head].0;
            stats.repair_visited += 1;
            for slot in 0..BIN_SIZE {
                let resident = self.table[bin].0[slot];
                let (first, second) = self.pair(resident);
                let other = if bin == first { second } else { first };
                if self.push(other, resident) {
                    let mut node = head;
                    let mut hole = slot;
                    let mut depth = 1;
                    while queue[node].1 != usize::MAX {
                        let (_, parent, parent_slot) = queue[node];
                        self.table[queue[node].0].0[hole] =
                            self.table[queue[parent].0].0[parent_slot];
                        node = parent;
                        hole = parent_slot;
                        depth += 1;
                    }
                    self.table[queue[node].0].0[hole] = key;
                    stats.repairs += 1;
                    stats.max_repair_path = stats.max_repair_path.max(depth);
                    return true;
                }
                if !queue.iter().any(|entry| entry.0 == other) {
                    if queue.len() < MAX_VISITED {
                        queue.push((other, head, slot));
                    } else {
                        stats.search_limit_hit = true;
                    }
                }
            }
            head += 1;
        }
        stats.exhausted_component_bins = queue.len();
        false
    }

    #[inline(always)]
    fn hash(&self, key: T) -> u128 {
        // One AES hash supplies independent bits for layer, position, braid.
        gxhash::gxhash128(&key.to_ne_bytes(), self.seed as i64)
    }

    #[inline(always)]
    fn primary_from_hash(&self, hash: u128) -> (usize, usize) {
        let draw = (((hash >> 96) as u64 * 61) >> 32) as u32;
        let layer = (draw >= 40) as usize + (draw >= 52) as usize + (draw >= 58) as usize;
        let size = self.smallest << (4 - layer);
        let local = ((hash as u64 as u128 * size as u128) >> 64) as usize;
        (layer, self.offsets[layer] + local)
    }

    #[inline(always)]
    fn primary(&self, key: T) -> (usize, usize) {
        self.primary_from_hash(self.hash(key))
    }

    #[inline(always)]
    fn secondary_from_hash(&self, hash: u128, layer: usize, _primary: usize) -> usize {
        // Apply the paper's braid to the high bits of the same hash that
        // selected the physical primary bin, in a virtual power-of-two layer.
        // For non-power-of-two layers, mix the braided address with a
        // multiplicative permutation before range reduction. This lossy
        // projection joins formerly separate virtual braid components.
        // Four guard bits avoid collapsing many keys onto the same small
        // set of virtual neighbors before projection. Consequently the
        // non-power-of-two variant relaxes the paper's exact 3-neighbor graph.
        // Exact power-of-two layers retain the paper's mapping unchanged.
        let guard_bits = if self.smallest.is_power_of_two() {
            0
        } else {
            4
        };
        let remaining_bits = self.virtual_smallest_bits + 3 - layer as u32 + guard_bits;
        let virtual_local = (hash as u64) >> (64 - remaining_bits);
        let shift = remaining_bits - 2 * (layer as u32 + 1);
        let change = 1 + ((((hash >> 64) as u32 as u64) * 3) >> 32);
        let braided = virtual_local ^ (change << shift);
        let physical_size = self.smallest << (3 - layer);
        let local = if physical_size.is_power_of_two() {
            braided as usize
        } else {
            let mixed = Self::projected_hash(braided, layer);
            ((mixed as u128 * physical_size as u128) >> 64) as usize
        };
        self.offsets[layer + 1] + local
    }

    #[inline(always)]
    fn projected_hash(address: u64, layer: usize) -> u64 {
        // A plain scaling or multiply preserves small dense components.
        // Two multiply/xor rounds scatter virtual addresses before the
        // intentionally lossy range reduction. The layer salt separates
        // the four projections.
        let mut x = address ^ (layer as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    // Exact-component identifiers are valid only without quantization.
    fn component(&self, local: usize, layer: usize) -> usize {
        debug_assert!(self.smallest.is_power_of_two());
        let shift = self.virtual_smallest_bits + 3 - layer as u32 - 2 * (layer as u32 + 1);
        (local & ((1 << shift) - 1)) | ((local >> (shift + 2)) << shift)
    }

    #[inline(always)]
    fn secondary(&self, key: T, layer: usize, primary: usize) -> usize {
        self.secondary_from_hash(self.hash(key), layer, primary)
    }

    #[inline(always)]
    fn pair(&self, key: T) -> (usize, usize) {
        let hash = self.hash(key);
        let (layer, first) = self.primary_from_hash(hash);
        (first, self.secondary_from_hash(hash, layer, first))
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table)
    }

    #[inline(always)]
    pub fn contains_lazy(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let hash = self.hash(key);
        let (layer, first) = self.primary_from_hash(hash);
        let bin = &self.table[first];
        let keys = S::splat(key as _);
        if bin.contains(keys) {
            return true;
        }
        if bin.0[BIN_SIZE - 1] == 0 {
            return false;
        }
        self.table[self.secondary_from_hash(hash, layer, first)].contains(keys)
    }

    #[inline(always)]
    pub fn contains_eager(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let (first, second) = self.pair(key);
        let keys = S::splat(key as _);
        self.table[first].contains(keys) | self.table[second].contains(keys)
    }

    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        match Mode::from(MODE) {
            Mode::Lazy => self.contains_lazy(key),
            Mode::Eager => self.contains_eager(key),
        }
    }

    /// Pack both indices on practical tables. Very large tables fall back
    /// to a primary index token instead of truncating either index.
    #[inline(always)]
    pub fn prefetch(&self, key: T) -> usize {
        let (first, second) = self.pair(key);
        prefetch_index::prefetch_index(&self.table, first);
        if Mode::from(MODE) == Mode::Eager {
            prefetch_index::prefetch_index(&self.table, second);
        }
        if usize::BITS == 64 && self.table.len() <= u32::MAX as usize {
            (first << 32) | second
        } else {
            first
        }
    }

    #[inline(always)]
    pub fn contains_with_token(&self, key: T, token: usize) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let (first, second) = if usize::BITS == 64 && self.table.len() <= u32::MAX as usize {
            (token >> 32, token as u32 as usize)
        } else {
            let layer = self.offsets.partition_point(|&offset| offset <= token) - 1;
            (token, self.secondary(key, layer, token))
        };
        let keys = S::splat(key as _);
        let bin = &self.table[first];
        match Mode::from(MODE) {
            Mode::Eager => bin.contains(keys) | self.table[second].contains(keys),
            Mode::Lazy => {
                if bin.contains(keys) {
                    return true;
                }
                if bin.0[BIN_SIZE - 1] == 0 {
                    return false;
                }
                self.table[second].contains(keys)
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        std::iter::repeat_n(0, self.has_zero as usize).chain(
            self.table
                .iter()
                .flat_map(|bin| bin.0)
                .filter(|&key| key != 0),
        )
    }

    pub fn test(&self) {
        assert_eq!(self.iter().count(), self.len());
        for key in self.iter() {
            assert!(self.contains(key));
        }
    }
}

impl<const MODE: u8> HashSet for BraidedCuckooSet<MODE> {
    fn name(&self) -> &'static str {
        match Mode::from(MODE) {
            Mode::Lazy => "BraidedCuckooSet<Lazy>",
            Mode::Eager => "BraidedCuckooSet<Eager>",
        }
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(Self::new(keys))
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len as f32 / (self.table.len() * BIN_SIZE) as f32
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    fn prefetch(&self, key: T) -> usize {
        self.prefetch(key)
    }
    fn contains(&self, key: T) -> bool {
        self.contains(key)
    }
    fn contains_with_token(&self, key: T, token: usize) -> bool {
        self.contains_with_token(key, token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<const MODE: u8>(keys: &[T]) {
        let set = BraidedCuckooSet::<MODE>::new(keys);
        let expected: std::collections::HashSet<_> = keys.iter().copied().collect();
        assert_eq!(set.len(), expected.len());
        assert_eq!(
            set.iter().collect::<std::collections::HashSet<_>>(),
            expected
        );
        assert!(set.load_factor() <= 0.9);
        for key in keys.iter().copied().chain(1_000_000..1_010_000) {
            let found = expected.contains(&key);
            assert_eq!(set.contains_lazy(key), found);
            assert_eq!(set.contains_eager(key), found);
            assert_eq!(set.contains_with_token(key, set.prefetch(key)), found);
        }
        for layer in 0..4 {
            let size = set.offsets[layer + 1] - set.offsets[layer];
            assert_eq!(size, (set.table.len() - set.offsets[4]) << (4 - layer));
        }
        for key in keys.iter().copied().filter(|&key| key != 0) {
            let (layer, first) = set.primary(key);
            let second = set.secondary(key, layer, first);
            if set.smallest.is_power_of_two() {
                let before = (first - set.offsets[layer]) >> 1;
                let after = second - set.offsets[layer + 1];
                let shift = set.virtual_smallest_bits + 3 - layer as u32 - 2 * (layer as u32 + 1);
                assert!((1..=3).contains(&((before ^ after) >> shift)));
                assert_eq!((before ^ after) & !(3 << shift), 0);
            }
            assert!(
                second
                    < set
                        .offsets
                        .get(layer + 2)
                        .copied()
                        .unwrap_or(set.table.len())
            );
        }
        set.test();
    }

    #[test]
    fn arbitrary_sizes_and_braid_boundaries() {
        for smallest in [256usize, 257, 263, 511, 512, 513] {
            let n = (smallest - 1) * 9 * BIN_SIZE * 31 / 10 + 1;
            let set = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::empty(n, false).unwrap();
            assert_eq!(set.smallest, smallest);
            for layer in 0..4 {
                let next_size = smallest << (3 - layer);
                let guard_bits = if smallest.is_power_of_two() { 0 } else { 4 };
                let bits = set.virtual_smallest_bits + 4 - layer as u32 + guard_bits;
                for virtual_primary in 0..(1usize << bits) {
                    for change in 1..=3usize {
                        let hash_lo = (virtual_primary as u64) << (64 - bits);
                        let hash = (((change - 1) as u128 * (1u128 << 32)).div_ceil(3) << 64)
                            | hash_lo as u128;
                        let primary = set.offsets[layer]
                            + ((hash_lo as u128 * (next_size * 2) as u128) >> 64) as usize;
                        let second =
                            set.secondary_from_hash(hash, layer, primary) - set.offsets[layer + 1];
                        assert!(second < next_size);
                        let shift = bits - 1 - 2 * (layer as u32 + 1);
                        let braided = (virtual_primary >> 1) ^ (change << shift);
                        if !smallest.is_power_of_two() {
                            let mixed = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::projected_hash(
                                braided as u64,
                                layer,
                            );
                            assert_eq!(
                                second,
                                ((mixed as u128 * next_size as u128) >> 64) as usize
                            );
                        }
                        if smallest.is_power_of_two() {
                            assert_eq!(
                                second,
                                ((primary - set.offsets[layer]) >> 1) ^ (change << shift)
                            );
                        }
                    }
                }
            }
        }

        let n = 1 << 26;
        let sizes = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::layer_bins_for_len(n).unwrap();
        let capacity = sizes.iter().sum::<usize>() * BIN_SIZE;
        let load = n as f64 / capacity as f64;
        assert!(load <= 0.9 && load > 0.89999);
        assert!(!sizes[0].is_power_of_two());
        for i in 1..5 {
            assert_eq!(sizes[i - 1], 2 * sizes[i]);
        }
    }

    #[test]
    fn overloaded_secondary_reports_failure() {
        let mut set = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::new(&[]);
        let mut groups = std::collections::HashMap::<_, Vec<T>>::new();
        let colliding = (1..)
            .find_map(|key| {
                let pair = set.pair(key);
                let group = groups.entry(pair).or_default();
                group.push(key);
                (group.len() == 2 * BIN_SIZE + 1).then(|| group.clone())
            })
            .unwrap();
        assert!(!set.place(&colliding));
    }

    #[test]
    fn impossible_component_is_reported() {
        let template = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::empty(97, false).unwrap();
        // More than 96 primary keys in the same eight-primary/four-secondary
        // component cannot fit, regardless of the displacement algorithm.
        let keys: Vec<_> = (1..)
            .filter(|&key| {
                let (layer, idx) = template.primary(key);
                layer == 0 && template.component(idx >> 1, layer) == 0
            })
            .take(97)
            .collect();
        let report = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::diagnose(&keys);
        assert_eq!(report[0].impossible_primary_components, Some(1));
        match BraidedCuckooSet::<{ Mode::Lazy as u8 }>::try_from_unique_keys(&keys, 1, true) {
            Err(BuildError::Placement(stats)) => {
                assert_eq!(stats.failed_layer, Some(0));
                assert!(stats.exhausted_component_bins <= 12);
                assert!(!stats.search_limit_hit);
            }
            _ => panic!("an over-capacity component must fail"),
        }
    }

    #[test]
    fn multi_edge_repair_rewires_the_entire_path() {
        let mut set = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::empty(25, false).unwrap();
        let key = (1..).find(|&key| set.primary(key).0 == 0).unwrap();
        let (a, b) = set.pair(key);
        let ab: Vec<_> = (1..)
            .filter(|&key| set.pair(key) == (a, b))
            .take(9)
            .collect();
        let key = (1..).find(|&key| set.primary(key) == (1, b)).unwrap();
        let (_, c) = set.pair(key);
        let bc: Vec<_> = (1..)
            .filter(|&key| set.pair(key) == (b, c))
            .take(8)
            .collect();
        let key = (1..).find(|&key| set.primary(key) == (2, c)).unwrap();
        let (_, d) = set.pair(key);
        let cd: Vec<_> = (1..)
            .filter(|&key| set.pair(key) == (c, d))
            .take(8)
            .collect();
        for &key in &ab[..8] {
            assert!(set.push(a, key));
        }
        for &key in &bc {
            assert!(set.push(b, key));
        }
        for &key in &cd {
            assert!(set.push(c, key));
        }
        let mut stats = BuildStats::default();
        assert!(set.repair(ab[8], a, b, &mut stats));
        assert_eq!(stats.max_repair_path, 2);
        assert_eq!(set.table[d].len(), 1);
        for key in ab.into_iter().chain(bc).chain(cd) {
            assert!(set.contains_lazy(key));
            assert!(set.contains_eager(key));
        }
        set.test();
    }

    #[test]
    fn repairs_preserve_lazy_termination() {
        let keys: Vec<_> = (0..45_000).collect();
        let (set, stats) =
            BraidedCuckooSet::<{ Mode::Lazy as u8 }>::try_from_unique_keys(&keys, 1, true).unwrap();
        assert!(stats.repairs > 0);
        for (idx, bin) in set.table.iter().enumerate() {
            let occupied = bin.0.iter().take_while(|&&key| key != 0).count();
            assert!(bin.0[occupied..].iter().all(|&key| key == 0));
            for key in bin.0.iter().copied().filter(|&key| key != 0) {
                let (first, second) = set.pair(key);
                assert!(idx == first || idx == second);
                if idx == second {
                    assert!(!set.table[first].has_zero());
                }
            }
        }
        for key in keys {
            assert!(set.contains_lazy(key));
        }
    }

    #[test]
    fn membership_and_braid() {
        for keys in [
            vec![],
            vec![0, 0, 1, 1, u64::MAX],
            (0..20_000).collect(),
            (0..45_000).collect(),
            (0..90_000).collect(),
            (0..25_000u64)
                .map(|x| x.wrapping_mul(0x9e3779b97f4a7c15))
                .collect(),
        ] {
            check::<{ Mode::Lazy as u8 }>(&keys);
            check::<{ Mode::Eager as u8 }>(&keys);
        }
    }
}

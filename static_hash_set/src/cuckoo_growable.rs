//! Prototype of rightward, incrementally growable bucketed cuckoo hashing.
use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasher, BuildHasherDefault};

use crate::bin::Bin;
use crate::traits::HashSet;
use crate::{BIN_SIZE, S, T};

type Hasher = BuildHasherDefault<gxhash::GxHasher>;
const SECOND_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

pub struct GrowableRightCuckooSet {
    pub slot_ratio: f32,
    table: Vec<Bin>,
    stash: Vec<T>,
    len: usize,
    has_zero: bool,
}

impl GrowableRightCuckooSet {
    pub fn new(slot_ratio: f32, keys: &[T]) -> Self {
        assert!(slot_ratio >= 1.0);
        let bins = ((keys.len() as f32 * slot_ratio).ceil() as usize)
            .div_ceil(BIN_SIZE)
            .max(1);
        let mut set = Self {
            slot_ratio,
            table: vec![Bin([0; BIN_SIZE]); bins],
            stash: Vec::new(),
            len: 0,
            has_zero: false,
        };
        for &key in keys {
            set.insert(key);
        }
        set
    }

    /// Linear hashing: an append splits precisely bucket `N - floor_pow2(N)`.
    #[inline]
    fn linear_bucket(hash: u64, n: usize) -> usize {
        let base = 1usize << (usize::BITS - 1 - n.leading_zeros());
        let split = n - base;
        let low = hash as usize & (base - 1);
        if low < split && (hash as usize & base) != 0 {
            low + base
        } else {
            low
        }
    }

    #[inline]
    fn candidates_at(key: T, n: usize) -> (usize, usize) {
        let h1 = Hasher::default().hash_one(key);
        let h2 = Hasher::default().hash_one(key ^ SECOND_SEED);
        let a = Self::linear_bucket(h1, n);
        let b = Self::linear_bucket(h2, n);
        (a.min(b), a.max(b))
    }

    #[inline]
    fn candidates(&self, key: T) -> (usize, usize) {
        Self::candidates_at(key, self.table.len())
    }

    pub fn contains(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let (first, second) = self.candidates(key);
        let needle = S::splat(key as _);
        self.table[first].contains(needle)
            || (first != second && self.table[second].contains(needle))
            || self.stash.contains(&key)
    }

    /// Search an acyclic, rightward augmenting path. Leave the table unchanged on failure.
    fn try_place(&mut self, key: T) -> bool {
        let (first, second) = self.candidates(key);
        for bucket in [first, second] {
            if let Some(slot) = self.table[bucket].0.iter().position(|&x| x == 0) {
                self.table[bucket].0[slot] = key;
                return true;
            }
        }
        // Bound work independently of table size. A failed search uses the growth/stash path.
        const SEARCH_BINS: usize = 256;
        let mut queue = VecDeque::from([0usize]);
        let mut nodes = vec![(first, None::<(usize, usize)>)];
        let mut seen = HashMap::from([(first, 0usize)]);
        if second != first {
            seen.insert(second, 1);
            nodes.push((second, None));
            queue.push_back(1);
        }
        let mut free = None;
        while let Some(node) = queue.pop_front() {
            let bucket = nodes[node].0;
            if let Some(slot) = self.table[bucket].0.iter().position(|&x| x == 0) {
                free = Some((node, slot));
                break;
            }
            for slot in 0..BIN_SIZE {
                let resident = self.table[bucket].0[slot];
                let (primary, secondary) = self.candidates(resident);
                if primary == bucket
                    && secondary > bucket
                    && !seen.contains_key(&secondary)
                    && nodes.len() < SEARCH_BINS
                {
                    let child = nodes.len();
                    seen.insert(secondary, child);
                    nodes.push((secondary, Some((node, slot))));
                    queue.push_back(child);
                }
            }
        }
        let Some((mut node, mut hole)) = free else {
            return false;
        };
        while let Some((previous, slot)) = nodes[node].1 {
            let bucket = nodes[node].0;
            let previous_bucket = nodes[previous].0;
            self.table[bucket].0[hole] = self.table[previous_bucket].0[slot];
            node = previous;
            hole = slot;
        }
        self.table[nodes[node].0].0[hole] = key;
        true
    }

    /// The new bucket can inherit keys only from one old bucket in each hash map.
    pub fn grow_one(&mut self) {
        let old_n = self.table.len();
        let base = 1usize << (usize::BITS - 1 - old_n.leading_zeros());
        let split = old_n - base;
        self.table.push(Bin([0; BIN_SIZE]));
        let mut keep = [0; BIN_SIZE];
        let mut moved = [0; BIN_SIZE];
        let mut ki = 0;
        let mut mi = 0;
        for &key in &self.table[split].0 {
            if key == 0 {
                continue;
            }
            let (a, b) = self.candidates(key);
            if a == split || b == split {
                keep[ki] = key;
                ki += 1;
            } else {
                debug_assert!(a == old_n || b == old_n);
                moved[mi] = key;
                mi += 1;
            }
        }
        self.table[split] = Bin(keep);
        self.table[old_n] = Bin(moved);
        let mut index = 0;
        while index < self.stash.len() {
            if self.try_place(self.stash[index]) {
                self.stash.swap_remove(index);
            } else {
                index += 1;
            }
        }
    }

    pub fn insert(&mut self, key: T) {
        if key == 0 {
            if !self.has_zero {
                self.has_zero = true;
                self.len += 1;
            }
            return;
        }
        if self.contains(key) {
            return;
        }
        if !self.try_place(key) {
            self.grow_one();
            if !self.try_place(key) {
                // The next linear-hashing split need not touch this key's component.
                self.stash.push(key);
            }
        }
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_bins(&self) -> usize {
        self.table.len()
    }

    pub fn stash_len(&self) -> usize {
        self.stash.len()
    }

    pub fn allocation_size(&self) -> usize {
        self.table.capacity() * std::mem::size_of::<Bin>()
            + self.stash.capacity() * std::mem::size_of::<T>()
    }
}

impl HashSet for GrowableRightCuckooSet {
    fn name(&self) -> &'static str {
        "GrowableRightCuckooSet"
    }
    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(Self::new(self.slot_ratio, keys))
    }
    fn allocation_size(&self) -> usize {
        self.allocation_size()
    }
    fn load_factor(&self) -> f32 {
        self.len as f32 / (self.table.len() * BIN_SIZE) as f32
    }
    fn contains(&self, key: T) -> bool {
        self.contains(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_changes_only_one_bucket_and_rightward_lookup_survives_growth() {
        for n in 1..80 {
            let mut set = GrowableRightCuckooSet::new(1.2, &[]);
            while set.num_bins() < n {
                set.grow_one();
            }
            let base = 1usize << (usize::BITS - 1 - n.leading_zeros());
            let split = n - base;
            for key in 1..1000u64 {
                let before = set.candidates(key);
                let after = GrowableRightCuckooSet::candidates_at(key, n + 1);
                for old in [before.0, before.1] {
                    if old != after.0 && old != after.1 {
                        assert_eq!(old, split);
                    }
                }
            }
        }
        let keys: Vec<u64> = (1..=25_000).map(|i| i * 0x9e37_79b9).collect();
        let mut set = GrowableRightCuckooSet::new(1.2, &keys);
        for &key in &keys {
            assert!(set.contains(key));
        }
        for _ in 0..100 {
            set.grow_one();
        }
        for &key in &keys {
            assert!(set.contains(key));
        }
        assert!(!set.contains(u64::MAX));
    }

    #[test]
    fn incremental_insertions_match_reference_set() {
        let mut set = GrowableRightCuckooSet::new(1.2, &[]);
        let mut reference = std::collections::HashSet::new();
        for i in 0..10_000u64 {
            let key = i.wrapping_mul(0xd6e8_feb8_6659_fd93);
            set.insert(key);
            reference.insert(key);
            if i % 347 == 0 {
                set.grow_one();
            }
        }
        for key in &reference {
            assert!(set.contains(*key), "missing {key}");
        }
        assert_eq!(set.len(), reference.len());
        for i in 10_000..12_000u64 {
            assert!(!set.contains(i.wrapping_mul(0xd6e8_feb8_6659_fd93)));
        }
    }
}

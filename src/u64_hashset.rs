//! A dense_hash_set for u64 keys.
//!
//! Compared to std::collections::HashSet<u64>, this uses a different layout: no metadata table, just plain data.
//! This is similar to Google's dense_hash_map, which predates the SwissTable design. By avoiding a metadata table,
//! we may need to do longer probe sequences (each probe is 8 bytes, not 1 byte), but on the other hand we only take
//! 1 cache miss per access, not 2.

use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::hash::{BuildHasher, BuildHasherDefault};
use std::hint::select_unpredictable;
use std::mem::transmute;
// type S = wide::i64x4;
type S = wide::i32x8;

use rustc_hash::FxHashMap;
use wide::CmpEq;

use crate::T;

type Hasher = BuildHasherDefault<rustc_hash::FxHasher>;

pub struct U64HashSet {
    pub slot_ratio: f32,
    buckets: usize,
    table: Box<[Bucket]>,
    len: usize,
    has_zero: bool,
    hits: usize,
    skips: usize,
    skips2: usize,
    probelen: FxHashMap<usize, usize>,
    last_b: usize,
    last_j: usize,
    last_empty: usize,
}

const PADDING: usize = 1000;

impl IntoIterator for &U64HashSet {
    type Item = T;

    type IntoIter = impl Iterator<Item = T>;

    fn into_iter(self) -> Self::IntoIter {
        std::iter::repeat_n(0, self.has_zero as usize).chain(
            self.table
                .iter()
                .enumerate()
                .flat_map(|(i, b)| {
                    if i < 5 {
                        eprintln!("bucket {i}")
                    };
                    b.0.iter().copied()
                })
                .filter(|x| *x != 0),
        )
    }
}

const BUCKET_SIZE: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(align(64))] // Cache line alignment
struct Bucket([T; BUCKET_SIZE]);

impl U64HashSet {
    pub fn new(slot_ratio: f32, keys: &[T]) -> Self {
        let mut this = Self::with_capacity(slot_ratio, keys.len());
        for &k in keys {
            this.insert(k);
        }
        this
    }
    fn with_capacity(slot_ratio: f32, n: usize) -> Self {
        let capacity = (n as f32 * slot_ratio).ceil() as usize;
        // TODO: integer overflow...
        let buckets = capacity.div_ceil(BUCKET_SIZE);
        // eprintln!("#buckets: {buckets}, with padding {}", buckets + PADDING);
        let table = vec![Bucket([0 as T; BUCKET_SIZE]); buckets + PADDING].into_boxed_slice();
        Self {
            slot_ratio,
            buckets,
            table,
            len: 0,
            has_zero: false,
            hits: 0,
            skips: 0,
            skips2: 0,
            probelen: Default::default(),
            last_b: 0,
            last_j: 0,
            last_empty: 0,
        }
    }

    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len + self.has_zero as usize
    }

    pub fn test(&self) {
        for x in self {
            if !self.contains(x) {
                eprintln!("Did not find {x}!");
                let hash64 = Hasher::default().hash_one(x);
                let bucket_i = (hash64 as usize).widening_mul(self.buckets).1;

                let [h1, h2]: &[S; 2] = unsafe { transmute(&self.table[bucket_i]) };
                let c0 = h1.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
                let c1 = h2.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
                let elems = BUCKET_SIZE - c0 - c1;
                eprintln!("Intended bucket {bucket_i} of size {elems}");
                let flat = unsafe { self.table.align_to::<T>().1 };
                let pos = flat.iter().position(|y| *y == x);
                eprintln!("Found at {pos:?}");
                if let Some(p) = pos {
                    let bucket = p / BUCKET_SIZE;
                    eprintln!("Actual bucket {bucket}");
                }

                panic!();
            }
        }
    }

    pub fn stats(&self) {
        let mut counts = [0; 9];

        eprintln!("Size    : {}", self.len);
        eprintln!("hits    : {}", self.hits);
        eprintln!("Skips   : {}", self.skips);
        eprintln!("Skips/el: {}", self.skips as f32 / self.hits as f32);
        eprintln!("Skips2/el {}", self.skips2 as f32 / self.hits as f32);
        // return;

        let mut sum = 0;
        let mut cnt = 0;
        for bucket in &self.table {
            let [h1, h2]: &[S; 2] = unsafe { transmute(&bucket.0) };
            let c0 = h1.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
            let c1 = h2.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
            let elems = BUCKET_SIZE - c0 - c1;
            counts[elems] += 1;
        }
        for i in 0..=8 {
            eprintln!("{i}: {:>9}", counts[i]);
        }
        eprintln!("buckets {cnt}");
        eprintln!("slots   {}", cnt * BUCKET_SIZE);
        eprintln!("sum {sum}");
        eprintln!("avg {}", sum as f32 / cnt as f32);

        let mut probes: Vec<(_, _)> = self.probelen.iter().collect();
        probes.sort();
        for (len, count) in probes {
            eprintln!("{len:>4} => {count:>9}");
        }
        self.test();
    }

    #[inline(always)]
    pub fn prefetch(&self, key: T) {
        let hash64 = Hasher::default().hash_one(key);
        let bucket_i = (hash64 as usize).widening_mul(self.buckets).1;
        // Safety: bucket_mask is correct because the number of buckets is a power of 2.
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(
                self.table.get_unchecked(bucket_i) as *const Bucket as *const i8
            );
        }
    }

    #[inline(always)]
    pub fn contains(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let hash64 = Hasher::default().hash_one(key);
        let mut bucket_i = (hash64 as usize).widening_mul(self.buckets).1;

        let keys = S::splat(key as _);

        let mut i = 1;
        loop {
            use std::mem::transmute;
            // Safety: bucket_mask is correct because the number of buckets is a power of 2.
            let bucket = unsafe { self.table.get_unchecked(bucket_i) };
            let [h1, h2]: [S; 2] = unsafe { transmute(bucket.0) };
            let has_key = (h1.cmp_eq(keys) | h2.cmp_eq(keys)).move_mask() as u8;
            if has_key > 0 {
                return true;
            }
            let has_zero = (h1.cmp_eq(S::ZERO) | h2.cmp_eq(S::ZERO)).move_mask() as u8;
            if has_zero > 0 {
                return false;
            }

            bucket_i += 1;
            i += 1;
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: T) {
        if key == 0 {
            self.len += !self.has_zero as usize;
            self.has_zero = true;
            return;
        }
        let hash64 = Hasher::default().hash_one(key);
        let mut bucket_i = (hash64 as usize).widening_mul(self.buckets).1;

        let keys = S::splat(key as _);

        let mut i = 1;
        loop {
            // Safety: bucket_mask is correct because the number of buckets is a power of 2.
            let bucket = &mut self.table[bucket_i];
            let [h1, h2]: &[S; 2] = unsafe { transmute(&bucket.0) };

            let has_key = (h1.cmp_eq(keys) | h2.cmp_eq(keys)).move_mask() as u8;
            if has_key > 0 {
                return;
            }

            let c0 = h1.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
            let c1 = h2.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
            let taken = BUCKET_SIZE - c0 - c1;

            if taken < BUCKET_SIZE {
                let element_i = taken;
                let element = &mut bucket.0[element_i];
                debug_assert!(*element == 0);
                *element = key;
                self.hits += 1;
                self.len += 1;
                return;
            }

            bucket_i += 1;
            i += 1;
            self.skips += 1;
        }
    }

    #[inline(always)]
    pub fn insert_in_order(&mut self, key: T) {
        self.len += 1;
        assert!(self.len <= self.buckets * BUCKET_SIZE);
        if key == 0 {
            self.has_zero = true;
            return;
        }
        let hash64 = Hasher::default().hash_one(key);
        let bucket_i = (hash64 as usize).widening_mul(self.buckets).1;
        // same bucket?
        self.last_j = select_unpredictable(bucket_i > self.last_b, 0, self.last_j + 1);
        self.last_b = self.last_b.max(bucket_i);
        self.last_b += self.last_j >> 3;
        self.last_j &= 7;
        self.hits += 1;
        self.skips += self.last_b - bucket_i;
        self.skips2 += (self.last_b - bucket_i).pow(2);
        // if self.len % (1024 * 1024) == 0 {
        //     eprintln!("elem {}: {} => {} of {}", self.len, bucket_i, self.last_b, self.buckets);
        // }
        self.table[self.last_b].0[self.last_j] = key;
    }

    pub fn stabilize(&self) -> Self {
        let mut i = 0;
        loop {
            let mut new = Self::with_capacity(self.slot_ratio, self.len);
            let start = std::time::Instant::now();
            eprintln!("Self:");
            for x in (&self).into_iter().take(20) {
                eprintln!("{x}");
            }
            for x in self {
                new.insert_in_order(x);
            }
            eprintln!("New:");
            for x in (&new).into_iter().take(20) {
                eprintln!("{x}");
            }
            eprintln!("{i:>2}: Re-build took {:?}", start.elapsed());
            i += 1;
            if self.table == new.table {
                eprintln!("DONe");
                return new;
            }
        }
    }
}

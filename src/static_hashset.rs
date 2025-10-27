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

use super::{to_bin, to_part, T};

pub struct StaticHashSet<const PROBE: bool> {
    pub slot_ratio: f32,
    pub meta_ratio: f32,
    k: usize,
    s: usize,
    n: usize,
    b: usize,
    m: usize,
    p: usize,
    table: Box<[Bucket]>,
    seeds: Vec<u8>,
    len: usize,
    has_zero: bool,
    hits: usize,
    none: usize,
    skips: usize,
    probelen: FxHashMap<usize, usize>,
    last_b: usize,
    last_j: usize,
    last_empty: usize,
}

const PADDING: usize = 100;

const BUCKET_SIZE: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(align(64))] // Cache line alignment
struct Bucket([T; BUCKET_SIZE]);

impl<const PROBE: bool> IntoIterator for &StaticHashSet<PROBE> {
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

impl<const PROBE: bool> StaticHashSet<PROBE> {
    pub fn new(slot_ratio: f32, meta_ratio: f32, keys: &[T]) -> Self {
        // eprintln!("building..");
        let k = BUCKET_SIZE;
        let s = 8;
        let n = keys.len();
        // buckets
        let b = ((n as f32 * slot_ratio) as usize).div_ceil(k);
        // total metadata bits
        let m = ((n * 64) as f32 * meta_ratio) as usize;
        // bits per part
        let p = m.div_ceil(s.max(1));

        // 1. sort the keys
        let mut keys = keys.to_vec();
        keys.sort_unstable();

        // 2. split into b even parts
        let mut part_sizes = vec![0; p];
        let shift = 64 - p.trailing_zeros();
        for key in &*keys {
            let p = to_part(*key, p, k);
            part_sizes[p] += 1;
        }
        let mut part_starts = vec![0; p + 1];
        for i in 1..=p {
            part_starts[i] = part_starts[i - 1] + part_sizes[i - 1];
        }

        // 3. Sort parts by decreasing size
        let mut perm = (0..p).collect::<Vec<_>>();
        perm.sort_by_key(|&i| std::cmp::Reverse(part_sizes[i]));

        // 4. init buckets
        let mut table = vec![Bucket([0; BUCKET_SIZE]); b + PADDING].into_boxed_slice();
        let mut seeds = vec![0u8; p];
        eprintln!();
        eprintln!("Size of table: {}", std::mem::size_of_val(&*table));
        eprintln!("Size of seeds: {}", std::mem::size_of_val(&*seeds));

        let mut bucket_sizes = vec![0u8; b];
        let mut collisions = vec![];
        // eprintln!("find seeds..");
        for (idx, &i) in perm.iter().enumerate() {
            let start = part_starts[i];
            let end = part_starts[i + 1];
            let len = end - start;
            // if i % 1024 == 0 {
            //     eprintln!("part {idx:>10}/{p:>10}, size {len:>10}");
            // }
            let part = &keys[start..end];

            // score function
            fn pow(pow: u32) -> impl Fn(&[usize]) -> usize {
                move |c: &[usize]| {
                    c.iter()
                        .enumerate()
                        .map(|(size, cnt)| cnt * size.pow(pow))
                        .sum::<usize>()
                }
            }
            let f = pow(7);

            // hash all keys with two seeds, and collect bucket size statistics
            let mut best = (usize::MAX, usize::MAX);
            for seed in 0..(1 << s) {
                let mut counts = vec![0; k + 1];
                for &key in part {
                    let bi = to_bin(key, seed as u64, b);
                    counts[bucket_sizes[bi] as usize] += 1;
                }
                let score = f(&counts);
                best = best.min((score, seed));
            }
            let seed = best.1;
            assert!(seed < 256);
            seeds[i] = seed as u8;

            for &key in part {
                let bi = to_bin(key, seed as u64, b);
                if bucket_sizes[bi] < k as u8 {
                    table[bi].0[bucket_sizes[bi] as usize] = key;
                    bucket_sizes[bi] += 1;
                } else {
                    collisions.push(key);
                }
            }
        }

        if !PROBE {
            assert!(collisions.is_empty());
        }

        // eprintln!("Collisions/elem:   {}", collisions.len() as f32 / n as f32);
        // Fix colliding keys.
        for key in collisions {
            let part = to_part(key, p, k);
            let seed = seeds[part];
            let mut bi = to_bin(key, seed as u64, b);
            while bucket_sizes[bi] == k as u8 {
                bi += 1;
            }
            table[bi].0[bucket_sizes[bi] as usize] = key;
            bucket_sizes[bi] += 1;
        }

        // bucket size distribution
        let mut bsizes = vec![0; k + 1];
        for bs in bucket_sizes {
            bsizes[bs as usize] += 1;
        }
        // for s in k..=k {
        //     eprintln!(
        //         "size {s:>2} => count  {:>6.3}%",
        //         bsizes[s] as f32 / b as f32 * 100.0
        //     );
        // }

        Self {
            slot_ratio,
            meta_ratio,
            k,
            s,
            n,
            b,
            m,
            p,
            table,
            seeds,
            len: 0,
            has_zero: false,
            hits: 0,
            none: 0,
            skips: 0,
            probelen: Default::default(),
            last_b: 0,
            last_j: 0,
            last_empty: 0,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len + self.has_zero as usize
    }

    // pub fn test(&mut self) {
    //     for x in &*self {
    //         if !self.contains(x) {
    //             eprintln!("Did not find {x}!");
    //             let hash64 = Hasher::default().hash_one(x);
    //             let bucket_i = (hash64 as usize).widening_mul(self.b).1;

    //             let [h1, h2]: &[S; 2] = unsafe { transmute(&self.table[bucket_i]) };
    //             let c0 = h1.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
    //             let c1 = h2.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
    //             let elems = BUCKET_SIZE - c0 - c1;
    //             eprintln!("Intended bucket {bucket_i} of size {elems}");
    //             let flat = unsafe { self.table.align_to::<T>().1 };
    //             let pos = flat.iter().position(|y| *y == x);
    //             eprintln!("Found at {pos:?}");
    //             if let Some(p) = pos {
    //                 let bucket = p / BUCKET_SIZE;
    //                 eprintln!("Actual bucket {bucket}");
    //             }

    //             panic!();
    //         }
    //     }
    // }

    pub fn stats(&self) {
        let mut counts = [0; 9];

        eprintln!("Size    : {}", self.len);
        eprintln!("hits    : {}", self.hits);
        eprintln!("None    : {}", self.none);
        eprintln!("Skips   : {}", self.skips);
        eprintln!("Skips/el: {}", self.skips as f32 / self.hits as f32);
        // return;

        let mut sum = 0;
        let mut cnt = 0;
        for bucket in &self.table {
            let [h1, h2]: &[S; 2] = unsafe { transmute(&bucket.0) };
            let c0 = h1.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
            let c1 = h2.cmp_eq(S::ZERO).move_mask().count_ones() as usize;
            let elems = BUCKET_SIZE - c0 - c1;
            counts[elems] += 1;
            cnt += 1;
            sum += elems;
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
        // self.test();
    }

    #[inline(always)]
    pub fn prefetch(&self, key: T) {
        let part = to_part(key, self.p, self.k);
        let seed = self.seeds[part];
        let bi = to_bin(key, seed as u64, self.b);
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(self.table.get_unchecked(bi) as *const Bucket as *const i8);
        }
    }

    #[inline(always)]
    pub fn contains(&mut self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let part = to_part(key, self.p, self.k);
        let seed = self.seeds[part];
        let mut bi = to_bin(key, seed as u64, self.b);

        let keys = S::splat(key as _);

        let mut i = 1;
        loop {
            use std::mem::transmute;
            // Safety: bucket_mask is correct because the number of buckets is a power of 2.
            let bucket = unsafe { self.table.get_unchecked(bi) };
            let [h1, h2]: [S; 2] = unsafe { transmute(bucket.0) };
            let mask = (h1.cmp_eq(keys) | h2.cmp_eq(keys)).move_mask() as u8;

            if !PROBE {
                return mask > 0;
            }

            if mask > 0 {
                self.hits += 1;
                return true;
            }
            let has_zero = (h1.cmp_eq(S::ZERO) | h2.cmp_eq(S::ZERO)).move_mask() as u8;
            if has_zero > 0 {
                self.none += 1;
                return false;
            }

            self.skips += 1;
            bi += 1;
            i += 1;
        }
    }

    pub fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table) + std::mem::size_of_val(&*self.seeds)
    }
}

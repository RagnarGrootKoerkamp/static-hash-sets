#![allow(unused)]
#![feature(bigint_helper_methods, impl_trait_in_assoc_type)]

mod hashset;
mod kphf;
mod static_hashset;
mod traits;

use std::{
    hash::{BuildHasher, BuildHasherDefault},
    hint::black_box,
};

use fxhash::{FxBuildHasher, FxHashSet};
use hashset::U64HashSet;
use mem_dbg::{MemSize, SizeFlags};
use rand::{
    seq::{IndexedRandom, SliceRandom},
    Rng,
};
use static_hashset::StaticHashSet;
use sux::{dict::EliasFanoBuilder, traits::IndexedDict};
use traits::HashSet;
type FxHasher = BuildHasherDefault<fxhash::FxHasher>;

type T = u32;

fn gen_keys(n: usize) -> Vec<T> {
    eprint!("Gen {n} keys..");
    let mut v = vec![0; n];
    rand::fill(&mut v[..]);
    eprintln!(" done");
    v
}

fn mul(a: usize, b: usize) -> usize {
    a.widening_mul(b).1
}

fn to_part(key: T, p: usize, k: usize) -> usize {
    let x = fxhash::hash64(&(key ^ 13245)) as usize;
    // first, replace x by 1-(1-x)^2 = 2x - x^2
    // let x = (2*x).wrapping_sub(mul(x, x));

    // quadratic: x^2
    let sq = mul(x, x);
    // qubic: (x^2 + x^3)/2
    let cube = mul(sq, x);
    let c = sq / 2 + cube / 2;
    // quartic: x^4/3 + x^3/6 + x^2/2
    let quart = mul(sq, sq);
    let q = quart / 3 + cube / 6 + sq / 2;

    // x**6
    let six = mul(quart, sq);
    let oct = mul(quart, quart);

    // c.widening_mul(p).1
    six.widening_mul(p).1
}

fn to_bin(key: T, seed: u64, b: usize) -> usize {
    (fxhash::hash64(&(key ^ seed as T)) as usize)
        .widening_mul(b)
        .1
}

// n: #keys, eg 1e9
// p: #parts, power of two, eg 1024
// b: #buckets
// k: bucket size, eg 8
// s: number of bits in each seed
fn test(n: usize, p: usize, k: usize, b: usize, s: usize) {
    let mut keys = gen_keys(n);
    // 1. sort the keys
    keys.sort();

    // 2. split into b even parts
    let mut part_sizes = vec![0; p];
    let shift = 64 - p.trailing_zeros();
    for &key in &keys {
        let p = to_part(key, p, k);
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
    let mut buckets = vec![0u8; b];
    let mut collisions = 0;
    for (idx, &i) in perm.iter().enumerate() {
        let start = part_starts[i];
        let end = part_starts[i + 1];
        let len = end - start;
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
        let f = pow(13);

        // hash all keys with two seeds, and collect bucket size statistics
        let mut best = (usize::MAX, usize::MAX);
        for seed in 0..(1 << s) {
            let mut counts = vec![0; k + 1];
            for &key in part {
                let bi = to_bin(key, seed as u64, b);
                counts[buckets[bi] as usize] += 1;
            }
            let score = f(&counts);
            best = best.min((score, seed));
        }
        let seed = best.1;

        let mut cs = 0;
        for &key in part {
            let bi = to_bin(key, seed as u64, b);
            if buckets[bi] < k as u8 {
                buckets[bi] += 1;
            } else {
                cs += 1;
            }
        }
        collisions += cs;

        // eprintln!("bucket {idx:>9} len {len:>9} collisions {cs:>9} total {collisions:>9}");
    }
    // eprintln!("Collisions: {collisions}");
    // eprintln!("Slots overhead:    {}", (b*k) as f32 / n as f32 - 1.0);
    // eprintln!("Metadata overhead: {}", (p * s) as f32 / (64 * n) as f32);
    eprintln!("Collisions/elem:   {}", collisions as f32 / n as f32);

    // bucket size distribution
    let mut bsizes = vec![0; k + 1];
    for bs in buckets {
        bsizes[bs as usize] += 1;
    }
    for s in k - 2..=k {
        eprintln!(
            "size {s:>2} => count  {:>6.3}%",
            bsizes[s] as f32 / b as f32 * 100.0
        );
    }
}

fn main() {
    // kphf::test();
    // return;

    let ns = (0..)
        .map(|i| (1_000_000. * 1.35f32.powi(i)) as usize)
        .take_while(|x| *x <= 100_000_000)
        .collect::<Vec<_>>();

    let hashers = vec![
        Box::new(FxHashSet::default()) as Box<dyn HashSet>,
        Box::new(U64HashSet::new(1.4, &[])),
        Box::new(U64HashSet::new(1.3, &[])),
        Box::new(U64HashSet::new(1.2, &[])),
        Box::new(
            fastbloom::BloomFilter::with_false_pos(0.1)
                .hasher(FxBuildHasher::default())
                .items(&[()]),
        ),
        Box::new(cuckoofilter::CuckooFilter::<fxhash::FxHasher>::with_capacity(0)),
        // Box::new(StaticHashSet::<true>::new(1.4, 0.002, &[])),
        // Box::new(StaticHashSet::<false>::new(1.4, 0.002, &[])),
        // Box::new(StaticHashSet::<true>::new(1.4, 0.001, &[])),
        Box::new(StaticHashSet::<false>::new(1.4, 0.001, &[])),
        // Box::new(StaticHashSet::<true>::new(1.4, 0.0005, &[])),
        // Box::new(StaticHashSet::<false>::new(1.4, 0.0005, &[])),
        // Box::new(StaticHashSet::<true>::new(1.3, 0.002, &[])),
        // Box::new(StaticHashSet::<false>::new(1.3, 0.002, &[])),
        // Box::new(StaticHashSet::<true>::new(1.3, 0.001, &[])),
        // Box::new(StaticHashSet::<false>::new(1.3, 0.001, &[])),
        // Box::new(StaticHashSet::<true>::new(1.3, 0.0005, &[])),
        // Box::new(StaticHashSet::<false>::new(1.3, 0.0005, &[])),
    ];
    traits::bench(&ns, &hashers);

    // let n = 118_000_000; // 2.844x overhead
    // let n = 117_000_000; // 1.434x overhead
    // for n in [100_000_000] {
    //     // absl(n);
    //     u64_hashset(n);
    //     static_hashset(n);
    //     // ef(n);
    // }

    // eprintln!("Slots overhead:    {}", (b * k) as f32 / n as f32 - 1.0);
    // eprintln!("Metadata overhead: {}", p_bits as f32 / (64 * n) as f32);
    // for s in 0..=8 {
    //     eprintln!("S = {s}");
    //     test(n, p_bits.div_ceil(s.max(1)), k, b, s);
    // }
}

fn absl(n: usize) {
    eprintln!("ABSL");
    let mut keys = gen_keys(n);
    let mut map: hashbrown::HashSet<T, FxHasher> =
        hashbrown::HashSet::from_iter(keys.iter().copied());
    eprintln!(
        "Overhead: {:.5}",
        map.allocation_size() as f32 / (n * std::mem::size_of::<T>()) as f32
    );

    let mut rng = &mut rand::rng();
    keys.shuffle(rng);

    let loops = 1;
    for pct in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0] {
        let mixed_keys = (0..n)
            .map(|_| {
                if rng.random_bool(pct) {
                    *keys.choose(rng).unwrap()
                } else {
                    rng.random()
                }
            })
            .collect::<Vec<_>>();

        let start = std::time::Instant::now();
        let mut count = 0;
        for _ in 0..loops {
            for &k in &mixed_keys {
                count += map.contains(&k) as usize;
            }
        }
        black_box(count);

        eprintln!(
            "{:3.0}% pos: {:5.2} ns/query",
            pct * 100.0,
            start.elapsed().as_secs_f32() / (loops * n) as f32 * 1e9
        );
    }

    eprintln!();
}

fn u64_hashset(n: usize) {
    // eprintln!("CUSTOM");
    let mut keys = gen_keys(n);
    let slot_ratio = 1.2;
    let mut map = hashset::U64HashSet::new(slot_ratio, &keys);
    eprintln!(
        "Overhead: {:.5}",
        map.allocation_size() as f32 / (n * std::mem::size_of::<T>()) as f32
    );

    let mut rng = &mut rand::rng();
    keys.shuffle(rng);

    let loops = 1;
    for pct in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0] {
        let mixed_keys = (0..n)
            .map(|_| {
                if rng.random_bool(pct) {
                    *keys.choose(rng).unwrap()
                } else {
                    rng.random()
                }
            })
            .collect::<Vec<_>>();

        let start = std::time::Instant::now();
        let mut count = 0;
        for _ in 0..loops {
            for &k in &mixed_keys {
                count += map.contains(k) as usize;
            }
        }
        black_box(count);

        eprint!(
            "{:3.0}% pos: {:5.2} ns/query",
            pct * 100.0,
            start.elapsed().as_secs_f32() / (loops * n) as f32 * 1e9
        );

        let lookahead = 32;
        let start = std::time::Instant::now();
        let mut count = 0;
        for _ in 0..loops {
            for i in 0..(n - lookahead) {
                map.prefetch(mixed_keys[i + lookahead]);
                count += map.contains(mixed_keys[i]) as usize;
            }
        }
        black_box(count);

        eprintln!(
            " ->  {:5.2} pf",
            start.elapsed().as_secs_f32() / (loops * n) as f32 * 1e9
        );
    }

    eprintln!();
}

fn static_hashset(n: usize) {
    // eprintln!("CUSTOM");
    let mut keys = gen_keys(n);
    let slot_ratio = 1.2;
    let meta_ratio = 0.001;
    let mut map = StaticHashSet::<true>::new(slot_ratio, meta_ratio, &mut keys.clone());
    eprintln!(
        "Overhead: {:.5}",
        map.allocation_size() as f32 / (n * std::mem::size_of::<T>()) as f32
    );

    let mut rng = &mut rand::rng();
    keys.shuffle(rng);

    let loops = 1;
    for pct in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0] {
        let mixed_keys = (0..n)
            .map(|_| {
                if rng.random_bool(pct) {
                    *keys.choose(rng).unwrap()
                } else {
                    rng.random()
                }
            })
            .collect::<Vec<_>>();

        let start = std::time::Instant::now();
        let mut count = 0;
        for _ in 0..loops {
            for &k in &mixed_keys {
                count += map.contains(k) as usize;
            }
        }
        black_box(count);

        eprint!(
            "{:3.0}% pos: {:5.2} ns/query",
            pct * 100.0,
            start.elapsed().as_secs_f32() / (loops * n) as f32 * 1e9
        );

        let lookahead = 32;
        let start = std::time::Instant::now();
        let mut count = 0;
        for _ in 0..loops {
            for i in 0..(n - lookahead) {
                map.prefetch(mixed_keys[i + lookahead]);
                count += map.contains(mixed_keys[i]) as usize;
            }
        }
        black_box(count);

        eprintln!(
            " ->  {:5.2} pf",
            start.elapsed().as_secs_f32() / (loops * n) as f32 * 1e9
        );
    }

    eprintln!();
}

fn ef(n: usize) {
    eprintln!("EF");
    let mut keys = gen_keys(n);
    keys.sort_unstable();

    let mut efb = EliasFanoBuilder::new(keys.len(), T::MAX as usize);
    for key in &keys {
        efb.push(*key as usize);
    }
    let ef = efb.build_with_seq_and_dict();

    eprintln!(
        "Overhead: {:.5}",
        ef.mem_size(SizeFlags::default()) as f32 / (n * std::mem::size_of::<T>()) as f32
    );

    let loops = 1;
    for pct in [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0] {
        let mut rng = &mut rand::rng();
        let mixed_keys = (0..n)
            .map(|_| {
                if rng.random_bool(pct) {
                    *keys.choose(rng).unwrap()
                } else {
                    rng.random()
                }
            })
            .collect::<Vec<_>>();

        let start = std::time::Instant::now();
        let mut count = 0;
        for _ in 0..loops {
            for &k in &mixed_keys {
                count += ef.contains(k as usize) as usize;
            }
        }
        black_box(count);

        eprintln!(
            "{:3.0}% pos: {:5.2} ns/query",
            pct * 100.0,
            start.elapsed().as_secs_f32() / (loops * n) as f32 * 1e9
        );
    }

    eprintln!();
}

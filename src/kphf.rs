use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
use std::hash::{BuildHasher, BuildHasherDefault};
use std::hint::select_unpredictable;
use std::mem::transmute;
type S = wide::i32x8;

use rustc_hash::FxHashMap;
use wide::CmpEq;

use super::{to_bin, to_part, T};

pub struct Kphf {
    pub slot_ratio: f32,
    pub meta_ratio: f32,
    k: usize,
    s: usize,
    n: usize,
    b: usize,
    m: usize,
    p: usize,
    seeds: Vec<u8>,
}

const BIN_SIZE: usize = 16;
const PADDING: usize = 100;

impl Kphf {
    pub fn new(slot_ratio: f32, meta_ratio: f32, keys: &[T], sort: bool, consensus: bool) -> Self {
        // eprintln!("building..");
        let k = BIN_SIZE;
        let s = 8;
        let n = keys.len();
        // bins
        let b = ((n as f32 * slot_ratio) as usize).div_ceil(k);
        // total metadata bits
        let m = ((n * 64) as f32 * meta_ratio) as usize;
        // bits per part
        let p = m.div_ceil(s.max(1));

        // 1. sort the keys
        let mut keys = keys.to_vec();
        keys.sort_unstable();
        keys.dedup();

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

        if consensus {
            assert!(!sort);
        }

        if sort {
            perm.sort_by_key(|&i| std::cmp::Reverse(part_sizes[i]));
        }

        // 4. init bin sizes
        let mut bin_sizes = vec![0u8; b + PADDING];
        let mut seeds = vec![0u8; p + 7];
        let mut tries = vec![0u8; p];

        let mut collisions = vec![];
        let mut backtracks = 0;
        // eprintln!("find seeds..");
        let mut elems_done = 0;
        let mut i = 0;
        while i < p {
            let idx = perm[i];
            let start = part_starts[idx];
            let end = part_starts[idx + 1];
            let len = end - start;

            assert_eq!(
                elems_done,
                bin_sizes.iter().map(|&x| x as usize).sum::<usize>() + collisions.len()
            );
            let num_full = bin_sizes.iter().filter(|&&x| x == k as u8).count();

            // if i % 1024 == 0 {
            //     eprintln!(
            //         "part {idx:>10}/{p:>10}, size {len:>10}, done {elems_done:>10} ({:>7.4})",
            //         elems_done as f32 / n as f32 * 100.0
            //     );
            // }
            elems_done += len;
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

            // hash all keys with two seeds, and collect bin size statistics
            let mut vals = vec![(usize::MAX, usize::MAX, usize::MAX)];

            let seed_offset = if consensus {
                u64::from_be_bytes(seeds[i..i + 8].try_into().unwrap())
            } else {
                0
            };
            eprintln!(
                "Part {i:>9} len {len:>7} tries {:>3} elems_done {:>7.4}%  full_bins {:>7.4}%",
                tries[i],
                elems_done as f32 / n as f32 * 100.0,
                num_full as f32 / b as f32 * 100.0
            );
            // eprintln!("read {i}..={} => {seed_offset:>0x}", i + 7);
            assert!(
                seed_offset % 256 == 0,
                "{seed_offset:>0x} {:?}",
                seed_offset.to_be_bytes()
            );

            for seed in 0..(1 << s) {
                let mut counts = vec![0; k + 1];
                for &key in part {
                    let bi = to_bin(key, seed_offset + seed as u64, b);
                    counts[(bin_sizes[bi] as usize).min(k)] += 1;
                    // update the bin_size to handle self-collisions
                    bin_sizes[bi] += 1;
                }
                let score = f(&counts);
                vals.push((counts[k], score, seed));
                for &key in part {
                    let bi = to_bin(key, seed_offset + seed as u64, b);
                    bin_sizes[bi] -= 1;
                }
            }
            vals.sort();
            let best = vals[tries[i] as usize];
            if consensus && best.0 > 0 {
                elems_done -= len;
                backtracks += 1;
                eprintln!(
                    "BACKTRACK part {i} len {len} trie {} collisions {} for {:?} while best is {:?}",
                    tries[i], best.0, best, vals[0]
                );
                // Backtrack 1 step.
                tries[i] = 0;
                if i > 0 {
                    assert!(tries[i - 1] < 255);
                    tries[i - 1] += 1;
                    // Reduce the bucket size of previous seed of parent.

                    let start = part_starts[idx - 1];
                    let end = part_starts[idx];
                    let len = end - start;
                    elems_done -= len;
                    let part = &keys[start..end];
                    eprintln!("Unset part {} len {len}", idx - 1);

                    // eprintln!("read {}..={} to empty parent bucket", i - 1, i + 6);
                    let seed = u64::from_be_bytes(seeds[i - 1..i + 7].try_into().unwrap());

                    for &key in part {
                        let bi = to_bin(key, seed, b);
                        assert!(bin_sizes[bi] > 0);
                        bin_sizes[bi] -= 1;
                    }

                    // eprintln!("Set {} to 0", i + 6);
                    seeds[i + 6] = 0;
                    i -= 1;
                }
                if i == 0 {
                    seeds[6] += 1;
                }
                continue;
            }
            let seed = best.2;
            assert!(seed < 256);
            // eprintln!("Set {} to {seed}; best: {best:?}", idx + 7);
            seeds[idx + 7] = seed as u8;

            for &key in part {
                let bi = to_bin(key, seed_offset + seed as u64, b);
                if bin_sizes[bi] < k as u8 {
                    bin_sizes[bi] += 1;
                } else {
                    collisions.push(key);
                    if consensus {
                        eprintln!(
                            "collision at {i} size {len} seed {seed} offset {seed_offset:>0x} best {best:?} bin id {bi} bin size {}", bin_sizes[bi]
                        );
                        panic!();
                    }
                }
            }

            i += 1;
        }
        let num_collisions = collisions.len();
        // Fix colliding keys.
        for key in collisions {
            let part = to_part(key, p, k);
            let seed = u64::from_be_bytes(seeds[part..part + 8].try_into().unwrap());
            let mut bi = to_bin(key, seed as u64, b);
            while bin_sizes[bi] == k as u8 {
                bi += 1;
            }
            bin_sizes[bi] += 1;
        }

        // bin size distribution
        let mut bsizes = vec![0; k + 1];
        for bs in bin_sizes {
            bsizes[bs as usize] += 1;
        }
        // for s in k..=k {
        //     eprintln!(
        //         "size {s:>2} => count  {:>6.3}%",
        //         bsizes[s] as f32 / b as f32 * 100.0
        //     );
        // }

        eprintln!(
            "slot_ratio: {slot_ratio:>4.2}, meta_ratio: {meta_ratio:<6}, sort: {sort:>5}, consensus: {consensus:>5}, bits/key: {:.4} Collisions: {:>7} BTs {backtracks:>7}",
            (m as f32) / (n as f32),
            num_collisions
        );
        Self {
            slot_ratio,
            meta_ratio,
            k,
            s,
            n,
            b,
            m,
            p,
            seeds,
        }
    }
}

pub fn test() {
    let n = 3_000_000;
    let mut keys = vec![0u32; n];
    rand::fill(&mut keys[..]);

    for slot_ratio in [1.25] {
        for (sort, cons) in [(true, false), (false, false), (false, true)] {
            for meta_ratio in [0.0005] {
                // for slot_ratio in [1.3, 1.25, 1.2] {
                //     for meta_ratio in [0.002, 0.001, 0.0005] {
                //         for (sort, cons) in [(true, false), (false, false), (false, true)] {
                let kphf = Kphf::new(slot_ratio, meta_ratio, &keys, sort, cons);
            }
        }
    }
}

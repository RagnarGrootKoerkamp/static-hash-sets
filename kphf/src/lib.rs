#![feature(impl_trait_in_assoc_type, widening_mul)]

pub trait KphfT {
    fn name(&self) -> &'static str;
    fn new(&self, keys: &[u32]) -> Self;
    fn has_prefetch(&self) -> bool {
        false
    }
    fn prefetch(&mut self, _key: u32) {}
    fn get(&mut self, key: u32) -> usize;
}

use std::array::from_fn;

fn mul(a: usize, b: usize) -> usize {
    a.widening_mul(b).1
}
type T = u32;

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

pub struct KptrHash {
    /// Fill ratio
    pub alpha: f32,
    /// Bits per key
    pub bits_per_key: f32,
    /// The construction mode.
    mode: Mode,
    /// Number of keys per bin.
    k: usize,
    s: usize,
    /// Number of keys.
    n: usize,
    /// Number of bins.
    b: usize,
    /// Metadata bits.
    m: usize,
    /// Number of buckets.
    p: usize,
    /// Actual seeds data.
    seeds: Vec<u8>,
    /// Bump structure
    bumped: Option<Box<Self>>,
}

const BIN_SIZE: usize = 8;
const PADDING: usize = 100;

#[derive(Debug, Copy, Clone)]
pub enum Mode {
    /// Process buckets left to right
    Linear,
    /// Process buckets left to right, and allow bumping
    LinearBump,
    /// Process buckets left to right, and allow smooth bumping.
    /// Picks the first key that does not overflow buckets.
    LinearSmoothBumpGreedy(usize),
    /// Process buckets left to right, and allow smooth bumping.
    /// Tries all keys up to threshold, and only then picks the first working key.
    LinearSmoothBump(usize),
    /// Process buckets large to small
    Sort,
    /// Process buckets large to small, and allow bumping
    SortBump,
    /// Process buckets large to small, and allow smooth bumping.
    /// Picks the first key that does not overflow buckets.
    SortSmoothBumpGreedy(usize),
    /// Process buckets large to small, and allow smooth bumping.
    /// Tries all keys up to threshold, and only then picks the first working key.
    SortSmoothBump(usize),
    /// Process buckets large to small, and allow smooth bumping.
    /// Tries all keys, and picks the best one with minimal bumping.
    SortSmoothBumpLazy(usize),
    /// Process buckets left to right, and backtrack
    Consensus,
}

impl KptrHash {
    pub fn new(alpha: f32, bits_per_key: f32, keys: &[T], mode: Mode) -> Option<Self> {
        let start = std::time::Instant::now();
        let sort = matches!(
            mode,
            Mode::Sort | Mode::SortBump | Mode::SortSmoothBumpGreedy(_) | Mode::SortSmoothBump(_)
        );
        let consensus = matches!(mode, Mode::Consensus);
        let bump = matches!(
            mode,
            Mode::LinearBump
                | Mode::LinearSmoothBumpGreedy(_)
                | Mode::LinearSmoothBump(_)
                | Mode::SortBump
                | Mode::SortSmoothBumpGreedy(_)
                | Mode::SortSmoothBump(_)
                | Mode::SortSmoothBumpLazy(_)
        );
        let smoothbump = matches!(
            mode,
            Mode::LinearSmoothBump(_)
                | Mode::LinearSmoothBumpGreedy(_)
                | Mode::SortSmoothBump(_)
                | Mode::SortSmoothBumpGreedy(_)
                | Mode::SortSmoothBumpLazy(_)
        );
        let threshold = if smoothbump {
            match mode {
                Mode::LinearSmoothBumpGreedy(t)
                | Mode::LinearSmoothBump(t)
                | Mode::SortSmoothBumpGreedy(t)
                | Mode::SortSmoothBump(t)
                | Mode::SortSmoothBumpLazy(t) => t,
                _ => unreachable!(),
            }
        } else {
            0
        };
        let smoothbumpgreedy = matches!(
            mode,
            Mode::LinearSmoothBumpGreedy(_) | Mode::SortSmoothBumpGreedy(_)
        );
        let smoothbumplazy = matches!(mode, Mode::SortSmoothBumpLazy(_));

        // eprintln!("building..");
        let k = BIN_SIZE;
        // bits per bucket
        let s = 8;
        let n = keys.len();
        // bins
        let b = ((n as f32 / alpha) as usize).div_ceil(k);
        // target metadata bits
        let m = (n as f32 * bits_per_key) as usize;
        // #buckets
        let p = m.div_ceil(s.max(1));
        // total bits
        let bits = s * p;

        // 1. sort the keys by part
        let mut keys = keys.to_vec();
        // keys.sort_unstable();
        keys.sort_unstable_by_key(|&key| (to_part(key, p, k), key));
        keys.dedup();

        // 2. split into p parts
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

        if sort {
            perm.sort_by_key(|&i| std::cmp::Reverse(part_sizes[i]));
        }

        // 4. init bin sizes
        let mut bin_sizes = vec![0u8; b + PADDING];
        let mut seeds = vec![0u8; p + 7];
        let mut tries = vec![0u8; p];

        let mut bumped = 0;
        let mut collisions = vec![];
        let mut backtracks = 0;
        // eprintln!("find seeds..");
        let mut elems_done = 0;
        let mut i = 0;

        let mut lg: [f64; BIN_SIZE + 1] = from_fn(|i| (i as f64).log2());
        lg[0] = f64::MIN;

        let mut bumped_keys = vec![];

        while i < p {
            let idx = perm[i];
            let start = part_starts[idx];
            let end = part_starts[idx + 1];
            let len = end - start;
            // if i % 1024 == 0 {
            //     eprintln!("bucket {i} of {p} len {len}");
            // }

            // assert_eq!(
            //     elems_done,
            //     bin_sizes.iter().map(|&x| x as usize).sum::<usize>() + collisions.len()
            // );
            let num_full = bin_sizes.iter().filter(|&&x| x == k as u8).count();

            // if i % 1024 == 0 {
            //     eprintln!(
            //         "part {idx:>10}/{p:>10}, size {len:>10}, done {elems_done:>10} ({:>7.4})",
            //         elems_done as f32 / n as f32 * 100.0
            //     );
            // }
            elems_done += len;
            let part = &keys[start..end];
            assert!(part.is_sorted());

            // score function
            fn pow(pow: u32) -> impl Fn(&[usize]) -> i64 {
                move |c: &[usize]| {
                    c.iter()
                        .enumerate()
                        .map(|(size, cnt)| cnt * size.pow(pow))
                        .sum::<usize>() as i64
                }
            }
            // New candidate score function:
            // Maximize the product of the number of empty slots in each bin.
            // eprintln!("lg: {:?}", &lg);
            // eprintln!("lg: {}", lg[0] * 2. + 1.0);
            let maybe_optimal = |counts: &[usize]| -> i64 {
                if counts[BIN_SIZE] > 0 {
                    return i64::MAX;
                }
                let q = counts
                    .iter()
                    .enumerate()
                    .map(|(size, &cnt)| cnt as f64 * lg[BIN_SIZE - size])
                    .sum::<f64>();
                // eprintln!("counts {counts:?} score: {q}");
                (-q) as i64
            };
            let f = pow(7);
            // let f = maybe_optimal;

            // hash all keys with two seeds, and collect bin size statistics
            let mut vals = vec![(usize::MAX, usize::MAX, i64::MAX, usize::MAX)];

            let seed_offset = if consensus {
                u64::from_be_bytes(seeds[i..i + 8].try_into().unwrap())
            } else {
                0
            };
            // eprintln!(
            //     "Part {i:>9} len {len:>7} tries {:>3} elems_done {:>7.4}%  full_bins {:>7.4}%",
            //     tries[i],
            //     elems_done as f32 / n as f32 * 100.0,
            //     num_full as f32 / b as f32 * 100.0
            // );
            // eprintln!("read {i}..={} => {seed_offset:>0x}", i + 7);
            assert!(
                seed_offset % 256 == 0,
                "{seed_offset:>0x} {:?}",
                seed_offset.to_be_bytes()
            );

            let max_key_for_seed = |seed: usize| {
                if seed < threshold {
                    u32::MAX
                } else {
                    u32::MAX / (256 - threshold as u32) * (255 - seed) as u32
                }
            };
            // let max_key_for_seed = |seed: usize| {
            //     if seed == 255 {
            //         return 0;
            //     }
            //     if seed < threshold {
            //         u32::MAX
            //     } else {
            //         u32::MAX / 100 * 80
            //     }
            // };

            let mut ok = false;

            for seed in 0..(1 << s) - if bump { 1 } else { 0 } {
                if smoothbump && !smoothbumplazy && seed >= threshold && ok {
                    break;
                }
                let mut counts = vec![0; k + 1];
                let max_key = max_key_for_seed(seed);
                let mut bumped = 0;
                for key in part {
                    if smoothbump && *key > max_key {
                        // eprintln!(
                        //     "smoothbump: seed {seed} max_key {max_key:>0x} => key {key:>0x} bumped idx {} of {}",
                        //     part.element_offset(key).unwrap(),
                        //     part.len(),

                        // );
                        bumped = part.len() - part.element_offset(key).unwrap();
                        break;
                    }
                    let bi = to_bin(*key, seed_offset + seed as u64, b);
                    counts[(bin_sizes[bi] as usize).min(k)] += 1;
                    // update the bin_size to handle self-collisions
                    bin_sizes[bi] += 1;
                }
                // eprintln!("Counts for seed {seed}: {counts:?}");
                let score = f(&counts);
                vals.push((counts[BIN_SIZE], bumped, score, seed));
                if bumped == 0 && counts[BIN_SIZE] == 0 {
                    ok = true;
                }
                for &key in part {
                    if smoothbump && key > max_key {
                        continue;
                    }
                    let bi = to_bin(key, seed_offset + seed as u64, b);
                    bin_sizes[bi] -= 1;
                }
                if smoothbump && !smoothbumplazy {
                    if smoothbumpgreedy {
                        if counts[BIN_SIZE] == 0 {
                            break;
                        }
                    } else {
                        if seed >= threshold && counts[BIN_SIZE] == 0 {
                            break;
                        }
                    }
                }
            }
            vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let best = vals[tries[i] as usize];
            if smoothbumplazy {
                eprintln!("best {best:?}");
            }
            let num_bumped_for_seed = best.1;
            // fix for added best.1 = #bumped count.
            let best = (best.0, best.2, best.3);
            if consensus && best.0 > 0 {
                elems_done -= len;
                backtracks += 1;
                // eprintln!(
                //     "BACKTRACK part {i} len {len} trie {} collisions {} for {:?} while best is {:?}",
                //     tries[i], best.0, best, vals[0]
                // );
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
                    // eprintln!("Unset part {} len {len}", idx - 1);

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

            if bump && best.0 > 0 {
                elems_done -= part.len();
                bumped += part.len();
                i += 1;
                bumped_keys.extend_from_slice(part);
                seeds[idx + 7] = 255;

                // eprintln!(
                //     "{i}: Bump bucket of size {}; total bumped {bumped}",
                //     part.len()
                // );
                continue;
            }

            let seed = best.2;
            assert!(seed < 256);
            // eprintln!("Set {} to {seed}; best: {best:?}", idx + 7);
            seeds[idx + 7] = seed as u8;
            let max_key = max_key_for_seed(seed);

            let mut smoothbumped = 0;
            for &key in part {
                if smoothbump && key > max_key {
                    smoothbumped += 1;
                    bumped_keys.push(key);
                    continue;
                }
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
            if smoothbumplazy {
                assert_eq!(num_bumped_for_seed, smoothbumped);
            }
            if smoothbump && smoothbumped > 0 {
                bumped += smoothbumped;
                // eprintln!(
                //     "smoothbump: fixing seed {seed:>4} for bucket {i:>6} of {p:>6} of size {:>3}. Bumped {smoothbumped:>3} [{bumped:>5} total]",
                //     part.len()
                // );
            }

            i += 1;
        }
        let num_collisions = collisions.len();
        // Fix colliding keys.
        // for key in collisions {
        //     let part = to_part(key, p, k);
        //     let seed = u64::from_be_bytes(seeds[part..part + 8].try_into().unwrap());
        //     let mut bi = to_bin(key, seed as u64, b);
        //     while bin_sizes[bi] == k as u8 {
        //         bi += 1;
        //     }
        //     bin_sizes[bi] += 1;
        // }

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

        let duration = start.elapsed();
        eprintln!(
            "alpha: {alpha:>4.2}, bits/key: {bits_per_key:<6}, mode: {:>27}, bits/key: {:.4} Collisions: {:>7} BTs {backtracks:>7} Bumped {bumped:>7} ({:.4}%) {:>6?}ms",
            format!("{mode:?}"),
            (m as f32) / (n as f32),
            num_collisions, bumped as f32 / n as f32 * 100.0,
            duration.as_millis()
        );

        if !bump && num_collisions > 0 {
            return None;
        }

        Some(Self {
            alpha,
            bits_per_key,
            mode,
            k,
            s,
            n,
            b,
            m,
            p,
            seeds,
            bumped: if bumped > 0 {
                Some(Box::new(Self::new(
                    // use a lazy load factor for fallback
                    0.5,
                    // double the bits/key for bumped keys
                    2.0 * bits_per_key,
                    &bumped_keys,
                    mode,
                )?))
            } else {
                None
            },
        })
    }

    fn bins(&self) -> usize {
        self.b + self.bumped.as_ref().map_or(0, |b| b.bins())
    }

    fn get(&self, key: T) -> usize {
        let part = to_part(key, self.p, self.k);
        let seed = match self.mode {
            Mode::Consensus => u64::from_be_bytes(self.seeds[part..part + 8].try_into().unwrap()),
            _ => self.seeds[part + 7] as u64,
        };
        if matches!(self.mode, Mode::LinearBump | Mode::SortBump) && seed == 255 {
            self.b + self.bumped.as_ref().unwrap().get(key)
        } else {
            to_bin(key, seed, self.b)
        }
    }
}

pub fn test() {
    let n = 1_000_000;
    let mut keys = vec![0u32; n];
    rand::fill(&mut keys[..]);

    for alpha in [0.90] {
        for bits_per_key in [0.60, 0.50, 0.45, 0.40, 0.35, 0.3, 0.275, 0.25, 0.225] {
            for mode in [
                // Mode::Linear,
                // Mode::LinearBump,
                // // Mode::LinearSmoothBumpGreedy(128),
                // // Mode::LinearSmoothBumpGreedy(192),
                // // Mode::LinearSmoothBumpGreedy(224),
                // // Mode::LinearSmoothBumpGreedy(240),
                // // Mode::LinearSmoothBumpGreedy(248),
                // // Mode::LinearSmoothBump(128),
                // // Mode::LinearSmoothBump(192),
                // // Mode::LinearSmoothBump(224),
                // // Mode::LinearSmoothBump(240),
                // // Mode::LinearSmoothBump(248),
                // Mode::Sort,
                Mode::SortBump,
                // Mode::SortSmoothBumpGreedy(128),
                // Mode::SortSmoothBumpGreedy(192),
                // Mode::SortSmoothBumpGreedy(224),
                // Mode::SortSmoothBumpGreedy(240),
                Mode::SortSmoothBumpGreedy(254),
                // Mode::SortSmoothBump(128),
                // Mode::SortSmoothBump(192),
                // Mode::SortSmoothBump(224),
                // Mode::SortSmoothBump(240),
                // Mode::SortSmoothBump(248),
                // Mode::SortSmoothBumpLazy(128),
                // Mode::SortSmoothBumpLazy(192),
                // Mode::SortSmoothBumpLazy(224),
                // Mode::SortSmoothBumpLazy(240),
                // Mode::SortSmoothBumpLazy(248),
                Mode::Consensus,
            ] {
                let kphf = KptrHash::new(alpha, bits_per_key, &keys, mode);
            }
            eprintln!();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    fn test_config(keys: &[u32], alpha: f32, bits_per_key: f32, mode: Mode) {
        let Some(kphf) = KptrHash::new(alpha, bits_per_key, &keys, mode) else {
            return;
        };
        let mut cnt = vec![0; kphf.bins()];
        for key in keys {
            let bi = kphf.get(*key);
            cnt[bi] += 1;
            assert!(cnt[bi] <= kphf.k, "bin {bi} has {} keys", cnt[bi]);
        }
    }

    #[test]
    fn correctness() {
        let n = 1_000_000;
        let keys = gen_keys(n);

        for alpha in [0.90] {
            for bits_per_key in [0.40, 0.35, 0.3, 0.25] {
                for mode in [
                    Mode::Linear,
                    Mode::LinearBump,
                    Mode::Sort,
                    Mode::SortBump,
                    Mode::Consensus,
                ] {
                    test_config(&keys, alpha, bits_per_key, mode);
                }
            }
        }
    }

    fn gen_keys(n: usize) -> Vec<u32> {
        let mut keys = std::collections::HashSet::with_capacity(n);
        let mut buf = vec![0u32; 1024];
        while keys.len() < n {
            rand::fill(&mut buf[..]);
            keys.extend(buf.iter().copied());
        }
        let keys: Vec<u32> = keys.into_iter().take(n).collect();
        keys
    }
}

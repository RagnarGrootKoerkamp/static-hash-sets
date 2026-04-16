#![allow(incomplete_features)]
#![feature(widening_mul, adt_const_params, generic_const_exprs)]

use std::{fmt::Debug, hint::cold_path};
use sux::{traits::BitVecOpsMut, utils::prefetch_index};

pub trait KphfT {
    fn name(&self) -> &'static str;
    fn new(&self, keys: &[u32]) -> Self;
    fn has_prefetch(&self) -> bool {
        false
    }
    fn prefetch(&mut self, _key: u32) {}
    fn get(&mut self, key: u32) -> usize;
}

fn mul(a: usize, b: usize) -> usize {
    a.widening_mul(b).1
}

pub trait Key:
    Copy + Ord + std::hash::Hash + std::ops::BitXor<Output = Self> + Debug + Default
{
    const SALT: Self;
    fn from_seed(seed: u64) -> Self;
}

impl Key for u32 {
    const SALT: Self = 13245;
    fn from_seed(seed: u64) -> Self {
        seed as u32
    }
}

impl Key for u64 {
    const SALT: Self = 13245;
    fn from_seed(seed: u64) -> Self {
        seed
    }
}

pub struct KptrHash<const MODE: Mode, const K: usize> {
    /// Fill ratio
    pub alpha: f32,
    /// Bits per key
    pub bits_per_key: f32,
    /// Number of keys.
    n: usize,
    /// Number of bins.
    num_bins: usize,
    /// Number of buckets.
    num_buckets: usize,
    /// Actual seeds data.
    seeds: Vec<u8>,
    /// Bump structure
    bumped: Option<Box<Self>>,
    /// Salt for hashing, against bad inputs, and so that bumped keys map differently.
    salt: u64,
}

const PADDING: usize = 1 << 6;

#[derive(Debug, Copy, Clone, PartialEq, Eq, std::marker::ConstParamTy)]
pub enum Mode {
    /// Process buckets left to right
    Linear,
    /// Process buckets left to right, and allow bumping
    LinearBump,
    /// Process buckets left to right, and allow bumping. Always take first working seed.
    LinearBumpGreedy,
    /// Process buckets large to small
    Sort,
    /// Process buckets large to small, and allow bumping
    SortBump,
    /// Process buckets large to small, and allow bumping. Always take first working seed.
    SortBumpGreedy,
    /// Process buckets left to right, and backtrack
    Consensus,
    /// Process buckets left to right, and backtrack. Always take first working seed.
    ConsensusGreedy,
}

/// Information-theoretic lower bound on bits/key for a static hash function
/// with bin size k and load factor alpha.
pub fn space_lower_bound(k: usize, alpha: f32) -> f32 {
    // Rows: alpha, columns: k
    const ALPHAS: [f32; 14] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0,
    ];
    const KS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    #[rustfmt::skip]
    const TABLE: [[f32; 8]; 14] = [
        [0.07466, 0.00849, 0.00022, 3e-7,     1e-12,    0.0,      0.0,      0.0     ],
        [0.15498, 0.03112, 0.00259, 4e-5,     2e-8,     2e-14,    0.0,      0.0     ],
        [0.24202, 0.06581, 0.00989, 5.2e-4,   3e-6,     4e-10,    0.0,      0.0     ],
        [0.33724, 0.11221, 0.02444, 2.67e-3,  8e-5,     2e-7,     3e-12,    0.0     ],
        [0.44269, 0.17114, 0.04833, 8.57e-3,  6.8e-4,   1.2e-5,   9e-9,     1e-14   ],
        [0.56140, 0.24472, 0.08410, 2.095e-2, 3.18e-3,  2.0e-4,   2e-6,     6e-10   ],
        [0.69828, 0.33704, 0.13568, 4.360e-2, 1.028e-2, 1.49e-3,  9e-5,     9e-7    ],
        [0.86221, 0.45624, 0.21044, 8.294e-2, 2.699e-2, 6.77e-3,  1.13e-3,  9e-5    ],
        [1.07359, 0.62179, 0.32631, 0.15418,  6.498e-2, 2.404e-2, 7.55e-3,  1.88e-3 ],
        [1.21522, 0.73972, 0.41651, 0.21663,  0.10374,  4.552e-2, 1.815e-2, 6.47e-3 ],
        [1.32751, 0.83742, 0.49629, 0.27699,  0.14560,  7.194e-2, 3.332e-2, 1.444e-2],
        [1.37558, 0.88056, 0.53318, 0.30680,  0.16811,  8.765e-2, 4.339e-2, 2.036e-2],
        [1.43271, 0.93321, 0.58010, 0.34702,  0.20111,  0.11335,  6.222e-2, 2.895e-2],
        [1.44269, 0.94265, 0.58893, 0.35509,  0.20832,  0.11967,  6.761e-2, 3.770e-2],
    ];

    let row = ALPHAS.iter().position(|&x| x == alpha).unwrap();
    let col = KS.iter().position(|&x| x == k).unwrap();
    TABLE[row][col]
}

const SEED_MASK: u64 = 0b0011_1111;

impl<const MODE: Mode, const K: usize> KptrHash<MODE, K> {
    #[inline(always)]
    fn to_bucket<T: Key>(&self, key: T) -> usize {
        let x = fxhash::hash64(&(key ^ T::from_seed(self.salt))) as usize;
        // quadratic: x^2
        let sq = mul(x, x);
        // x**6
        let six = mul(mul(sq, sq), sq);
        six.widening_mul(self.num_buckets).1
    }

    #[inline(always)]
    fn to_bin<T: Key>(&self, key: T, seed: u64) -> usize {
        // The low 6 bits indicate a shift.
        (fxhash::hash64(&(key ^ T::from_seed(seed & !SEED_MASK))) as usize)
            .widening_mul(self.num_bins - PADDING)
            .1
            + (seed & SEED_MASK) as usize
    }

    pub fn new<T: Key>(alpha: f32, bits_per_key: f32, keys: &[T]) -> Option<Self> {
        let n = keys.len();
        // bins
        let num_bins = ((n as f32 / alpha) as usize).div_ceil(K) + PADDING;
        // target metadata bits
        let m = (n as f32 * bits_per_key) as usize;
        // #buckets
        let num_buckets = m.div_ceil(8);

        // eprintln!("n: {n}, alpha: {alpha}, bits/key: {bits_per_key}, bins: {num_bins}, buckets: {num_buckets}");

        let mut kphf = Self {
            alpha,
            bits_per_key,
            n,
            num_bins,
            num_buckets,
            seeds: vec![],
            bumped: None,
            salt: rand::random(),
        };
        kphf.build(keys).map(|_| kphf)
    }

    fn build<T: Key>(&mut self, keys: &[T]) -> Option<()> {
        let start = std::time::Instant::now();
        let sort = matches!(MODE, Mode::Sort | Mode::SortBump | Mode::SortBumpGreedy);
        let consensus = matches!(MODE, Mode::Consensus | Mode::ConsensusGreedy);
        let bump = matches!(
            MODE,
            Mode::LinearBump | Mode::SortBump | Mode::LinearBumpGreedy | Mode::SortBumpGreedy
        );
        let greedy = matches!(
            MODE,
            Mode::LinearBumpGreedy | Mode::SortBumpGreedy | Mode::ConsensusGreedy
        );

        let n = self.n;

        // 1. count keys per bucket
        let mut bucket_sizes = vec![0; self.num_buckets + 1];
        for key in keys {
            let bi = self.to_bucket(*key);
            bucket_sizes[bi] += 1;
        }
        // eprintln!("KEYS: {keys:?}");
        // 2. get start positions
        let mut bucket_starts = bucket_sizes;
        let mut sum = 0;
        for i in 0..=self.num_buckets {
            let x = sum;
            sum += bucket_starts[i];
            bucket_starts[i] = x;
        }
        // 3. write keys into buckets
        let mut bucketed_keys = vec![T::default(); n];
        for key in keys {
            let bi = self.to_bucket(*key);
            bucketed_keys[bucket_starts[bi]] = *key;
            bucket_starts[bi] += 1;
        }
        // 4. restore bucket_starts
        for i in (1..=self.num_buckets).rev() {
            bucket_starts[i] = bucket_starts[i - 1];
        }
        bucket_starts[0] = 0;
        let keys = bucketed_keys;

        // keys.sort_by_cached_key(|&key| (self.to_bucket(key), key));
        // keys.dedup();

        // 3. Sort buckets by decreasing size
        let mut perm = (0..self.num_buckets).collect::<Vec<_>>();

        if sort {
            perm.sort_by_key(|&i| std::cmp::Reverse(bucket_starts[i + 1] - bucket_starts[i]));
        }

        // 4. init bin sizes
        let mut bin_sizes = vec![0u8; self.num_bins];
        let mut non_full_bins = sux::bit_vec![true; self.num_bins];

        let mut seeds = vec![0u8; self.num_buckets + 7];
        let mut tries = vec![0u8; self.num_buckets];

        let mut bumped = 0;
        let mut backtracks = 0;
        let mut i = 0;

        let mut lg: [f64; K] = std::array::from_fn(|i| (i as f64).log2());
        lg[0] = f64::MIN;

        let mut bumped_keys: Vec<T> = vec![];

        let mut bins = vec![];
        let mut bin_counts = vec![];

        // eprintln!("Start construction");

        // score function
        fn pow<const K: usize>(pow: u32) -> impl Fn(&[isize]) -> i64 {
            let pows: [isize; K] = std::array::from_fn(|i| (i as isize + 1).pow(pow));
            move |c: &[isize]| {
                c.iter()
                    .enumerate()
                    .map(|(size, cnt)| *cnt * pows[size])
                    .sum::<isize>() as i64
            }
        }
        // New candidate score function:
        // Maximize the product of the number of empty slots in each bin.
        #[allow(unused)]
        let maybe_optimal = |counts: &[isize]| -> i64 {
            if counts[K] > 0 {
                return i64::MAX;
            }
            let q = counts
                .iter()
                .enumerate()
                .map(|(size, &cnt)| cnt as f64 * lg[K - 1 - size])
                .sum::<f64>();
            (-q) as i64
        };
        let f = pow::<K>(7);
        // let f = maybe_optimal;

        'bucket: while i < self.num_buckets {
            let idx = perm[i];
            let start = bucket_starts[idx];
            let end = bucket_starts[idx + 1];
            let len = end - start;

            if len == 0 {
                i += 1;
                continue 'bucket;
            }

            let bucket = &keys[start..end];

            // hash all keys with all seeds, and collect bin size statistics
            let mut vals = vec![(usize::MAX, i64::MAX, usize::MAX)];

            let seed_offset = if consensus {
                u64::from_be_bytes(seeds[i..i + 8].try_into().unwrap())
            } else {
                0
            };
            assert!(
                seed_offset % 256 == 0,
                "{seed_offset:>0x} {:?}",
                seed_offset.to_be_bytes()
            );

            let mut mask = u64::MAX;
            let mut max_count: usize;
            let mut counts = vec![0isize; K + 1];

            'seed: for seed in 0..256_usize - if bump { 1 } else { 0 } {
                if seed % 64 == 0 {
                    mask = u64::MAX;
                    bins.clear();
                    bin_counts.clear();

                    // Get the bins this set of keys maps to.
                    for key in bucket {
                        let bi = self.to_bin(*key, seed_offset + seed as u64);
                        bins.push(bi);
                    }

                    // Group the bins into (bin, count) pairs.
                    bins.sort_unstable();
                    let mut last_bi = bins[0];
                    let mut count = 1;
                    max_count = 0;
                    for &bi in &bins[1..] {
                        if bi != last_bi {
                            bin_counts.push((last_bi, count));
                            max_count = max_count.max(count);
                            last_bi = bi;
                            count = 1;
                        } else {
                            count += 1;
                        }
                    }
                    bin_counts.push((last_bi, count));

                    if max_count > K {
                        // Nothing works here
                        mask = 0;
                        continue 'seed;
                    }

                    for &(bi, _count) in &bin_counts {
                        // update mask
                        mask &=
                            sux::traits::BitVecValueOps::<usize>::get_value(&non_full_bins, bi, 64)
                                as u64;
                    }
                    // eprintln!("Bin sizes for seed {seed}: {}", bins.len());
                }
                if (mask >> (seed % 64)) & 1 == 0 {
                    continue 'seed;
                }
                if i == 0 && (seed % 64) != 0 {
                    // For the first bucket, everything is still empty so no use in shifting.
                    continue;
                }

                // More refined check that considers within-bucket collisions.
                counts.fill(0);
                for &(bi, count) in &bin_counts {
                    let bi = bi + (seed & SEED_MASK as usize) as usize;
                    let s = unsafe { *bin_sizes.get_unchecked(bi) } as usize;
                    if s + count > K {
                        continue 'seed;
                    }
                    if !greedy {
                        unsafe {
                            *counts.get_unchecked_mut(s) += 1;
                            *counts.get_unchecked_mut(s + count) -= 1;
                        }
                    }
                }

                let score = if !greedy && !consensus {
                    for i in 1..=K {
                        counts[i] += counts[i - 1];
                    }
                    debug_assert_eq!(counts[K], 0);
                    f(&counts[..K])
                } else {
                    0
                };
                if consensus {
                    vals.push((0, score, seed));
                } else {
                    if score < vals[0].1 {
                        vals[0] = (0, score, seed);
                    }
                }
                if greedy && (consensus || tries[i] < vals.len() as u8) {
                    break;
                }
            }
            if vals.len() > 1 {
                vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
            }
            let best = vals[tries[i] as usize];
            if consensus && best.0 > 0 {
                backtracks += 1;

                if backtracks > n as u32 {
                    // Too many backtracks, give up.
                    return None;
                }

                // Backtrack 1 step.
                tries[i] = 0;
                if i > 0 {
                    assert!(tries[i - 1] < 255);
                    tries[i - 1] += 1;
                    // Reduce the bucket size of previous seed of parent.

                    let start = bucket_starts[idx - 1];
                    let end = bucket_starts[idx];
                    let bucket = &keys[start..end];

                    let seed = u64::from_be_bytes(seeds[i - 1..i + 7].try_into().unwrap());

                    for &key in bucket {
                        let bi = self.to_bin(key, seed);
                        assert!(bin_sizes[bi] > 0);
                        bin_sizes[bi] -= 1;
                        non_full_bins.set(bi, true);
                    }

                    seeds[i + 6] = 0;
                    i -= 1;
                }
                if i == 0 {
                    seeds[6] += 1;
                }
                continue;
            }

            if best.0 > 0 {
                if !bump {
                    // Unfixable collision found.
                    return None;
                }
                bumped += bucket.len();
                i += 1;
                bumped_keys.extend_from_slice(bucket);
                seeds[idx + 7] = 255;
                continue;
            }

            let seed = best.2;
            assert!(seed < 256);
            seeds[idx + 7] = seed as u8;

            for &key in bucket {
                let bi = self.to_bin(key, seed_offset + seed as u64);
                assert!( bin_sizes[bi] < K as u8, "collision at {i} size {len} seed {seed} offset {seed_offset:>0x} best {best:?} bin id {bi} bin size {}", bin_sizes[bi]);
                bin_sizes[bi] += 1;
                if bin_sizes[bi] == K as u8 {
                    non_full_bins.set(bi, false);
                }
            }

            i += 1;
        }

        let _duration = start.elapsed();
        // eprintln!(
        //     "alpha: {:>4.2}, bits/key: {:<6}, mode: {:>27} BTs {backtracks:>7} Bumped {bumped:>7} ({:.4}%) {:>6?}ms",
        //     self.alpha,
        //     self.bits_per_key,
        //     format!("{MODE:?}"),
        //     bumped as f32 / n as f32 * 100.0,
        //     duration.as_millis()
        // );

        self.seeds = seeds;
        if bumped > 0 {
            log::warn!(
                "Bumping {bumped} keys = {:>.1}%",
                bumped as f32 / n as f32 * 100.0
            );
            if bumped > n / 10 {
                eprintln!(
                    "Bumping {bumped} keys = {:>.1}%",
                    bumped as f32 / n as f32 * 100.0
                );
            }
            assert!(bumped < n / 2, "Too many bumped keys: {bumped} out of {n}.");
            self.bumped = Some(Box::new(Self::new::<T>(
                // use a lazy load factor for fallback
                0.5,
                // double the bits/key for bumped keys
                2.0 * self.bits_per_key,
                &bumped_keys,
            )?));
        }
        Some(())
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins + self.bumped.as_ref().map_or(0, |b| b.num_bins())
    }

    pub fn bits_used(&self) -> usize {
        self.seeds.len() * 8 + self.bumped.as_ref().map_or(0, |b| b.bits_used())
    }

    pub fn num_bumped(&self) -> usize {
        self.bumped.as_ref().map_or(0, |b| b.n)
    }

    #[inline(always)]
    pub fn prefetch<T: Key>(&self, key: T) {
        let bi = self.to_bucket(key);
        prefetch_index(&self.seeds, bi + 7);
    }
    #[inline(always)]
    pub fn get<T: Key>(&self, key: T) -> usize {
        let bi = self.to_bucket(key);
        let seed = if matches!(MODE, Mode::Consensus) {
            u64::from_be_bytes(self.seeds[bi..bi + 8].try_into().unwrap())
        } else {
            unsafe { *self.seeds.get_unchecked(bi + 7) as u64 }
        };
        if matches!(
            MODE,
            Mode::LinearBump | Mode::SortBump | Mode::LinearBumpGreedy | Mode::SortBumpGreedy
        ) && seed == 255
        {
            cold_path();
            self.num_bins + self.bumped.as_ref().unwrap().get(key)
        } else {
            self.to_bin(key, seed)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    fn test_config<const MODE: Mode, const K: usize>(keys: &[u32], alpha: f32, bits_per_key: f32)
    where
        [(); K + 1]:,
    {
        let Some(kphf) = KptrHash::<MODE, K>::new(alpha, bits_per_key, keys) else {
            return;
        };
        let mut cnt = vec![0; kphf.num_bins()];
        for key in keys {
            let bi = kphf.get(*key);
            cnt[bi] += 1;
            assert!(cnt[bi] <= K, "bin {bi} has {} keys", cnt[bi]);
        }
    }

    #[test]
    fn correctness() {
        let n = 1_000_000;
        let keys = gen_keys(n);

        for alpha in [0.90] {
            for bits_per_key in [0.40, 0.35, 0.3, 0.25] {
                test_config::<{ Mode::Linear }, 8>(&keys, alpha, bits_per_key);
                test_config::<{ Mode::LinearBump }, 8>(&keys, alpha, bits_per_key);
                test_config::<{ Mode::Sort }, 8>(&keys, alpha, bits_per_key);
                test_config::<{ Mode::SortBump }, 8>(&keys, alpha, bits_per_key);
                test_config::<{ Mode::Consensus }, 8>(&keys, alpha, bits_per_key);
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

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
}

const BIN_SIZE: usize = 8;
const BITS_PER_BUCKET: usize = 8;
const PADDING: usize = 100;

#[derive(Debug, Copy, Clone)]
pub enum Mode {
    /// Process buckets left to right
    Linear,
    /// Process buckets left to right, and allow bumping
    LinearBump,
    /// Process buckets large to small
    Sort,
    /// Process buckets large to small, and allow bumping
    SortBump,
    /// Process buckets left to right, and backtrack
    Consensus,
}

impl KptrHash {
    pub fn new(alpha: f32, bits_per_key: f32, keys: &[T], mode: Mode) -> Option<Self> {
        let start = std::time::Instant::now();
        let sort = matches!(mode, Mode::Sort | Mode::SortBump);
        let consensus = matches!(mode, Mode::Consensus);
        let bump = matches!(mode, Mode::LinearBump | Mode::SortBump);

        let k = BIN_SIZE;
        // bits per bucket
        let n = keys.len();
        // bins
        let num_bins = ((n as f32 / alpha) as usize).div_ceil(k);
        // target metadata bits
        let m = (n as f32 * bits_per_key) as usize;
        // #buckets
        let num_buckets = m.div_ceil(BITS_PER_BUCKET);

        // 1. sort the keys by part
        let mut keys = keys.to_vec();
        // keys.sort_unstable();
        keys.sort_unstable_by_key(|&key| (to_part(key, num_buckets, k), key));
        keys.dedup();

        // 2. split into p parts
        let mut part_sizes = vec![0; num_buckets];
        for key in &*keys {
            let p = to_part(*key, num_buckets, k);
            part_sizes[p] += 1;
        }
        let mut part_starts = vec![0; num_buckets + 1];
        for i in 1..=num_buckets {
            part_starts[i] = part_starts[i - 1] + part_sizes[i - 1];
        }

        // 3. Sort parts by decreasing size
        let mut perm = (0..num_buckets).collect::<Vec<_>>();

        if sort {
            perm.sort_by_key(|&i| std::cmp::Reverse(part_sizes[i]));
        }

        // 4. init bin sizes
        let mut bin_sizes = vec![0u8; num_bins + PADDING];
        let mut seeds = vec![0u8; num_buckets + 7];
        let mut tries = vec![0u8; num_buckets];

        let mut bumped = 0;
        let mut backtracks = 0;
        let mut i = 0;

        let mut lg: [f64; BIN_SIZE + 1] = from_fn(|i| (i as f64).log2());
        lg[0] = f64::MIN;

        let mut bumped_keys = vec![];

        while i < num_buckets {
            let idx = perm[i];
            let start = part_starts[idx];
            let end = part_starts[idx + 1];
            let len = end - start;

            let num_full = bin_sizes.iter().filter(|&&x| x == k as u8).count();

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
            let maybe_optimal = |counts: &[usize]| -> i64 {
                if counts[BIN_SIZE] > 0 {
                    return i64::MAX;
                }
                let q = counts
                    .iter()
                    .enumerate()
                    .map(|(size, &cnt)| cnt as f64 * lg[BIN_SIZE - size])
                    .sum::<f64>();
                (-q) as i64
            };
            let f = pow(7);
            // let f = maybe_optimal;

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

            for seed in 0..(1 << BITS_PER_BUCKET) - if bump { 1 } else { 0 } {
                let mut counts = vec![0; k + 1];
                for key in part {
                    let bi = to_bin(*key, seed_offset + seed as u64, num_bins);
                    counts[(bin_sizes[bi] as usize).min(k)] += 1;
                    // update the bin_size to handle self-collisions
                    bin_sizes[bi] += 1;
                }
                let score = f(&counts);
                vals.push((counts[BIN_SIZE], score, seed));
                for &key in part {
                    let bi = to_bin(key, seed_offset + seed as u64, num_bins);
                    bin_sizes[bi] -= 1;
                }
            }
            vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let best = vals[tries[i] as usize];
            if consensus && best.0 > 0 {
                backtracks += 1;
                // Backtrack 1 step.
                tries[i] = 0;
                if i > 0 {
                    assert!(tries[i - 1] < 255);
                    tries[i - 1] += 1;
                    // Reduce the bucket size of previous seed of parent.

                    let start = part_starts[idx - 1];
                    let end = part_starts[idx];
                    let len = end - start;
                    let part = &keys[start..end];

                    let seed = u64::from_be_bytes(seeds[i - 1..i + 7].try_into().unwrap());

                    for &key in part {
                        let bi = to_bin(key, seed, num_bins);
                        assert!(bin_sizes[bi] > 0);
                        bin_sizes[bi] -= 1;
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
                bumped += part.len();
                i += 1;
                bumped_keys.extend_from_slice(part);
                seeds[idx + 7] = 255;
                continue;
            }

            let seed = best.2;
            assert!(seed < 256);
            seeds[idx + 7] = seed as u8;

            for &key in part {
                let bi = to_bin(key, seed_offset + seed as u64, num_bins);
                assert!( bin_sizes[bi] < k as u8, "collision at {i} size {len} seed {seed} offset {seed_offset:>0x} best {best:?} bin id {bi} bin size {}", bin_sizes[bi]);
                bin_sizes[bi] += 1;
            }

            i += 1;
        }

        let duration = start.elapsed();
        eprintln!(
            "alpha: {alpha:>4.2}, bits/key: {bits_per_key:<6}, mode: {:>27}, bits/key: {:.4} BTs {backtracks:>7} Bumped {bumped:>7} ({:.4}%) {:>6?}ms",
            format!("{mode:?}"),
            (m as f32) / (n as f32),
            bumped as f32 / n as f32 * 100.0,
            duration.as_millis()
        );

        Some(Self {
            alpha,
            bits_per_key,
            mode,
            k,
            n,
            num_bins,
            num_buckets,
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

    pub fn bins(&self) -> usize {
        self.num_bins + self.bumped.as_ref().map_or(0, |b| b.bins())
    }

    pub fn get(&self, key: T) -> usize {
        let part = to_part(key, self.num_buckets, self.k);
        let seed = match self.mode {
            Mode::Consensus => u64::from_be_bytes(self.seeds[part..part + 8].try_into().unwrap()),
            _ => self.seeds[part + 7] as u64,
        };
        if matches!(self.mode, Mode::LinearBump | Mode::SortBump) && seed == 255 {
            self.num_bins + self.bumped.as_ref().unwrap().get(key)
        } else {
            to_bin(key, seed, self.num_bins)
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
                // Mode::Sort,
                Mode::SortBump,
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

//! Reproducible incremental quotienting benchmark with nonzero 62-bit keys (single thread).
use static_hash_set::HashSet;
use std::{hint::black_box, time::Instant};
type T = u64;
// Older tables in this comparison still require 62-bit keys.
const KEY_MASK: u64 = (1 << 62) - 1;
mod cuckoo {
    pub use static_hash_set::cuckoo::Mode;
}
mod traits {
    pub use static_hash_set::HashSet;
}
#[path = "../../eval/quotienting/baseline.rs"]
mod baseline;
#[path = "../../eval/quotienting/fast-range.rs"]
mod fast_range;
#[path = "../../eval/quotienting/power2.rs"]
mod power2;
#[path = "../../eval/quotienting/power2-zero.rs"]
mod power2_zero;

trait Table: HashSet {
    fn lookup(&self, key: u64, lazy: bool) -> bool;
    fn token(&self, key: u64, lazy: bool) -> usize;
    fn lookup_token(&self, key: u64, token: usize, lazy: bool) -> bool;
    fn verify(&self, n: usize);
}
macro_rules! table {
    ($ty:ty) => {
        impl Table for $ty {
            fn lookup(&self, k: u64, lazy: bool) -> bool {
                if lazy {
                    self.contains_lazy(k)
                } else {
                    self.contains_eager(k)
                }
            }
            fn token(&self, k: u64, lazy: bool) -> usize {
                if lazy {
                    self.prefetch_first(k)
                } else {
                    self.prefetch_both(k)
                }
            }
            fn lookup_token(&self, k: u64, t: usize, lazy: bool) -> bool {
                if lazy {
                    self.contains_lazy_with_token(k, t)
                } else {
                    self.contains_eager_with_token(k, t)
                }
            }
            fn verify(&self, n: usize) {
                assert_eq!(self.len(), n);
                let mut seen = vec![false; n];
                for k in self.iter() {
                    let i = unmix(k).checked_sub(1).expect("unexpected zero key") as usize;
                    assert!(i < n && !seen[i], "invalid or duplicate iteration key");
                    seen[i] = true;
                }
                assert!(seen.into_iter().all(|x| x));
            }
        }
    };
}
table!(baseline::CuckooSet<0>);
table!(power2::CuckooSet<0>);
table!(fast_range::CuckooSet<0>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 0, false>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 8, false>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 8, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 16, false>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 16, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 32, false>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 32, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 0, false, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 8, false, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 8, true, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 16, false, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 16, true, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 32, false, true>);
table!(static_hash_set::cuckoo_quotienting::CuckooSet<0, 32, true, true>);
table!(power2_zero::CuckooSet<0, 0, false>);
table!(power2_zero::CuckooSet<0, 8, false>);
table!(power2_zero::CuckooSet<0, 8, true>);
table!(power2_zero::CuckooSet<0, 16, false>);
table!(power2_zero::CuckooSet<0, 16, true>);
table!(power2_zero::CuckooSet<0, 32, false>);
table!(power2_zero::CuckooSet<0, 32, true>);
fn mix(mut x: u64) -> u64 {
    x &= KEY_MASK;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9) & KEY_MASK;
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb) & KEY_MASK;
    x ^ (x >> 31)
}
fn unmix(mut x: u64) -> u64 {
    x ^= (x >> 31) ^ (x >> 62);
    x = x.wrapping_mul(0x319642b2d24d8ec3) & KEY_MASK;
    x ^= (x >> 27) ^ (x >> 54);
    x = x.wrapping_mul(0x96de1b173f119089) & KEY_MASK;
    x ^ (x >> 30) ^ (x >> 60)
}
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    // Sampling and shuffling use a full-width random value: range reduction
    // below takes the high 64 bits of a product.
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn count<H: Table>(set: &H, qs: &[u64], lazy: bool, prefetch: bool) -> usize {
    let mut found = 0;
    if prefetch {
        let mut tokens = [0; 32];
        for i in 0..qs.len().min(32) {
            tokens[i] = set.token(qs[i], lazy);
        }
        for (i, &k) in qs.iter().enumerate() {
            found += set.lookup_token(k, tokens[i % 32], lazy) as usize;
            if i + 32 < qs.len() {
                tokens[i % 32] = set.token(qs[i + 32], lazy);
            }
        }
    } else {
        for &k in qs {
            found += set.lookup(k, lazy) as usize;
        }
    }
    black_box(found)
}
fn bench<H: Table>(
    stage: &str,
    n: usize,
    q: usize,
    repeats: usize,
    alpha: f32,
    constructor: impl FnOnce(&[u64]) -> H,
) {
    let keys: Vec<_> = (1..=n as u64).map(mix).collect();
    let start = Instant::now();
    let set = constructor(&keys);
    let build = start.elapsed().as_secs_f64();
    drop(keys);
    eprintln!(
        "{stage}: built in {build:.3}s, alpha={:.6}, memory={:.6}",
        set.load_factor(),
        set.allocation_size() as f64 / (n * 8) as f64
    );
    set.verify(n);
    for k in (1..=n as u64).map(mix) {
        assert!(set.lookup(k, false));
        assert!(set.lookup(k, true));
    }
    for positive in [1, 50, 99] {
        let expected = q * positive / 100;
        let mut state = 42;
        let mut qs: Vec<_> = (0..q)
            .map(|i| {
                let idx = ((random(&mut state) as u128 * n as u128) >> 64) as u64 + 1;
                mix(idx + if i < expected { 0 } else { n as u64 })
            })
            .collect();
        for i in (1..q).rev() {
            let j = (random(&mut state) as u128 * (i + 1) as u128 >> 64) as usize;
            qs.swap(i, j);
        }
        for lazy in [false, true] {
            for prefetch in [false, true] {
                assert_eq!(count(&set, &qs, lazy, prefetch), expected);
                let mut best = f64::INFINITY;
                for _ in 0..repeats {
                    let start = Instant::now();
                    let found = count(black_box(&set), black_box(&qs), lazy, prefetch);
                    let elapsed = start.elapsed().as_secs_f64();
                    assert_eq!(found, expected);
                    best = best.min(elapsed * 1e9 / q as f64);
                }
                println!(
                    "{stage},{},{n},{alpha},{:.8},{:.8},{build:.6},{},{positive},{q},{repeats},{best:.3}",
                    if lazy { "lazy" } else { "eager" },
                    set.load_factor(),
                    set.allocation_size() as f64 / (n * 8) as f64,
                    if prefetch { "prefetch2" } else { "loop" }
                );
            }
        }
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let arg = |flag: &str, default: &str| {
        args.windows(2)
            .find(|p| p[0] == flag)
            .map_or(default.to_string(), |p| p[1].clone())
    };
    let stage = arg("--stage", "baseline");
    let n = arg("--n", "67108864").parse().unwrap();
    let q = arg("--queries", "1048576").parse().unwrap();
    let repeats = arg("--repeats", "3").parse().unwrap();
    let alpha: f32 = arg("--alpha", "0.99").parse().unwrap();
    assert!(n > 0 && q > 0 && repeats > 0 && alpha > 0.0 && alpha < 1.0);
    assert!(
        (n as u128) * 2 <= KEY_MASK as u128,
        "keys and negative queries must fit in 62 bits"
    );
    println!(
        "stage,mode,n,target_alpha,alpha,memory_ratio,build_seconds,metric,positive_percent,queries,repeats,ns_per_query"
    );
    match stage.as_str() {
        "baseline" => bench(&stage, n, q, repeats, alpha, |keys| {
            baseline::CuckooSet::<0>::new(1.0 / alpha, keys)
        }),
        "power2" => bench(&stage, n, q, repeats, alpha, |keys| {
            power2::CuckooSet::<0>::new(1.0 / alpha, keys)
        }),
        "zero" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 0, false>::new(1.0 / alpha, keys)
        }),
        "prefix8-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 8, false>::new(1.0 / alpha, keys)
        }),
        "prefix8-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 8, true>::new(1.0 / alpha, keys)
        }),
        "prefix16-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 16, false>::new(1.0 / alpha, keys)
        }),
        "prefix16-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 16, true>::new(1.0 / alpha, keys)
        }),
        "prefix32-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 32, false>::new(1.0 / alpha, keys)
        }),
        "prefix32-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 32, true>::new(1.0 / alpha, keys)
        }),
        "fast-range" => bench(&stage, n, q, repeats, alpha, |keys| {
            fast_range::CuckooSet::<0>::new(1.0 / alpha, keys)
        }),
        "power2-zero" => bench(&stage, n, q, repeats, alpha, |keys| {
            power2_zero::CuckooSet::<0, 0, false>::new(1.0 / alpha, keys)
        }),
        "power2-prefix8-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            power2_zero::CuckooSet::<0, 8, false>::new(1.0 / alpha, keys)
        }),
        "power2-prefix8-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            power2_zero::CuckooSet::<0, 8, true>::new(1.0 / alpha, keys)
        }),
        "power2-prefix16-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            power2_zero::CuckooSet::<0, 16, false>::new(1.0 / alpha, keys)
        }),
        "power2-prefix16-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            power2_zero::CuckooSet::<0, 16, true>::new(1.0 / alpha, keys)
        }),
        "power2-prefix32-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            power2_zero::CuckooSet::<0, 32, false>::new(1.0 / alpha, keys)
        }),
        "power2-prefix32-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            power2_zero::CuckooSet::<0, 32, true>::new(1.0 / alpha, keys)
        }),
        "ef-zero" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 0, false, true>::new(
                1.0 / alpha,
                keys,
            )
        }),
        "ef-prefix8-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 8, false, true>::new(
                1.0 / alpha,
                keys,
            )
        }),
        "ef-prefix8-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 8, true, true>::new(
                1.0 / alpha,
                keys,
            )
        }),
        "ef-prefix16-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 16, false, true>::new(
                1.0 / alpha,
                keys,
            )
        }),
        "ef-prefix16-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 16, true, true>::new(
                1.0 / alpha,
                keys,
            )
        }),
        "ef-prefix32-scalar" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 32, false, true>::new(
                1.0 / alpha,
                keys,
            )
        }),
        "ef-prefix32-simd" => bench(&stage, n, q, repeats, alpha, |keys| {
            static_hash_set::cuckoo_quotienting::CuckooSet::<0, 32, true, true>::new(
                1.0 / alpha,
                keys,
            )
        }),
        _ => panic!("unknown stage"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_pipeline_counts_all_queries_including_short_tails() {
        let keys: Vec<_> = (1..=64).map(mix).collect();
        let set = static_hash_set::cuckoo_quotienting::CuckooSet::<0, 16, true>::new(1.1, &keys);
        for q in [0usize, 1, 31, 32, 33, 65] {
            let queries: Vec<_> = (0..q)
                .map(|i| {
                    mix(if i % 2 == 0 {
                        (i % 64 + 1) as u64
                    } else {
                        1000 + i as u64
                    })
                })
                .collect();
            for lazy in [false, true] {
                for prefetch in [false, true] {
                    assert_eq!(count(&set, &queries, lazy, prefetch), q.div_ceil(2));
                }
            }
        }
    }
}

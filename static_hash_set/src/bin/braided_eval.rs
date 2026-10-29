//! Single-threaded evaluator. Example:
//! cargo run --release --no-default-features --bin braided_eval -- --n 67108864
use static_hash_set::braided_cuckoo::{BraidedCuckooSet, Mode};
use std::{hint::black_box, time::Instant};

fn option(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2).find(|p| p[0] == flag).map_or(default, |p| {
        p[1].parse().expect("expected an integer option value")
    })
}

// SplitMix's bijective finalizer gives deterministic distinct keys.
fn permute(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    permute(*state)
}

fn count<const MODE: u8>(set: &BraidedCuckooSet<MODE>, queries: &[u64], metric: &str) -> usize {
    const AHEAD: usize = 32;
    let mut found = 0;
    match metric {
        "loop" => {
            for &key in queries {
                found += set.contains(key) as usize;
            }
        }
        "prefetch" => {
            for (i, &key) in queries.iter().enumerate() {
                if i + AHEAD < queries.len() {
                    black_box(set.prefetch(queries[i + AHEAD]));
                }
                found += set.contains(key) as usize;
            }
        }
        "prefetch2" => {
            let mut tokens = [0; AHEAD];
            for (i, &key) in queries.iter().take(AHEAD).enumerate() {
                tokens[i] = set.prefetch(key);
            }
            for (i, &key) in queries.iter().enumerate() {
                found += set.contains_with_token(key, tokens[i % AHEAD]) as usize;
                if i + AHEAD < queries.len() {
                    tokens[i % AHEAD] = set.prefetch(queries[i + AHEAD]);
                }
            }
        }
        _ => unreachable!(),
    }
    black_box(found)
}

fn bench<const MODE: u8>(
    set: &BraidedCuckooSet<MODE>,
    n: usize,
    query_count: usize,
    repeats: usize,
    build_ns: f64,
) {
    use static_hash_set::HashSet;
    let mut state = 42;
    for positive in [1usize, 50, 99] {
        let expected = query_count * positive / 100;
        let mut queries: Vec<_> = (0..query_count)
            .map(|i| {
                let idx = ((random(&mut state) as u128 * n as u128) >> 64) as u64;
                permute(if i < expected { idx } else { n as u64 + idx })
            })
            .collect();
        // Shuffle so the branch predictor cannot infer the hit rate from order.
        for i in (1..queries.len()).rev() {
            let j = ((random(&mut state) as u128 * (i + 1) as u128) >> 64) as usize;
            queries.swap(i, j);
        }
        for metric in ["loop", "prefetch", "prefetch2"] {
            assert_eq!(
                count(set, &queries, metric),
                expected,
                "warmup/correctness: {metric}"
            );
            let mut best = f64::INFINITY;
            for _ in 0..repeats {
                let start = Instant::now();
                let found = count(black_box(set), black_box(&queries), metric);
                let elapsed = start.elapsed().as_secs_f64();
                assert_eq!(found, expected, "query count: {metric}");
                best = best.min(elapsed * 1e9 / query_count as f64);
            }
            println!(
                "{},{n},{:.6},{build_ns:.3},{metric},{positive},{query_count},{repeats},{best:.3}",
                set.name(),
                set.load_factor()
            );
        }
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--help") {
        println!(
            "braided_eval [--n N] [--queries Q] [--repeats R] [--attempts A] [--greedy] [--diagnose]\n\
            Default: n=2^20, queries=2^20, repeats=3, attempts=1.\n\
            --diagnose: report one complete greedy phase at a time, then exit.\n\
            --greedy: disable local displacement repairs. All inputs are deterministic and unique."
        );
        return;
    }
    let n = option(&args, "--n", 1 << 20);
    let query_count = option(&args, "--queries", 1 << 20);
    let repeats = option(&args, "--repeats", 3);
    let attempts = option(&args, "--attempts", 1);
    assert!(n > 0 && query_count > 0 && repeats > 0 && attempts > 0);
    let keys: Vec<_> = (0..n as u64).map(permute).collect();
    eprintln!("n={n}, queries={query_count}, repeats={repeats}, threads=1");
    let layers =
        BraidedCuckooSet::<{ Mode::Lazy as u8 }>::layer_bins_for_len(n).expect("capacity overflow");
    let slots = layers.iter().sum::<usize>() * 8;
    eprintln!(
        "layer buckets={layers:?}, slots={slots}, load={:.8}",
        n as f64 / slots as f64
    );
    let start = Instant::now();
    if args.iter().any(|arg| arg == "--diagnose") {
        for stats in BraidedCuckooSet::<{ Mode::Lazy as u8 }>::diagnose(&keys) {
            println!("{stats:?}");
        }
        eprintln!("diagnosis: {:.3}s", start.elapsed().as_secs_f64());
        return;
    }
    let repair = !args.iter().any(|arg| arg == "--greedy");
    let (set, stats) = match BraidedCuckooSet::<{ Mode::Lazy as u8 }>::try_from_unique_keys(
        &keys, attempts, repair,
    ) {
        Ok(result) => result,
        Err(error) => {
            eprintln!(
                "construction failed after {:.3}s: {error:?}",
                start.elapsed().as_secs_f64()
            );
            std::process::exit(1);
        }
    };
    let build_seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "build: {build_seconds:.3}s, {:.1} ns/key, {:.1} MiB, {stats:?}",
        build_seconds * 1e9 / n as f64,
        set.allocation_size() as f64 / (1 << 20) as f64
    );
    // Check every inserted key outside the timed region, including zero.
    for &key in &keys {
        assert!(set.contains(key), "missing key {key}");
    }
    assert_eq!(set.len(), n);
    drop(keys);
    println!(
        "set,n,load_factor,build_ns_per_key,metric,positive_percent,queries,repeats,ns_per_query"
    );
    bench(
        &set,
        n,
        query_count,
        repeats,
        build_seconds * 1e9 / n as f64,
    );
    let eager = set.into_mode::<{ Mode::Eager as u8 }>();
    bench(
        &eager,
        n,
        query_count,
        repeats,
        build_seconds * 1e9 / n as f64,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<const MODE: u8>() {
        let keys: Vec<_> = (0..100).map(permute).collect();
        let set = BraidedCuckooSet::<MODE>::new(&keys);
        for n in [0, 1, 31, 32, 33, 127, 257] {
            let queries: Vec<_> = (0..n as u64).map(permute).collect();
            for metric in ["loop", "prefetch", "prefetch2"] {
                assert_eq!(count(&set, &queries, metric), n.min(100));
            }
        }
    }
    #[test]
    fn query_loops_include_short_inputs_and_tails() {
        check::<{ Mode::Lazy as u8 }>();
        check::<{ Mode::Eager as u8 }>();
    }
}

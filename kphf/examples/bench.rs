#![allow(incomplete_features)]
#![feature(adt_const_params, generic_const_exprs)]
use std::cell::RefCell;

use kphf::{space_lower_bound, KptrHash, Mode};
use rand::seq::SliceRandom;

fn gen_keys(n: usize) -> Vec<u32> {
    let mut keys = std::collections::HashSet::with_capacity(n);
    let mut buf = vec![0u32; 1024];
    while keys.len() < n {
        rand::fill(&mut buf[..]);
        keys.extend(buf.iter().copied());
    }
    let mut keys: Vec<u32> = keys.into_iter().take(n).collect();
    keys.sort_unstable();
    keys.dedup();
    keys.shuffle(&mut rand::rng());
    keys
}

thread_local! {
    static CSV_WRITER: RefCell<csv::Writer<std::io::Stdout>> =
        RefCell::new(csv::Writer::from_writer(std::io::stdout()));
}

#[derive(serde::Serialize)]
struct Result {
    n: usize,
    alg: &'static str,
    alpha: f32,
    factor: f32,
    target_bits_per_key: f32,
    repeat: usize,
    build_ns: f32,
    bumped_frac: f32,
    actual_bits_per_key: f32,
    actual_alpha: f32,
    loop_ns: f32,
    throughput_ns: f32,
}

const REPEATS: usize = 1;
const NS: [usize; 2] = [10_000_000, 100_000_000];
const ALPHAS: [f32; 4] = [0.7, 0.8, 0.9, 0.99];
const FACTORS: [f32; 7] = [2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0];
const QUERY: bool = true;

// const NS: [usize; 1] = [10_000_000];
// const ALPHAS: [f32; 3] = [0.7, 0.8, 0.9];
// const FACTORS: [f32; 5] = [1.6, 1.5, 1.4, 1.3, 1.2];
// const QUERY: bool = false;

fn bench<const MODE: Mode, const K: usize>()
where
    [(); K + 1]:,
{
    for &n in &NS {
        let keys = std::array::from_fn::<_, REPEATS, _>(|_| gen_keys(n));
        for &alpha in &ALPHAS {
            let lb = space_lower_bound(K, alpha);
            for &factor in &FACTORS {
                let target_bits_per_key = lb * factor;

                for (repeat, keys) in std::iter::zip(0..REPEATS, &keys) {
                    // Construction
                    let t0 = std::time::Instant::now();
                    let kphf = KptrHash::<MODE, K>::new(alpha, target_bits_per_key, &keys).unwrap();
                    let build_ns = t0.elapsed().as_nanos() as f32 / n as f32;

                    // Space
                    let actual_bits_per_key = kphf.bits_used() as f32 / n as f32;
                    let bumped_frac = kphf.num_bumped() as f32 / n as f32;
                    let actual_alpha = n as f32 / (kphf.num_bins() as f32 * K as f32);

                    // Queries
                    let mut loop_ns = 0.0;
                    let mut throughput_ns = 0.0;
                    if QUERY {
                        let start = std::time::Instant::now();
                        let mut c = 0;
                        for &key in keys {
                            c += kphf.get(key);
                        }
                        std::hint::black_box(c);
                        loop_ns = start.elapsed().as_nanos() as f32 / n as f32;

                        let start = std::time::Instant::now();
                        let lookahead = 32;
                        let mut c = 0;
                        for i in 0..keys.len().saturating_sub(lookahead) {
                            kphf.prefetch(keys[i + lookahead]);
                            c += kphf.get(keys[i]) as usize;
                        }
                        std::hint::black_box(c);
                        throughput_ns = start.elapsed().as_nanos() as f32 / n as f32;
                    }

                    let result = Result {
                        n,
                        alg: std::any::type_name_of_val(&kphf),
                        alpha,
                        factor,
                        target_bits_per_key,
                        repeat,
                        build_ns,
                        bumped_frac,
                        actual_bits_per_key,
                        actual_alpha,
                        loop_ns,
                        throughput_ns,
                    };
                    CSV_WRITER.with_borrow_mut(|w| {
                        w.serialize(&result).unwrap();
                        w.flush().unwrap();
                    });
                }
            }
        }
    }
}

fn main() {
    bench::<{ Mode::SortBumpGreedy }, 4>();
    bench::<{ Mode::SortBump }, 4>();

    bench::<{ Mode::SortBumpGreedy }, 8>();
    bench::<{ Mode::SortBump }, 8>();

    bench::<{ Mode::SortBumpGreedy }, 16>();
    bench::<{ Mode::SortBump }, 16>();
}

#![allow(unused)]

mod bin;
mod cuckoo;
#[cfg(feature = "extc")]
mod ekphf;
#[cfg(feature = "extc")]
mod fph_table;
mod kphf_set;
mod kphf_trait;
#[cfg(feature = "extc")]
mod mapembed;
mod mock_hashset;
mod phf_set;
mod phf_trait;
#[cfg(test)]
mod test;
mod traits;
mod u64_hashset;

use std::{cell::RefCell, hint::black_box};

use cuckoo::{CuckooSet, Mode};
#[cfg(feature = "extc")]
use engineering_kphf::{Hd8Set, Tbb84pSet, Tbb85Set};
#[cfg(feature = "extc")]
use fph_table::FphDynSet;
#[cfg(feature = "kphf")]
use kphf::{self, space_lower_bound, KptrHash};
use kphf_set::KphfSet;
use mock_hashset::MockHashSet;
use phf_set::PhfSet;
use rand::seq::IndexedRandom;
use traits::HashSet;
use u64_hashset::U64HashSet;

// type T = u32;
// const BUCKET_SIZE: usize = 16;
// type S = wide::i32x8;
type T = u64;
const BIN_SIZE: usize = 8;
type S = wide::i64x4;
use bin::Bin;

const QUERIES: usize = 3_000_000;
const REPEATS: usize = 1;
const QUERY_REPEATS: usize = 3;
#[cfg(all(not(feature = "diffie"), not(feature = "floyd")))]
const THREADS: [usize; 2] = [1, 12];
#[cfg(feature = "diffie")]
const THREADS: [usize; 2] = [1, 192];
#[cfg(feature = "floyd")]
const THREADS: [usize; 2] = [1, 128];

// const THREADS: [usize; 0] = [];
const PERCENTILES: [f64; 3] = [0.01, 0.5, 0.99];
const MODES: [&str; 3] = ["loop", "prefetch", "prefetch2"];
// const MODES: [&str; 1] = ["prefetch"];

fn main() {
    let ns = (0..)
        .map(|i| (1_000_000. * 1.2f32.powi(i)) as usize)
        .take_while(|x| *x <= 1_000_000_000)
        .collect::<Vec<_>>();

    let hashers = vec![
        // Mock throughput hashset
        (
            |alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
                Some(Box::new(MockHashSet::new(1.0 / alpha, keys)))
            } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
            vec![1.0],
        ),
        // k-PHF-set
        // (
        //     |alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
        //         Some(Box::new(KphfSet::<
        //             KptrHash<{ kphf::Mode::SortBump as u8 }, BIN_SIZE>,
        //             BIN_SIZE,
        //         >::try_new(
        //             alpha,
        //             1.5 * space_lower_bound(BIN_SIZE, alpha),
        //             keys,
        //         )?))
        //     } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
        //     // vec![0.7, 0.8, 0.9, 0.99],
        //     // vec![0.9, 0.99],
        //     // vec![0.9, 0.99],
        //     vec![0.7, 0.9, 0.95],
        //     // vec![0.9],
        // ),
        // // PtrHash
        // (
        //     |_alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
        //         Some(Box::new(PhfSet::<phf_trait::PtrHash>::new(0.0, 0.0, keys)) as Box<dyn HashSet>)
        //     } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
        //     vec![0.99],
        // ),
        // // non-minimal PHast
        // (
        //     |_alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
        //         Some(Box::new(PhfSet::<phf_trait::PHast>::new(0.0, 0.0, keys)) as Box<dyn HashSet>)
        //     } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
        //     vec![0.98],
        // ),
        // // SwissTable
        // (
        //     |_alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
        //         Some(Box::new(
        //             hashbrown::HashSet::<T, gxhash::GxBuildHasher>::from_iter(keys.iter().cloned()),
        //         ))
        //     } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
        //     vec![0.5],
        // ),
        // // // U64HashSet
        // // (
        // //     |alpha: f32, keys: &[T]| Some(Box::new(U64HashSet::new(1. / alpha, keys))),
        // //     // vec![0.7, 0.8, 0.9, 0.95],
        // //     vec![0.7],
        // // ),
        // // Eager Cuckoo
        // (
        //     |alpha: f32, keys: &[T]| {
        //         Some(Box::new(CuckooSet::<{ Mode::Eager as u8 }>::new(
        //             1. / alpha,
        //             keys,
        //         )))
        //     },
        //     vec![0.99],
        // ),
        // // Lazy Cuckoo
        // (
        //     |alpha: f32, keys: &[T]| {
        //         Some(Box::new(CuckooSet::<{ Mode::Lazy as u8 }>::new(
        //             1. / alpha,
        //             keys,
        //         )))
        //     },
        //     vec![0.7],
        // ),
        // // FPH
        // (
        //     |alpha: f32, keys: &[T]| {
        //         Some(Box::new(FphDynSet::new(alpha, keys)?) as Box<dyn HashSet>)
        //     },
        //     vec![0.95],
        // ),
        // // // engineering k-PHF:
        // // // - Only use the Hash-displace variant with faster queries.
        // // // - Skip threshold-based-bumping variants
        // // #[cfg(feature = "ekphf")]
        // // (
        // //     |_alpha: f32, keys: &[T]| {
        // //         Some(
        // //             Box::new(KphfSet::<Hd8Set, BIN_SIZE>::try_new(0.0, 0.0, keys)?)
        // //                 as Box<dyn HashSet>,
        // //         )
        // //     },
        // //     vec![1.0],
        // // ),
        // // // MapEmbed
        // // (
        // //     |_alpha: f32, keys: &[T]| {
        // //         Some(Box::new(mapembed::MapEmbed::new(keys)?) as Box<dyn HashSet>)
        // //     },
        // //     vec![0.9],
        // // ),
    ];
    for repeat in 0..REPEATS {
        for &n in &ns {
            let bencher = Bencher::new(n);
            for (constructor, alphas) in &hashers {
                for &alpha in alphas {
                    bencher.bench(*constructor, alpha, repeat);
                }
            }
        }
    }
}

fn time<T>(mut f: impl FnMut() -> T) -> (f32, T) {
    let start = std::time::Instant::now();
    let out = f();
    let duration = start.elapsed();
    (duration.as_nanos() as f32, out)
}

pub struct Bencher {
    n: usize,
    keys: Vec<T>,
    queries: Vec<Vec<[Vec<T>; 5]>>,
}

#[derive(serde::Serialize)]
pub struct Result {
    h: String,
    pf: bool,
    n: usize,
    alpha: f32,
    repeat: usize,
    build: f32,
    load_factor: f32,
    bumped_frac: f32,
    overhead: f32,
    kphf_target_bits_per_key: f32,
    kphf_bits_per_key: f32,
    threads: usize,
    metric: &'static str,
    q01: f32,
    q10: f32,
    q50: f32,
    q90: f32,
    q99: f32,
}

thread_local! {
    static CSV_WRITER: RefCell<csv::Writer<std::io::Stdout>> =
        RefCell::new(csv::Writer::from_writer(std::io::stdout()));
}

impl Bencher {
    pub fn new(n: usize) -> Self {
        let mut keys = vec![0; n];
        rand::fill(&mut keys[..]);
        let mut queries = vec![
            vec![
                [
                    vec![0; QUERIES],
                    vec![0; QUERIES],
                    vec![0; QUERIES],
                    vec![0; QUERIES],
                    vec![0; QUERIES],
                ];
                REPEATS
            ];
            *THREADS.last().unwrap_or(&0)
        ];
        for threads_queries in &mut queries {
            for repeat_queries in threads_queries {
                let rng = &mut rand::rng();
                for (q, p) in std::iter::zip(&mut repeat_queries.iter_mut(), PERCENTILES) {
                    for x in q.iter_mut() {
                        if rand::random_bool(p) {
                            *x = *keys.choose(rng).unwrap();
                        } else {
                            *x = rand::random();
                        }
                    }
                }
            }
        }
        Self { n, keys, queries }
    }

    pub fn bench(
        &self,
        constructor: fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
        alpha: f32,
        repeat: usize,
    ) {
        let (build, Some(ref h)) = time(|| constructor(alpha, &self.keys)) else {
            return;
        };
        let name = h.name();
        let build = build / self.n as f32;
        let bumped_frac = h.bumped_frac();
        let bits_per_key = h.allocation_size() as f32 * 8.0 / self.n as f32;
        let kphf_target_bits_per_key = h.kphf_target_bits_per_key();
        let kphf_bits_per_key = h.kphf_size() as f32 * 8.0 / self.n as f32;
        let overhead = bits_per_key / T::BITS as f32;
        let load_factor = h.load_factor();
        let pf = h.has_prefetch();
        for &threads in &THREADS {
            for metric in MODES {
                for query_repeat in 0..QUERY_REPEATS {
                    let mut query = [0f32; 5];
                    for (qi, &_p) in PERCENTILES.iter().enumerate() {
                        let start = std::time::Instant::now();
                        let worker = |qs: &[T]| match metric {
                            "loop" => {
                                let c = h.count_loop(&qs);
                                black_box(c);
                            }
                            "prefetch" => {
                                let c = h.count_prefetch(&qs);
                                black_box(c);
                            }
                            "prefetch2" => {
                                let c = h.count_prefetch2(&qs);
                                black_box(c);
                            }
                            _ => unreachable!(),
                        };
                        if threads == 1 {
                            let qs = &self.queries[0][repeat][qi];
                            worker(qs);
                        } else {
                            std::thread::scope(|scope| {
                                for t in 0..threads {
                                    let qs = &self.queries[t][repeat][qi];
                                    scope.spawn(move || worker(qs));
                                }
                            });
                        }
                        query[qi] = start.elapsed().as_nanos() as f32 / (threads * QUERIES) as f32;
                    }
                    let result = Result {
                        h: name.to_string(),
                        pf,
                        n: self.n,
                        alpha,
                        repeat: repeat * QUERY_REPEATS + query_repeat,
                        threads,
                        metric,
                        build,
                        load_factor,
                        bumped_frac,
                        overhead,
                        kphf_target_bits_per_key,
                        kphf_bits_per_key,
                        // kphf_alpha,
                        q01: query[0],
                        q10: 0.0,
                        q50: query[1],
                        q90: 0.0,
                        q99: query[2],
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

#![allow(unused)]

mod bin;
mod braided_cuckoo;
mod cuckoo;
mod cuckoo_filtered;
mod cuckoo_growable;
mod cuckoo_inline_filters;
mod cuckoo_quotienting;
mod cuckoo_spill;
#[cfg(feature = "extc")]
mod ekphf;
#[cfg(feature = "bench")]
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
#[cfg(feature = "tpht")]
mod tinyptr;
#[cfg(feature = "tpht")]
mod tpht;
mod traits;
mod u64_hashset;

use std::{cell::RefCell, hint::black_box};

use braided_cuckoo::BraidedCuckooSet;
use cuckoo::{CuckooSet, Mode};
use cuckoo_filtered::FilteredCuckooSet;
use cuckoo_growable::GrowableRightCuckooSet;
use cuckoo_inline_filters::InlineFilterCuckooSet;
use cuckoo_quotienting::CuckooSet as CuckooQuotientSet;
use cuckoo_spill::SpillCuckooSet;
#[cfg(feature = "extc")]
use engineering_kphf::{Hd8Set, Tbb84pSet, Tbb85Set};
#[cfg(feature = "bench")]
use fph_table::FphDynSet;
#[cfg(feature = "kphf")]
use kphf::{self, KptrHash, space_lower_bound};
use kphf_set::KphfSet;
use mock_hashset::MockHashSet;
use phf_set::PhfSet;
use rand::seq::IndexedRandom;
use rayon::prelude::*;
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
const MODES: [&str; 6] = ["loop", "prefetch", "prefetch2", "batch8", "batch16", "batch32"];
// const MODES: [&str; 1] = ["prefetch"];

fn quick_bench() -> bool {
    std::env::args().any(|arg| arg == "--quick")
}

fn batch_bench() -> bool {
    std::env::args().any(|arg| arg == "--batch")
}

fn selected_threads() -> Vec<usize> {
    let args: Vec<_> = std::env::args().collect();
    if let Some(values) = args
        .windows(2)
        .find(|pair| pair[0] == "--threads")
        .map(|pair| {
            pair[1]
                .split(',')
                .map(|value| value.parse::<usize>().expect("--threads values must be integers"))
                .collect::<Vec<_>>()
        })
    {
        assert!(
            !values.is_empty() && values.iter().all(|&threads| threads > 0),
            "--threads values must be positive"
        );
        values
    } else if quick_bench() {
        vec![1]
    } else {
        THREADS.to_vec()
    }
}

fn implementation_selected(name: &str) -> bool {
    let args: Vec<_> = std::env::args().collect();
    let Some(index) = args.iter().position(|arg| arg == "--only") else {
        return !name.starts_with("quotient-")
            || (name.contains("onehot") && !name.split('-').any(|part| part == "ef"));
    };
    args.get(index + 1)
        .is_some_and(|names| names.split(',').any(|selected| selected == name))
}

fn main() {
    let quick = quick_bench();
    let growth = if quick { 1.4f64 } else { 1.2f64 };
    let max_n = if quick { 10_000_000 } else { 1_000_000_000 };
    let args = std::env::args().collect::<Vec<_>>();
    let ns_list = args.windows(2).find(|pair| pair[0] == "--ns").map(|pair| {
        pair[1]
            .split(',')
            .map(|n| {
                n.parse::<usize>()
                    .expect("--ns values must be positive integers")
            })
            .collect::<Vec<_>>()
    });
    let fixed_n = args.windows(2).find(|pair| pair[0] == "--n").map(|pair| {
        pair[1]
            .parse::<usize>()
            .expect("--n requires a positive integer")
    });
    let ns = if let Some(ns) = ns_list {
        assert!(
            !ns.is_empty() && ns.iter().all(|&n| n > 0),
            "--ns values must be positive"
        );
        ns
    } else if let Some(n) = fixed_n {
        assert!(n > 0, "--n must be positive");
        vec![n]
    } else {
        (0..)
            .map(|i| (1_000_000. * growth.powi(i)) as usize)
            .take_while(|&n| n <= max_n)
            .collect::<Vec<_>>()
    };

    let mut hashers = vec![
        // Mock throughput hashset
        (
            |alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
                if !implementation_selected("mock") {
                    return None;
                }
                Some(Box::new(MockHashSet::new(1.0 / alpha, keys)))
            } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
            vec![1.0],
        ),
        // k-PHF-set
        #[cfg(feature = "kphf")]
        (
            |alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
                if !implementation_selected("kphf") {
                    return None;
                }
                Some(Box::new(KphfSet::<
                    KptrHash<{ kphf::Mode::SortBump as u8 }, BIN_SIZE>,
                    BIN_SIZE,
                >::try_new(
                    alpha,
                    1.5 * space_lower_bound(BIN_SIZE, alpha),
                    keys,
                )?))
            } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
            vec![0.95],
        ),
        // PtrHash
        #[cfg(feature = "bench-ptrhash")]
        (
            |_alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
                if !implementation_selected("ptrhash") {
                    return None;
                }
                Some(Box::new(PhfSet::<phf_trait::PtrHash>::new(0.0, 0.0, keys)) as Box<dyn HashSet>)
            } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
            vec![0.99],
        ),
        // non-minimal PHast
        #[cfg(feature = "bench")]
        (
            |_alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
                if !implementation_selected("phast") {
                    return None;
                }
                Some(Box::new(PhfSet::<phf_trait::PHast>::new(0.0, 0.0, keys)) as Box<dyn HashSet>)
            } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
            vec![0.98],
        ),
        // SwissTable
        #[cfg(feature = "bench-hashbrown")]
        (
            |alpha: f32, keys: &[T]| -> Option<Box<dyn HashSet>> {
                if !implementation_selected("hashbrown") && !implementation_selected("swisstable") {
                    return None;
                }
                let set =
                    hashbrown::HashSet::<T, gxhash::GxBuildHasher>::from_iter(keys.iter().copied());
                Some(Box::new(set))
            } as fn(f32, &[T]) -> Option<Box<dyn HashSet>>,
            // This existing constructor uses Hashbrown's default capacity policy;
            // its actual occupancy is measured from len/capacity.
            vec![0.9],
        ),
        // // U64HashSet
        // (
        //     |alpha: f32, keys: &[T]| Some(Box::new(U64HashSet::new(1. / alpha, keys))),
        //     // vec![0.7, 0.8, 0.9, 0.95],
        //     vec![0.7],
        // ),
        // Static braided cuckoo (power-of-two layers, at least 10% slack).
        (
            |_alpha: f32, keys: &[T]| {
                if !implementation_selected("braided-eager") {
                    return None;
                }
                let set = BraidedCuckooSet::<{ Mode::Eager as u8 }>::try_new(keys);
                if set.is_none() {
                    eprintln!(
                        "BraidedCuckooSet<Eager>: placement failed for n={}",
                        keys.len()
                    );
                }
                Some(Box::new(set?))
            },
            vec![0.9],
        ),
        (
            |_alpha: f32, keys: &[T]| {
                if !implementation_selected("braided-lazy") {
                    return None;
                }
                let set = BraidedCuckooSet::<{ Mode::Lazy as u8 }>::try_new(keys);
                if set.is_none() {
                    eprintln!(
                        "BraidedCuckooSet<Lazy>: placement failed for n={}",
                        keys.len()
                    );
                }
                Some(Box::new(set?))
            },
            vec![0.9],
        ),
        // Eager Cuckoo
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("cuckoo-eager") {
                    return None;
                }
                Some(Box::new(CuckooSet::<{ Mode::Eager as u8 }>::new(
                    1. / alpha,
                    keys,
                )))
            },
            vec![0.7, 0.9],
        ),
        // Lazy Cuckoo
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("cuckoo-lazy") {
                    return None;
                }
                Some(Box::new(CuckooSet::<{ Mode::Lazy as u8 }>::new(
                    1. / alpha,
                    keys,
                )))
            },
            vec![0.9],
        ),
        // Lazy Cuckoo with a 16-bit spill summary.
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("growable-right-cuckoo") {
                    return None;
                }
                Some(Box::new(GrowableRightCuckooSet::new(1. / alpha, keys)))
            },
            vec![0.7, 0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("filtered-cuckoo-lazy") {
                    return None;
                }
                Some(Box::new(FilteredCuckooSet::<{ Mode::Lazy as u8 }>::new(
                    1. / alpha,
                    keys,
                )))
            },
            vec![0.7, 0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("inline-bloom-cuckoo-lazy") {
                    return None;
                }
                Some(Box::new(
                    InlineFilterCuckooSet::<{ Mode::Lazy as u8 }, false>::new(1. / alpha, keys),
                ))
            },
            vec![0.7, 0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("inline-onehot-cuckoo-lazy") {
                    return None;
                }
                Some(Box::new(
                    InlineFilterCuckooSet::<{ Mode::Lazy as u8 }, true>::new(1. / alpha, keys),
                ))
            },
            vec![0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("spill-before-miss-u8-lazy") {
                    return None;
                }
                Some(Box::new(
                    SpillCuckooSet::<{ Mode::Lazy as u8 }, true, 8>::new(1. / alpha, keys),
                ))
            },
            vec![0.7, 0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("spill-after-miss-u8-lazy") {
                    return None;
                }
                Some(Box::new(
                    SpillCuckooSet::<{ Mode::Lazy as u8 }, false, 8>::new(1. / alpha, keys),
                ))
            },
            vec![0.7, 0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("spill-before-miss-u16-lazy") {
                    return None;
                }
                Some(Box::new(
                    SpillCuckooSet::<{ Mode::Lazy as u8 }, true, 16>::new(1. / alpha, keys),
                ))
            },
            vec![0.7, 0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("spill-after-miss-u16-lazy") {
                    return None;
                }
                Some(Box::new(
                    SpillCuckooSet::<{ Mode::Lazy as u8 }, false, 16>::new(1. / alpha, keys),
                ))
            },
            vec![0.7, 0.9],
        ),
        // Promising prefix filters: compare SIMD widths at the same occupancies.
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-eager-8") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Eager as u8 }, 8>::new(1. / alpha, keys),
                ))
            },
            vec![0.99],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-lazy-8") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Lazy as u8 }, 8>::new(1. / alpha, keys),
                ))
            },
            vec![0.7],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-eager-16") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Eager as u8 }, 16>::new(1. / alpha, keys),
                ))
            },
            vec![0.99],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-lazy-16") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Lazy as u8 }, 16>::new(1. / alpha, keys),
                ))
            },
            vec![0.7],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-lazy-32-onehot") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Lazy as u8 }, 32, false, true>::new(
                        1. / alpha,
                        keys,
                    ),
                ))
            },
            vec![0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-eager-32") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Eager as u8 }, 32>::new(1. / alpha, keys),
                ))
            },
            vec![0.99],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-lazy-32") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Lazy as u8 }, 32>::new(1. / alpha, keys),
                ))
            },
            vec![0.9],
        ),
        // Compare the same prefix widths with Elias–Fano compression.
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-eager-8-ef") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Eager as u8 }, 8, true>::new(
                        1. / alpha,
                        keys,
                    ),
                ))
            },
            vec![0.99],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-lazy-8-ef") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Lazy as u8 }, 8, true>::new(
                        1. / alpha,
                        keys,
                    ),
                ))
            },
            vec![0.7],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-eager-16-ef") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Eager as u8 }, 16, true>::new(
                        1. / alpha,
                        keys,
                    ),
                ))
            },
            vec![0.99],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-lazy-16-ef") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Lazy as u8 }, 16, true>::new(
                        1. / alpha,
                        keys,
                    ),
                ))
            },
            vec![0.7],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-lazy-32-ef-onehot") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Lazy as u8 }, 32, true, true>::new(
                        1. / alpha,
                        keys,
                    ),
                ))
            },
            vec![0.9],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-eager-32-ef") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Eager as u8 }, 32, true>::new(
                        1. / alpha,
                        keys,
                    ),
                ))
            },
            vec![0.99],
        ),
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("quotient-lazy-32-ef") {
                    return None;
                }
                Some(Box::new(
                    CuckooQuotientSet::<{ Mode::Lazy as u8 }, 32, true>::new(
                        1. / alpha,
                        keys,
                    ),
                ))
            },
            vec![0.9],
        ),
        // FPH
        #[cfg(feature = "extc")]
        (
            |alpha: f32, keys: &[T]| {
                if !implementation_selected("fph") {
                    return None;
                }
                Some(Box::new(FphDynSet::new(alpha, keys)?) as Box<dyn HashSet>)
            },
            vec![0.95],
        ),
        // TPHT
        #[cfg(feature = "tpht")]
        (
            |_alpha: f32, keys: &[T]| {
                if !implementation_selected("tpht") {
                    return None;
                }
                Some(Box::new(tpht::Tpht::new(keys)?) as Box<dyn HashSet>)
            },
            vec![0.9],
        ),
        // TinyPtr non-concurrent Blast hash table
        #[cfg(feature = "tpht")]
        (
            |_alpha: f32, keys: &[T]| {
                // TinyPtr currently panics during construction on quick-run inputs.
                if quick_bench() || !implementation_selected("tinyptr") {
                    return None;
                }
                Some(Box::new(tinyptr::TinyPtr::new(keys)?) as Box<dyn HashSet>)
            },
            vec![0.85],
        ),
        // // engineering k-PHF:
        // // - Only use the Hash-displace variant with faster queries.
        // // - Skip threshold-based-bumping variants
        // #[cfg(feature = "ekphf")]
        // (
        //     |_alpha: f32, keys: &[T]| {
        //         Some(
        //             Box::new(KphfSet::<Hd8Set, BIN_SIZE>::try_new(0.0, 0.0, keys)?)
        //                 as Box<dyn HashSet>,
        //         )
        //     },
        //     vec![1.0],
        // ),
        // // MapEmbed
        // (
        //     |_alpha: f32, keys: &[T]| {
        //         Some(Box::new(mapembed::MapEmbed::new(keys)?) as Box<dyn HashSet>)
        //     },
        //     vec![0.9],
        // ),
    ];
    if let Some(alphas) = args
        .windows(2)
        .find(|pair| pair[0] == "--alphas")
        .map(|pair| {
            pair[1]
                .split(',')
                .map(|alpha| {
                    alpha
                        .parse::<f32>()
                        .expect("--alphas values must be numbers")
                })
                .collect::<Vec<_>>()
        })
    {
        assert!(
            !alphas.is_empty() && alphas.iter().all(|alpha| *alpha > 0.0 && *alpha < 1.0),
            "--alphas values must be in (0, 1)"
        );
        for (_, configured_alphas) in &mut hashers {
            *configured_alphas = alphas.clone();
        }
    }
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
    query_count: usize,
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
    primary_hit_ratio: Option<f32>,
    max_primary_bucket_size: Option<usize>,
    negative_filter_pass_ratio: Option<f32>,
    bumped_frac: f32,
    overhead: f32,
    /// Allocated bytes divided by the input's n * sizeof(u64) bytes.
    memory_ratio: f32,
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
        let (chunks, tail) = keys.as_chunks_mut::<{ 1 << 20 }>();
        chunks.par_iter_mut().for_each(|chunk| {
            rand::fill(chunk);
            for key in chunk {
                *key = (*key).max(1);
            }
        });
        rand::fill(tail);
        for key in tail {
            *key = (*key).max(1);
        }
        let args: Vec<_> = std::env::args().collect();
        let query_count = args
            .windows(2)
            .find(|pair| pair[0] == "--queries-per-thread")
            .map(|pair| {
                let count: usize = pair[1].parse().expect("--queries-per-thread requires an integer");
                assert!(count > 0, "--queries-per-thread must be positive");
                count
            })
            .unwrap_or_else(|| if quick_bench() { 1_000_000 } else { QUERIES });
        let thread_count = selected_threads().into_iter().max().unwrap_or(1);
        let mut queries = vec![
            vec![
                [
                    vec![0; query_count],
                    vec![0; query_count],
                    vec![0; query_count],
                    vec![0; query_count],
                    vec![0; query_count],
                ];
                REPEATS
            ];
            thread_count
        ];

        queries.par_iter_mut().for_each(|threads_queries| {
            for repeat_queries in threads_queries {
                let rng = &mut rand::rng();
                for (q, p) in std::iter::zip(&mut repeat_queries.iter_mut(), PERCENTILES) {
                    for x in q.iter_mut() {
                        if rand::random_bool(p) {
                            *x = *keys.choose(rng).unwrap();
                        } else {
                            *x = rand::random::<T>().max(1);
                        }
                    }
                }
            }
        });
        Self {
            n,
            keys,
            queries,
            query_count,
        }
    }

    fn negative_filter_pass_ratio(&self, set: &dyn HashSet) -> Option<f32> {
        const SAMPLES: usize = 200_000;
        set.negative_passes_filter(1)?;
        let mut passed = 0;
        let mut sampled = 0;
        let mut queries = [0; 1024];
        while sampled < SAMPLES {
            rand::fill(&mut queries);
            for key in queries {
                if key == 0 || set.contains(key) {
                    continue;
                }
                passed += usize::from(set.negative_passes_filter(key)?);
                sampled += 1;
                if sampled == SAMPLES {
                    break;
                }
            }
        }
        Some(passed as f32 / sampled as f32)
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
        let memory_ratio = overhead;
        let load_factor = h.load_factor();
        let primary_hit_ratio = h.primary_hit_ratio();
        let max_primary_bucket_size = h.max_primary_bucket_size();
        eprintln!(
            "Benchmarking {name}: n={}, alpha={load_factor:.3}, memory_ratio={memory_ratio:.3}",
            self.n,
        );
        let pf = h.has_prefetch();
        let batch_enabled = batch_bench() && h.has_batch();
        let negative_filter_pass_ratio = self.negative_filter_pass_ratio(h.as_ref());
        for threads in selected_threads() {
            for metric in MODES {
                if matches!(metric, "batch8" | "batch16" | "batch32") && !batch_enabled {
                    continue;
                }
                for query_repeat in 0..if quick_bench() { 1 } else { QUERY_REPEATS } {
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
                            "batch8" => {
                                let c = h.count_batch(qs, 8);
                                black_box(c);
                            }
                            "batch16" => {
                                let c = h.count_batch(qs, 16);
                                black_box(c);
                            }
                            "batch32" => {
                                let c = h.count_batch(qs, 32);
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
                        query[qi] =
                            start.elapsed().as_nanos() as f32 / (threads * self.query_count) as f32;
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
                        primary_hit_ratio,
                        max_primary_bucket_size,
                        negative_filter_pass_ratio,
                        bumped_frac,
                        overhead,
                        memory_ratio,
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

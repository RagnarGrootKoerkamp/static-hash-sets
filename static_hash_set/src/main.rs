#![allow(incomplete_features)]
#![feature(
    impl_trait_in_assoc_type,
    widening_mul,
    explicit_tail_calls,
    adt_const_params,
    generic_const_exprs
)]

pub mod cuckoo;
pub mod kphf_set;
#[cfg(test)]
mod test;
mod traits;
mod u64_hashset;

use std::{hash::BuildHasherDefault, hint::black_box};

use cuckoo::{CuckooSet, Mode};
use kphf::space_lower_bound;
use kphf_set::KphfSet;
use rand::seq::IndexedRandom;
use traits::HashSet;
use u64_hashset::U64HashSet;
type FxHasher = BuildHasherDefault<fxhash::FxHasher>;

// type T = u32;
// const BUCKET_SIZE: usize = 16;
// type S = wide::i32x8;
type T = u64;
const BIN_SIZE: usize = 8;
type S = wide::i64x4;

fn main() {
    let ns = (0..)
        .map(|i| (1_000_000. * 2.5f32.powi(i)) as usize)
        .take_while(|x| *x <= 40_000_000)
        .collect::<Vec<_>>();

    let hashers = vec![
        Box::new(hashbrown::HashSet::<T, FxHasher>::default()) as Box<dyn HashSet>,
        Box::new(U64HashSet::new(1.4, &[])),
        Box::new(U64HashSet::new(1.2, &[])),
        Box::new(U64HashSet::new(1.1, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchBoth }>::new(1.4, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchOneLazy }>::new(1.4, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchOneEager }>::new(1.4, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchBoth }>::new(1.2, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchOneLazy }>::new(1.2, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchOneEager }>::new(1.2, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchBoth }>::new(1.1, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchOneLazy }>::new(1.1, &[])),
        Box::new(CuckooSet::<{ Mode::PrefetchOneEager }>::new(1.1, &[])),
        Box::new(KphfSet::<{ kphf::Mode::Sort }, BIN_SIZE>::new(
            0.7,
            2.0 * space_lower_bound(BIN_SIZE, 0.7),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(KphfSet::<{ kphf::Mode::SortBump }, BIN_SIZE>::new(
            0.7,
            2.0 * space_lower_bound(BIN_SIZE, 0.7),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(KphfSet::<{ kphf::Mode::Sort }, BIN_SIZE>::new(
            0.8,
            2.0 * space_lower_bound(BIN_SIZE, 0.8),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(KphfSet::<{ kphf::Mode::SortBump }, BIN_SIZE>::new(
            0.8,
            2.0 * space_lower_bound(BIN_SIZE, 0.8),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(KphfSet::<{ kphf::Mode::Sort }, BIN_SIZE>::new(
            0.9,
            2.0 * space_lower_bound(BIN_SIZE, 0.9),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(KphfSet::<{ kphf::Mode::SortBump }, BIN_SIZE>::new(
            0.9,
            2.0 * space_lower_bound(BIN_SIZE, 0.9),
            &[],
        )) as Box<dyn HashSet>,
    ];
    bench(&ns, &hashers);

    // TODO: Slots overhead and metadata overhead separately?
    // TODO: Multithreaded benchmarks.
    // TODO: Benchmark on server.
}

fn time<T>(mut f: impl FnMut() -> T) -> (f32, T) {
    let start = std::time::Instant::now();
    let out = f();
    let duration = start.elapsed();
    (duration.as_secs_f32() * 1e9, out)
}

const QUERIES: usize = 2_000_000;
const REPEATS: usize = 3;
const THREADS: [usize; 3] = [1, 6, 12];

pub struct Bencher {
    n: usize,
    keys: Vec<T>,
    queries: Vec<Vec<[Vec<T>; 5]>>,
}

pub struct BenchResult {
    h: String,
    n: usize,
    pf: bool,
    build: f32,
    overhead: f32,
    threads: usize,
    metric: &'static str,
    repeat: usize,
    q01: f32,
    q10: f32,
    q50: f32,
    q90: f32,
    q99: f32,
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
            *THREADS.last().unwrap()
        ];
        for threads_queries in &mut queries {
            for repeat_queries in threads_queries {
                let p = [0.01, 0.1, 0.5, 0.9, 0.99];
                let rng = &mut rand::rng();
                for (q, p) in std::iter::zip(&mut repeat_queries.iter_mut(), p) {
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

    pub fn bench(&self, h: &dyn HashSet) -> Vec<BenchResult> {
        const PS: [f64; 5] = [0.01, 0.1, 0.5, 0.9, 0.99];

        let name = h.name();
        eprint!("{:<30} {:>11} | ", name, self.n);
        let (build, h) = time(|| h.new(&self.keys));
        let build = build / self.n as f32;
        let bits_per_key = h.allocation_size() as f32 * 8.0 / self.n as f32;
        let overhead = bits_per_key / T::BITS as f32;
        let pf = h.has_prefetch();
        eprint!("{:>8.3} {:>8.3} |", build, overhead);

        let mut results = vec![];

        eprintln!();
        for &threads in &THREADS {
            for metric in ["latency", "loop", "prefetch"] {
                for repeat in 0..REPEATS {
                    eprint!("{:>64} {threads:>8} {metric:>10}", "");
                    let mut query = [0f32; 5];
                    for (qi, &_p) in PS.iter().enumerate() {
                        let start = std::time::Instant::now();
                        std::thread::scope(|scope| {
                            for t in 0..threads {
                                let qs = &self.queries[t][repeat][qi];
                                let h = &h;
                                scope.spawn(move || match metric {
                                    "latency" => {
                                        let c = h.count_latency(&qs);
                                        black_box(c);
                                    }
                                    "loop" => {
                                        let c = h.count_loop(&qs);
                                        black_box(c);
                                    }
                                    "prefetch" => {
                                        let c = h.count_prefetch(&qs);
                                        black_box(c);
                                    }
                                    _ => unreachable!(),
                                });
                            }
                        });
                        query[qi] = start.elapsed().as_nanos() as f32 / (threads * QUERIES) as f32;
                        eprint!(" {:>8.3}", query[qi]);
                    }
                    eprintln!();
                    results.push(BenchResult {
                        h: name.to_string(),
                        n: self.n,
                        pf,
                        build,
                        overhead,
                        threads,
                        metric,
                        repeat,
                        q01: query[0],
                        q10: query[1],
                        q50: query[2],
                        q90: query[3],
                        q99: query[4],
                    });
                }
            }
        }
        results
    }
}

pub fn bench(ns: &[usize], hs: &[Box<dyn HashSet>]) {
    eprintln!(
        "{:<30} {:>11} | {:>8} {:>8} {:>10} | {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} | ",
        "Type", "n", "build", "overhead", "threads", "metric", "1%", "10%", "50%", "90%", "99%",
    );
    println!("h,n,pf,build,overhead,threads,metric,repeat,q01,q10,q50,q90,q99");
    for &n in ns {
        let bencher = Bencher::new(n);
        for h in hs {
            for r in bencher.bench(&**h) {
                println!(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                    r.h,
                    r.n,
                    r.pf,
                    r.build,
                    r.overhead,
                    r.threads,
                    r.metric,
                    r.repeat,
                    r.q01,
                    r.q10,
                    r.q50,
                    r.q90,
                    r.q99
                );
            }
        }
        eprintln!();
    }
}

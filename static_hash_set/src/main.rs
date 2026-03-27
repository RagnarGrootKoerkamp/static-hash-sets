#![allow(unused)]
#![feature(impl_trait_in_assoc_type, widening_mul)]

mod static_hashset;
mod traits;
mod u64_hashset;

use std::{
    hash::{BuildHasher, BuildHasherDefault},
    hint::black_box,
};

use fxhash::{FxBuildHasher, FxHashSet};
use mem_dbg::{MemSize, SizeFlags};
use rand::{
    seq::{IndexedRandom, SliceRandom},
    Rng,
};
use static_hashset::StaticHashSet;
use sux::{dict::EliasFanoBuilder, traits::IndexedDict};
use traits::HashSet;
use u64_hashset::U64HashSet;
type FxHasher = BuildHasherDefault<fxhash::FxHasher>;

type T = u32;

fn gen_keys(n: usize) -> Vec<T> {
    eprint!("Gen {n} keys..");
    let mut v = vec![0; n];
    rand::fill(&mut v[..]);
    eprintln!(" done");
    v
}

fn main() {
    let ns = (0..)
        .map(|i| (1_000_000. * 1.35f32.powi(i)) as usize)
        .take_while(|x| *x <= 100_000_000)
        .collect::<Vec<_>>();

    let hashers = vec![
        Box::new(hashbrown::HashSet::<u32, FxHasher>::default()) as Box<dyn HashSet>,
        // some slow external stuff
        //Box::new(
        //    fastbloom::BloomFilter::with_false_pos(0.1)
        //        .hasher(FxBuildHasher::default())
        //        .items(&[()]),
        //),
        //Box::new(cuckoofilter::CuckooFilter::<fxhash::FxHasher>::with_capacity(0)),
        Box::new(U64HashSet::new(1.4, &[])),
        Box::new(U64HashSet::new(1.3, &[])),
        Box::new(U64HashSet::new(1.2, &[])),
        Box::new(U64HashSet::new(1.1, &[])),
        Box::new(StaticHashSet::<false>::new(1.4, 0.001, &[])),
        Box::new(StaticHashSet::<false>::new(1.2, 0.003, &[])),
        Box::new(StaticHashSet::<false>::new(1.1, 0.006, &[])),
    ];
    bench(&ns, &hashers);

    // let n = 118_000_000; // 2.844x overhead
    // let n = 117_000_000; // 1.434x overhead
    // for n in [100_000_000] {
    //     // absl(n);
    //     u64_hashset(n);
    //     static_hashset(n);
    //     // ef(n);
    // }

    // eprintln!("Slots overhead:    {}", (b * k) as f32 / n as f32 - 1.0);
    // eprintln!("Metadata overhead: {}", p_bits as f32 / (64 * n) as f32);
    // for s in 0..=8 {
    //     eprintln!("S = {s}");
    //     test(n, p_bits.div_ceil(s.max(1)), k, b, s);
    // }
}

fn time<T>(mut f: impl FnMut() -> T) -> (f32, T) {
    let start = std::time::Instant::now();
    let out = f();
    let duration = start.elapsed();
    (duration.as_secs_f32() * 1e9, out)
}

const QUERIES: usize = 5_000_000;

pub struct Bencher {
    n: usize,
    keys: Vec<u32>,
    queries: [Vec<u32>; 5],
}

#[derive(serde::Serialize)]
pub struct BenchResult {
    h: String,
    n: usize,
    pf: bool,
    build: f32,
    overhead: f32,
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
        let mut queries = [
            vec![0; QUERIES],
            vec![0; QUERIES],
            vec![0; QUERIES],
            vec![0; QUERIES],
            vec![0; QUERIES],
        ];
        let p = [0.01, 0.1, 0.5, 0.9, 0.99];
        let rng = &mut rand::rng();
        for (q, p) in std::iter::zip(&mut queries.iter_mut(), p) {
            for x in q.iter_mut() {
                if rand::random_bool(p) {
                    *x = *keys.choose(rng).unwrap();
                } else {
                    *x = rand::random();
                }
            }
        }
        Self { n, keys, queries }
    }

    pub fn bench(&self, h: &dyn HashSet) -> BenchResult {
        let name = h.name();
        eprint!("{:<30} {:>11} | ", name, self.n);
        let (build, mut h) = time(|| h.new(&self.keys));
        let build = build / self.n as f32;
        let bits_per_key = h.allocation_size() as f32 * 8.0 / self.n as f32;
        let overhead = bits_per_key / 32.0;
        eprint!("{:>8.3} {:>8.3} | ", build, overhead);
        let mut query = [0f32; 5];
        for i in 0..5 {
            query[i] = time(|| h.count(&self.queries[i])).0 / QUERIES as f32;
            eprint!("{:>8.3} ", query[i]);
        }
        eprintln!();
        BenchResult {
            h: name.to_string(),
            n: self.n,
            pf: h.has_prefetch(),
            build,
            overhead,
            q01: query[0],
            q10: query[1],
            q50: query[2],
            q90: query[3],
            q99: query[4],
        }
    }
}

pub fn bench(ns: &[usize], hs: &[Box<dyn HashSet>]) {
    eprintln!(
        "{:<30} {:>11} | {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8} ",
        "Type", "n", "build", "overhead", "p=0.01", "p=0.10", "p=0.5", "p=0.90", "p=0.99"
    );
    let mut results = vec![];
    for &n in ns {
        let bencher = Bencher::new(n);
        for h in hs {
            results.push(bencher.bench(&**h));
        }
        eprintln!();
    }
    serde_json::to_writer_pretty(std::io::stdout(), &results).unwrap();
}

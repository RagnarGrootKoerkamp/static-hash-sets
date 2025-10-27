use fxhash::{FxBuildHasher, FxHashSet, FxHasher};
use hashbrown::DefaultHashBuilder;
use rand::seq::IndexedRandom;

use crate::{hashset::U64HashSet, static_hashset::StaticHashSet};

pub trait HashSet {
    fn name(&self) -> &'static str;
    fn new(&self, keys: &[u32]) -> Box<dyn HashSet>;
    fn has_prefetch(&self) -> bool {
        false
    }
    fn prefetch(&mut self, key: u32) {}
    fn get(&mut self, key: u32) -> bool;
    fn count(&mut self, keys: &[u32]) -> usize {
        let lookahead = 32;
        let mut c = 0;
        for i in 0..keys.len().saturating_sub(lookahead) {
            self.prefetch(keys[i + lookahead]);
            c += self.get(keys[i]) as usize;
        }
        std::hint::black_box(c);
        c
    }
}

impl HashSet for FxHashSet<u32> {
    fn name(&self) -> &'static str {
        "FxHashSet"
    }
    fn new(&self, keys: &[u32]) -> Box<dyn HashSet> {
        let mut h = FxHashSet::with_capacity_and_hasher(keys.len(), Default::default());
        for &k in keys {
            h.insert(k);
        }
        Box::new(h)
    }
    #[inline(always)]
    fn get(&mut self, key: u32) -> bool {
        self.contains(&key)
    }
}

impl HashSet for fastbloom::BloomFilter<FxBuildHasher> {
    fn name(&self) -> &'static str {
        "BloomFilter"
    }
    fn new(&self, keys: &[u32]) -> Box<dyn HashSet> {
        let mut h = fastbloom::BloomFilter::with_false_pos(0.01)
            .hasher(FxBuildHasher::default())
            .items(keys.iter().copied());
        // eprintln!("\nBits/elem:  {}", h.num_bits() as f32 / keys.len() as f32);
        // eprintln!("Num hashes: {}", h.num_hashes());
        Box::new(h)
    }
    #[inline(always)]
    fn get(&mut self, key: u32) -> bool {
        self.contains(&key)
    }
}

impl HashSet for cuckoofilter::CuckooFilter<FxHasher> {
    fn name(&self) -> &'static str {
        "CuckooFilter"
    }
    fn new(&self, keys: &[u32]) -> Box<dyn HashSet> {
        let mut h = Self::with_capacity(keys.len() + keys.len() / 10);
        for &k in keys {
            h.add(&k).unwrap();
        }
        Box::new(h)
    }
    #[inline(always)]
    fn get(&mut self, key: u32) -> bool {
        self.contains(&key)
    }
}

impl HashSet for U64HashSet {
    fn name(&self) -> &'static str {
        "U64HashSet"
    }
    fn new(&self, keys: &[u32]) -> Box<dyn HashSet> {
        let mut h = U64HashSet::new(self.slot_ratio, keys);
        Box::new(h)
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    #[inline(always)]
    fn prefetch(&mut self, key: u32) {
        U64HashSet::prefetch(self, key)
    }
    #[inline(always)]
    fn get(&mut self, key: u32) -> bool {
        self.contains(key)
    }
}

impl<const PROBE: bool> HashSet for StaticHashSet<PROBE> {
    fn name(&self) -> &'static str {
        "StaticHashSet"
    }
    fn new(&self, keys: &[u32]) -> Box<dyn HashSet> {
        let h = StaticHashSet::<PROBE>::new(self.slot_ratio, self.meta_ratio, keys);
        Box::new(h)
    }
    fn has_prefetch(&self) -> bool {
        true
    }
    #[inline(always)]
    fn prefetch(&mut self, key: u32) {
        StaticHashSet::prefetch(self, key)
    }
    #[inline(always)]
    fn get(&mut self, key: u32) -> bool {
        self.contains(key)
    }
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
    queries: [Vec<u32>; 3],
}

#[derive(serde::Serialize)]
pub struct BenchResult {
    h: String,
    n: usize,
    pf: bool,
    build: f32,
    q01: f32,
    q50: f32,
    q99: f32,
}

impl Bencher {
    pub fn new(n: usize) -> Self {
        let mut keys = vec![0; n];
        rand::fill(&mut keys[..]);
        let mut queries = [vec![0; QUERIES], vec![0; QUERIES], vec![0; QUERIES]];
        let p = [0.01, 0.5, 0.99];
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
        eprint!("{:>8.3} | ", build);
        let mut query = [0f32; 3];
        for i in 0..3 {
            query[i] = time(|| h.count(&self.queries[i])).0 / QUERIES as f32;
            eprint!("{:>8.3} ", query[i]);
        }
        eprintln!();
        BenchResult {
            h: name.to_string(),
            n: self.n,
            pf: h.has_prefetch(),
            build,
            q01: query[0],
            q50: query[1],
            q99: query[2],
        }
    }
}

pub fn bench(ns: &[usize], hs: &[Box<dyn HashSet>]) {
    eprintln!(
        "{:<30} {:>11} | {:>8} | {:>8} {:>8} {:>8} ",
        "Type", "n", "build", "p=0.01", "p=0.5", "p=0.99"
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

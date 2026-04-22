//! Wrappers around engineering-k-perfect-hashing library (ThresholdBasedBumping and HashDisplace).
//!
//! These are k-perfect hash functions (k-PHFs): each key maps to a bucket that holds
//! at most k=8 keys. To use as a set, actual keys are stored in bins indexed by the PHF.
use crate::traits::HashSet;
use crate::u64_hashset::Bin;
use crate::BIN_SIZE;
use crate::S;
use crate::T;
use engineering_kphf::{Hd8Set, Tbb84pSet, Tbb85Set};

fn build_table<F: Fn(T) -> u64>(keys: &[T], num_buckets: usize, query: F) -> Box<[Bin]> {
    let mut table = vec![Bin([0u64; BIN_SIZE]); num_buckets].into_boxed_slice();
    for &key in keys {
        if key == 0 {
            continue;
        }
        let bucket = query(key) as usize;
        let bin = &mut table[bucket];
        let slot = bin.0.iter().position(|&x| x == 0).expect("PHF bucket overflow");
        bin.0[slot] = key;
    }
    table
}

pub struct EkphfTbb85 {
    overload: f64,
    phf: Tbb85Set,
    table: Box<[Bin]>,
    len: usize,
    has_zero: bool,
}

impl EkphfTbb85 {
    pub fn new(keys: &[T], overload: f64) -> Self {
        let non_zero: Vec<T> = keys.iter().copied().filter(|&k| k != 0).collect();
        let phf = Tbb85Set::new(&non_zero, overload);
        // num buckets = ceil(n / k)
        let num_buckets = non_zero.len().div_ceil(8).max(1);
        let table = build_table(&non_zero, num_buckets, |k| phf.query(k));
        Self {
            overload,
            phf,
            table,
            len: keys.len(),
            has_zero: keys.contains(&0),
        }
    }
}

impl HashSet for EkphfTbb85 {
    fn name(&self) -> &'static str {
        "EkphfTbb85"
    }

    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(EkphfTbb85::new(keys, self.overload))
    }

    fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table) + self.phf.count_bits() / 8
    }

    fn load_factor(&self) -> f32 {
        self.len as f32 / (self.table.len() as f32 * BIN_SIZE as f32)
    }

    fn kphf_size(&self) -> usize {
        self.phf.count_bits() / 8
    }

    fn contains(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let bucket = self.phf.query(key) as usize;
        let Some(bin) = self.table.get(bucket) else { return false; };
        bin.contains(S::splat(key as _))
    }
}

pub struct EkphfTbb84p {
    overload: f64,
    phf: Tbb84pSet,
    table: Box<[Bin]>,
    len: usize,
    has_zero: bool,
}

impl EkphfTbb84p {
    pub fn new(keys: &[T], overload: f64) -> Self {
        let non_zero: Vec<T> = keys.iter().copied().filter(|&k| k != 0).collect();
        let phf = Tbb84pSet::new(&non_zero, overload);
        let num_buckets = non_zero.len().div_ceil(8).max(1);
        let table = build_table(&non_zero, num_buckets, |k| phf.query(k));
        Self {
            overload,
            phf,
            table,
            len: keys.len(),
            has_zero: keys.contains(&0),
        }
    }
}

impl HashSet for EkphfTbb84p {
    fn name(&self) -> &'static str {
        "EkphfTbb84p"
    }

    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(EkphfTbb84p::new(keys, self.overload))
    }

    fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table) + self.phf.count_bits() / 8
    }

    fn load_factor(&self) -> f32 {
        self.len as f32 / (self.table.len() as f32 * BIN_SIZE as f32)
    }

    fn kphf_size(&self) -> usize {
        self.phf.count_bits() / 8
    }

    fn contains(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let bucket = self.phf.query(key) as usize;
        let Some(bin) = self.table.get(bucket) else { return false; };
        bin.contains(S::splat(key as _))
    }
}

pub struct EkphfHd8 {
    bucket_size: u64,
    phf: Hd8Set,
    table: Box<[Bin]>,
    len: usize,
    has_zero: bool,
}

impl EkphfHd8 {
    pub fn new(keys: &[T], bucket_size: u64) -> Self {
        let non_zero: Vec<T> = keys.iter().copied().filter(|&k| k != 0).collect();
        let phf = Hd8Set::new(&non_zero, bucket_size);
        // nbins = ceil(n / k) with k=8 (template param); this is the PHF output range
        let num_buckets = non_zero.len().div_ceil(8).max(1);
        let table = build_table(&non_zero, num_buckets, |k| phf.query(k));
        Self {
            bucket_size,
            phf,
            table,
            len: keys.len(),
            has_zero: keys.contains(&0),
        }
    }
}

impl HashSet for EkphfHd8 {
    fn name(&self) -> &'static str {
        "EkphfHd8"
    }

    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(EkphfHd8::new(keys, self.bucket_size))
    }

    fn allocation_size(&self) -> usize {
        std::mem::size_of_val(&*self.table) + self.phf.count_bits() / 8
    }

    fn load_factor(&self) -> f32 {
        self.len as f32 / (self.table.len() as f32 * self.bucket_size as f32)
    }

    fn kphf_size(&self) -> usize {
        self.phf.count_bits() / 8
    }

    fn contains(&self, key: T) -> bool {
        if key == 0 {
            return self.has_zero;
        }
        let bucket = self.phf.query(key) as usize;
        let bin = unsafe { self.table.get_unchecked(bucket) };
        bin.contains(S::splat(key as _))
    }
}

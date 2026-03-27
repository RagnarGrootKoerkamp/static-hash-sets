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
        Box::new(FxHashSet::default()) as Box<dyn HashSet>,
        Box::new(U64HashSet::new(1.4, &[])),
        Box::new(U64HashSet::new(1.3, &[])),
        Box::new(U64HashSet::new(1.2, &[])),
        Box::new(
            fastbloom::BloomFilter::with_false_pos(0.1)
                .hasher(FxBuildHasher::default())
                .items(&[()]),
        ),
        Box::new(cuckoofilter::CuckooFilter::<fxhash::FxHasher>::with_capacity(0)),
        // Box::new(StaticHashSet::<true>::new(1.4, 0.002, &[])),
        // Box::new(StaticHashSet::<false>::new(1.4, 0.002, &[])),
        // Box::new(StaticHashSet::<true>::new(1.4, 0.001, &[])),
        Box::new(StaticHashSet::<false>::new(1.4, 0.001, &[])),
        // Box::new(StaticHashSet::<true>::new(1.4, 0.0005, &[])),
        // Box::new(StaticHashSet::<false>::new(1.4, 0.0005, &[])),
        // Box::new(StaticHashSet::<true>::new(1.3, 0.002, &[])),
        // Box::new(StaticHashSet::<false>::new(1.3, 0.002, &[])),
        // Box::new(StaticHashSet::<true>::new(1.3, 0.001, &[])),
        // Box::new(StaticHashSet::<false>::new(1.3, 0.001, &[])),
        // Box::new(StaticHashSet::<true>::new(1.3, 0.0005, &[])),
        // Box::new(StaticHashSet::<false>::new(1.3, 0.0005, &[])),
    ];
    traits::bench(&ns, &hashers);

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

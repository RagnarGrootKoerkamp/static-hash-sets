//! Shared end-to-end construction/lookup fuzz driver for cuckoo variants.
#![allow(dead_code)]
#[path = "../bin.rs"]
mod bin;
#[path = "../cuckoo.rs"]
mod cuckoo;
#[path = "../cuckoo_inline_filters.rs"]
mod cuckoo_inline_filters;
#[path = "../cuckoo_quotienting.rs"]
mod cuckoo_quotienting;
#[path = "../traits.rs"]
mod traits;
use bin::Bin;
use cuckoo_inline_filters::InlineFilterCuckooSet as Inline;
use cuckoo_quotienting::CuckooSet as Quotient;
use traits::HashSet;
type T = u64;
const BIN_SIZE: usize = 8;
type S = wide::i64x4;
fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .unwrap_or("10000".into())
        .parse()
        .unwrap();
    for seed in 1u64..=3 {
        let mut rng = seed;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let keys: Vec<_> = (0..n).map(|_| next()).collect();
        let absent: Vec<_> = (0..n).map(|_| next()).collect();
        let expected: std::collections::HashSet<_> = keys.iter().copied().collect();
        let mut input = keys.clone();
        input.extend_from_slice(&keys[..n.min(17)]);
        for alpha in [0.9, 0.95] {
            let sets: Vec<Box<dyn HashSet>> = vec![
                Box::new(Inline::<1, false, false, true>::new(1. / alpha, &input)),
                Box::new(Quotient::<1, 32, false, true, true, true>::new(
                    1. / alpha,
                    &input,
                )),
                Box::new(Inline::<1, true>::new(1. / alpha, &input)),
                Box::new(Inline::<1, true, false, true>::new(1. / alpha, &input)),
                Box::new(Quotient::<1, 32, false, true>::new(1. / alpha, &input)),
                Box::new(Quotient::<1, 32, false, true, true>::new(
                    1. / alpha,
                    &input,
                )),
            ];
            for set in sets {
                for &key in keys.iter().chain(&absent) {
                    assert_eq!(
                        set.contains(key),
                        expected.contains(&key),
                        "{}: seed {seed}, alpha {alpha}, key {key}",
                        set.name()
                    );
                }
                assert_eq!(set.count_loop(&keys), n);
                assert_eq!(set.count_prefetch(&keys), n);
                assert_eq!(set.count_prefetch2(&keys), n);
                eprintln!("{}: seed {seed}, alpha {alpha}, n {n}: passed", set.name());
            }
            let filters: Vec<(u32, Box<dyn HashSet>)> = vec![
                (
                    24,
                    Box::new(Quotient::<1, 32, false, true, true, false, 24>::new(
                        1. / alpha,
                        &input,
                    )),
                ),
                (
                    24,
                    Box::new(Quotient::<1, 32, false, true, true, true, 24>::new(
                        1. / alpha,
                        &input,
                    )),
                ),
                (
                    32,
                    Box::new(Quotient::<1, 32, false, true, true, false, 32>::new(
                        1. / alpha,
                        &input,
                    )),
                ),
                (
                    32,
                    Box::new(Quotient::<1, 32, false, true, true, true, 32>::new(
                        1. / alpha,
                        &input,
                    )),
                ),
                (
                    40,
                    Box::new(Quotient::<1, 32, false, true, true, true, 40>::new(
                        1. / alpha,
                        &input,
                    )),
                ),
            ];
            for (bits, filter) in filters {
                for &key in &keys {
                    assert!(
                        filter.contains(key),
                        "filter false negative: bits {bits}, seed {seed}, alpha {alpha}, key {key}"
                    );
                }
                assert_eq!(filter.count_loop(&keys), n);
                assert_eq!(filter.count_prefetch(&keys), n);
                assert_eq!(filter.count_prefetch2(&keys), n);
                let fp = absent
                    .iter()
                    .filter(|&&key| !expected.contains(&key) && filter.contains(key))
                    .count();
                eprintln!(
                    "{}: bits {bits}, seed {seed}, alpha {alpha}, n {n}: passed, {fp}/{} negative queries accepted, {:.3} bits/key",
                    filter.name(),
                    absent.len(),
                    filter.allocation_size() as f64 * 8. / n as f64
                );
            }
        }
    }
}

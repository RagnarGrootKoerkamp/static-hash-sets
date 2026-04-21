use super::*;
use fph_table::{FphDynSet, FphMetaSet};

#[test]
fn test() {
    let hashers = vec![
        Box::new(hashbrown::HashSet::<T, FxHasher>::default()) as Box<dyn HashSet>,
        Box::new(U64HashSet::new(1.1, &[])),
        Box::new(CuckooSet::<{ Mode::Lazy }>::new(1.1, &[])),
        Box::new(CuckooSet::<{ Mode::Eager }>::new(1.1, &[])),
        Box::new(KphfSet::<{ kphf::Mode::SortBump }, BIN_SIZE>::new(
            0.9,
            2.0 * space_lower_bound(BIN_SIZE, 0.9),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(KphfSet::<{ kphf::Mode::LinearBump }, BIN_SIZE>::new(
            0.9,
            2.0 * space_lower_bound(BIN_SIZE, 0.9),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(KphfSet::<{ kphf::Mode::Linear }, BIN_SIZE>::new(
            0.9,
            2.5 * space_lower_bound(BIN_SIZE, 0.9),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(KphfSet::<{ kphf::Mode::SortBumpGreedy }, BIN_SIZE>::new(
            0.9,
            2.0 * space_lower_bound(BIN_SIZE, 0.9),
            &[],
        )) as Box<dyn HashSet>,
        Box::new(FphDynSet::new(0.9, &[]).unwrap()) as Box<dyn HashSet>,
        Box::new(FphMetaSet::new(0.9, &[]).unwrap()) as Box<dyn HashSet>,
    ];

    for n in [100_000, 1_000_000] {
        let keys = (0..n as u64).collect::<Vec<_>>();
        for hasher in &hashers {
            eprintln!("Test {}", hasher.name());
            test_one(&keys, &**hasher);
        }
    }
}

fn test_one(keys: &[u64], h: &dyn HashSet) {
    let n = keys.len();
    let h = h.new(&keys);

    for i in 0..n {
        assert!(h.contains(i as u64), "Failed for {i}");
    }
    for i in n..2 * n {
        assert!(!h.contains(i as u64), "Failed for {i}");
    }
}

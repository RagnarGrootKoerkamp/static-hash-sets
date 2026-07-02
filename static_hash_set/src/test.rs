use super::*;
#[cfg(feature = "extc")]
use engineering_kphf::{Hd8Set, Tbb84pSet, Tbb85Set};
#[cfg(feature = "extc")]
use fph_table::FphDynSet;
#[cfg(feature = "extc")]
use mapembed::MapEmbed;
use phf_set::PhfSet;

#[test]
fn test() {
    let mut hashers = vec![
        Box::new(U64HashSet::new(1.1, &[])) as Box<dyn HashSet>,
        Box::new(CuckooSet::<{ Mode::Lazy as u8 }>::new(1.1, &[])),
        Box::new(CuckooSet::<{ Mode::Eager as u8 }>::new(1.1, &[])),
    ];

    #[cfg(feature = "kphf")]
    {
        hashers.extend(vec![
        Box::new(
            KphfSet::<KptrHash<{ kphf::Mode::SortBump as u8 }, BIN_SIZE>, BIN_SIZE>::try_new(
                0.9,
                2.0 * space_lower_bound(BIN_SIZE, 0.9),
                &[],
            )
            .unwrap(),
        ) as Box<dyn HashSet>,
        Box::new(
            KphfSet::<KptrHash<{ kphf::Mode::LinearBump as u8 }, BIN_SIZE>, BIN_SIZE>::try_new(
                0.9,
                2.0 * space_lower_bound(BIN_SIZE, 0.9),
                &[],
            )
            .unwrap(),
        ) as Box<dyn HashSet>,
        Box::new(
            KphfSet::<KptrHash<{ kphf::Mode::Linear as u8 }, BIN_SIZE>, BIN_SIZE>::try_new(
                0.9,
                2.5 * space_lower_bound(BIN_SIZE, 0.9),
                &[],
            )
            .unwrap(),
        ) as Box<dyn HashSet>,
        Box::new(
            KphfSet::<KptrHash<{ kphf::Mode::SortBumpGreedy as u8 }, BIN_SIZE>, BIN_SIZE>::try_new(
                0.9,
                2.0 * space_lower_bound(BIN_SIZE, 0.9),
                &[],
            )
            .unwrap(),
        ) as Box<dyn HashSet>,
        ]);
    }

    #[cfg(feature = "ext")]
    {
        hashers.extend(vec![
            Box::new(hashbrown::HashSet::<T, gxhash::GxBuildHasher>::default()) as Box<dyn HashSet>,
            Box::new(PhfSet::<phf_trait::PtrHash>::new(0.0, 0.0, &[])) as Box<dyn HashSet>,
            Box::new(PhfSet::<phf_trait::PHast>::new(0.0, 0.0, &[])) as Box<dyn HashSet>,
        ]);
    }

    #[cfg(feature = "extc")]
    {
        hashers.extend(vec![
            Box::new(FphDynSet::new(0.9, &[]).unwrap()) as Box<dyn HashSet>,
            Box::new(MapEmbed::new(&[]).unwrap()) as Box<dyn HashSet>,
            Box::new(KphfSet::<Tbb85Set, BIN_SIZE>::try_new(0.0, 0.0, &[]).unwrap())
                as Box<dyn HashSet>,
            Box::new(KphfSet::<Tbb84pSet, BIN_SIZE>::try_new(0.0, 0.0, &[]).unwrap())
                as Box<dyn HashSet>,
            Box::new(KphfSet::<Hd8Set, BIN_SIZE>::try_new(0.0, 0.0, &[]).unwrap())
                as Box<dyn HashSet>,
        ]);
    }

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

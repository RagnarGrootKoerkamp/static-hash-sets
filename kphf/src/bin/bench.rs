#![feature(adt_const_params, generic_const_exprs)]
use kphf::{KptrHash, Mode};

fn gen_keys(n: usize) -> Vec<u32> {
    let mut keys = std::collections::HashSet::with_capacity(n);
    let mut buf = vec![0u32; 1024];
    while keys.len() < n {
        rand::fill(&mut buf[..]);
        keys.extend(buf.iter().copied());
    }
    let mut keys: Vec<u32> = keys.into_iter().take(n).collect();
    keys.sort_unstable();
    keys
}

/// Information-theoretic lower bound on bits/key for a static hash function
/// with bin size k and load factor alpha.
fn space_lower_bound(k: usize, alpha: f32) -> f32 {
    // Rows: alpha, columns: k
    const ALPHAS: [f32; 14] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0,
    ];
    const KS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    #[rustfmt::skip]
    const TABLE: [[f32; 8]; 14] = [
        [0.07466, 0.00849, 0.00022, 3e-7,     1e-12,    0.0,      0.0,      0.0     ],
        [0.15498, 0.03112, 0.00259, 4e-5,     2e-8,     2e-14,    0.0,      0.0     ],
        [0.24202, 0.06581, 0.00989, 5.2e-4,   3e-6,     4e-10,    0.0,      0.0     ],
        [0.33724, 0.11221, 0.02444, 2.67e-3,  8e-5,     2e-7,     3e-12,    0.0     ],
        [0.44269, 0.17114, 0.04833, 8.57e-3,  6.8e-4,   1.2e-5,   9e-9,     1e-14   ],
        [0.56140, 0.24472, 0.08410, 2.095e-2, 3.18e-3,  2.0e-4,   2e-6,     6e-10   ],
        [0.69828, 0.33704, 0.13568, 4.360e-2, 1.028e-2, 1.49e-3,  9e-5,     9e-7    ],
        [0.86221, 0.45624, 0.21044, 8.294e-2, 2.699e-2, 6.77e-3,  1.13e-3,  9e-5    ],
        [1.07359, 0.62179, 0.32631, 0.15418,  6.498e-2, 2.404e-2, 7.55e-3,  1.88e-3 ],
        [1.21522, 0.73972, 0.41651, 0.21663,  0.10374,  4.552e-2, 1.815e-2, 6.47e-3 ],
        [1.32751, 0.83742, 0.49629, 0.27699,  0.14560,  7.194e-2, 3.332e-2, 1.444e-2],
        [1.37558, 0.88056, 0.53318, 0.30680,  0.16811,  8.765e-2, 4.339e-2, 2.036e-2],
        [1.43271, 0.93321, 0.58010, 0.34702,  0.20111,  0.11335,  6.222e-2, 2.895e-2],
        [1.44269, 0.94265, 0.58893, 0.35509,  0.20832,  0.11967,  6.761e-2, 3.770e-2],
    ];

    let row = ALPHAS.iter().position(|&x| x == alpha).unwrap();
    let col = KS.iter().position(|&x| x == k).unwrap();
    TABLE[row][col]
}

fn bench<const MODE: Mode, const K: usize>()
where
    [(); K + 1]:,
{
    println!("mode,k,n,alpha,lb,factor,bits_per_key,actual_bpk,pct_bumped,build_ns,throughput_ns,latency_ns");
    for &n in &[100_000usize, 1_000_000, 10_000_000] {
        let keys = gen_keys(n);
        for &alpha in &[0.8, 0.9, 0.95, 0.98, 0.99] {
            let lb = space_lower_bound(K, alpha);
            for &factor in &[5.0, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5] {
                let bits_per_key = lb * factor;

                // Construction
                let t0 = std::time::Instant::now();
                let Some(kphf) = KptrHash::<MODE, K>::new(alpha, bits_per_key, &keys) else {
                    let actual_bpk = 0;
                    let pct_bumped = 0;
                    let build_ns = 0;
                    let throughput_ns = 0;
                    let latency_ns = 0;
                    println!(
                    "{MODE:?},{K},{n},{alpha:.2},{lb:.6},{factor:.2},{bits_per_key:.6},{actual_bpk:.4},{pct_bumped:.4},{build_ns:.1},{throughput_ns:.1},{latency_ns:.1}"
                );
                    break;
                };
                let build_ns = t0.elapsed().as_nanos() as f32 / n as f32;

                // Space
                let actual_bpk = kphf.bits_used() as f32 / n as f32;
                let pct_bumped = kphf.num_bumped() as f32 / n as f32 * 100.0;

                // Query throughput: independent queries
                let t0 = std::time::Instant::now();
                let mut sink = 0usize;
                for &key in &keys {
                    sink ^= kphf.get(key);
                }
                std::hint::black_box(sink);
                let throughput_ns = t0.elapsed().as_nanos() as f32 / n as f32;

                // Query latency: dependent queries to prevent pipelining
                let t0 = std::time::Instant::now();
                let mut x = 0u32;
                for key in &keys {
                    x = kphf.get(key & !(x & 1)) as u32;
                }
                std::hint::black_box(x);
                let latency_ns = t0.elapsed().as_nanos() as f32 / n as f32;

                println!(
                    "{MODE:?},{K},{n},{alpha:.2},{lb:.6},{factor:.2},{bits_per_key:.6},{actual_bpk:.4},{pct_bumped:.4},{build_ns:.1},{throughput_ns:.1},{latency_ns:.1}"
                );
            }
        }
    }
}

fn main() {
    // bench::<{ Mode::Linear }, 4>();
    // bench::<{ Mode::LinearBump }, 4>();
    // bench::<{ Mode::Sort }, 4>();
    // bench::<{ Mode::SortBump }, 4>();

    bench::<{ Mode::Linear }, 8>();
    bench::<{ Mode::LinearBump }, 8>();
    bench::<{ Mode::Sort }, 8>();
    bench::<{ Mode::SortBump }, 8>();

    bench::<{ Mode::Linear }, 16>();
    bench::<{ Mode::LinearBump }, 16>();
    bench::<{ Mode::Sort }, 16>();
    bench::<{ Mode::SortBump }, 16>();
}

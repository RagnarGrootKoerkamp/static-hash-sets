#![allow(incomplete_features)]
#![feature(adt_const_params, generic_const_exprs)]
use kphf::{space_lower_bound, KptrHash, Mode};

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

fn bench<const MODE: Mode, const K: usize>()
where
    [(); K + 1]:,
{
    println!("mode,k,n,alpha,lb,factor,bits_per_key,actual_bpk,pct_bumped,build_ns,throughput_ns,latency_ns");
    for &n in &[3_000_000] {
        let keys = gen_keys(n);
        for &alpha in &[0.8, 0.9, 0.95, 0.98, 0.99] {
            let lb = space_lower_bound(K, alpha);
            for &factor in &[
                5.0, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5, 1.4, 1.3, 1.25, 1.2, 1.15,
            ] {
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
    // bench::<{ Mode::Consensus }, 4>();

    bench::<{ Mode::SortBump }, 8>();
    bench::<{ Mode::LinearBump }, 8>();
    bench::<{ Mode::Sort }, 8>();
    bench::<{ Mode::Linear }, 8>();
    bench::<{ Mode::Consensus }, 8>();

    bench::<{ Mode::SortBump }, 16>();
    bench::<{ Mode::LinearBump }, 16>();
    bench::<{ Mode::Sort }, 16>();
    bench::<{ Mode::Linear }, 16>();
    bench::<{ Mode::Consensus }, 16>();
}

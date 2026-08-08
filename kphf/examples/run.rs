use kphf::{space_lower_bound, KptrHash, Mode};

fn gen_keys(n: usize) -> Vec<u32> {
    let mut keys = std::collections::HashSet::with_capacity(n);
    let mut buf = vec![0u32; 1024];
    while keys.len() < n {
        rand::fill(&mut buf[..]);
        keys.extend(buf.iter().copied());
    }
    let keys: Vec<u32> = keys.into_iter().take(n).collect();
    // keys.sort_unstable();
    // keys.dedup();
    // keys.shuffle(&mut rand::rng());
    keys
}

const REPEATS: usize = 1;
const NS: [usize; 1] = [10_000_000];
const ALPHAS: [f32; 1] = [0.9];
const FACTORS: [f32; 1] = [1.5];

fn bench<const MODE: u8, const K: usize>() {
    for &n in &NS {
        let keys = std::array::from_fn::<_, REPEATS, _>(|_| gen_keys(n));
        for &alpha in &ALPHAS {
            let lb = space_lower_bound(K, alpha);
            for &factor in &FACTORS {
                let target_bits_per_key = lb * factor;

                for (_repeat, keys) in std::iter::zip(0..REPEATS, &keys) {
                    let kphf = KptrHash::<MODE, K>::new(alpha, target_bits_per_key, &keys).unwrap();

                    let bumped_frac = kphf.num_bumped() as f32 / n as f32 * 100.;
                    eprintln!("Bumped: {bumped_frac}%");

                    let start = std::time::Instant::now();
                    let mut c = 0;
                    for &key in keys {
                        c += kphf.get(key);
                    }
                    std::hint::black_box(c);
                    let loop_ns = start.elapsed().as_nanos() as f32 / n as f32;
                    eprintln!("Loop:   {loop_ns}");

                    let start = std::time::Instant::now();
                    let lookahead = 32;
                    let mut c = 0;
                    for i in 0..keys.len().saturating_sub(lookahead) {
                        kphf.prefetch(keys[i + lookahead]);
                        c += kphf.get(keys[i]) as usize;
                    }
                    std::hint::black_box(c);
                    let throughput_ns = start.elapsed().as_nanos() as f32 / n as f32;
                    eprintln!("pref:   {throughput_ns}");
                }
            }
        }
    }
}

fn main() {
    bench::<{ Mode::SortBump as u8 }, 8>();
}

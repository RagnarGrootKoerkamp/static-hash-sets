# k-PtrHash and k-PHF-sets

This repository contains two Rust crates:
- `kphf`: implements k-PtrHash, a non-minimal k-perfect hash function (k-PHF), and
- `static_hash_set`: a static hash set based on `kphf`, alongside some simpler
  cache line-based hash tables.
  
The main motivation is to build a high throughput static hash set. We use the
k-PHF to map each 64-bit key to a cache line, and then use SIMD
instructions to check whether the key is in the cache line or not.
As long as the k-PHF fits in cache, this allows for single-cache-miss
containment queries, which many other data structures cannot achieve:
- Swisstables (as used in Abseil and hashbrown) need probing and have a load factor as low as 0.5;
- eager Cuckoo hashing fetches 2 cache lines;
- lazy Cuckoo hashing still fetches 2 cache lines for negative queries;
- in general we need a load factor as low as 0.7 to have direct-hit positive queries.

By using a k-PHF, a single cache miss suffices as long as the PHF fits in cache.
Because we use a k-PHF rather than a 1-PHF, and because we use a load factor of
e.g. 0.9 instead of 0.98, the space usage of the k-PHF is much smaller than that
of a classical minimal PHF (MPHF).

## k-PtrHash

This is a relatively simple non-minimal k-perfect hash function based on PtrHash
(which itself is based on CHD and PTHash). Given a set of keys, it builds a
small data structure that maps each key to a bucket such that at most `k` keys
are mapped to each bucket. For example, this can be used to make sure at most 8
keys map to a cache line.

```rust
// The number of keys per cache line.
const K: usize = 8;
// The load factor
let alpha = 0.9;
// The space lower bound for this K and alpha.
let lb = space_lower_bound(K, alpha);
// The space overhead we're aiming for.
let target_bits_per_key = 1.5 * lb;
// The keys.
let keys = vec![1, 2, 10, 34, ...];
let n = keys.len();

// Build the kPHF.
let kphf = KptrHash::<MODE, K>::new(alpha, target_bits_per_key, &keys).unwrap();

// Properties of the kPHF

// The finally used number of bits per key. Larger than the target due to bumping.
let actual_bits_per_key = kphf.bits_used() as f32 / n as f32;
// The fraction of keys that was bumped.
let bumped_frac = kphf.num_bumped() as f32 / n as f32;
// The true load factor.
let actual_alpha = n as f32 / (kphf.num_bins() as f32 * K as f32);

// Query the kPHF.
for i in 0..n {
    if i + 32 < n {
        // Optionally, start loading the pPHF memory for upcoming query keys early.
        // Usually not needed since the kphf is small and fits in cache.
        kphf.prefetch(keys[i+32]);
    }
    // get the bucket for this key.
    c += kphf.get(keys[i]);
}
```

**Evals.** 
Experiments in the paper are run using `cd kphf; cargo run -r --example bench >
eval/data.csv`. Then make `plot.pdf` using `cd eval; python plot.py data`.


## static-hash-sets
This crate implements a number of static hash sets:
- A simple cache line-based linear probing hash set.
- A cache line-based cuckoo hash set.
- A PHF-based set that uses a 1-PHF to map each key to a slot.
- A kPHF-based set that uses k-PtrHash as k-PHF to map each key to a cache line.

The `ext/` directory contains a number of other hash set implementations.

Usage example:

```rust
// Build the static hash set
let kphf_set = KphfSet::<KptrHash<{ kphf::Mode::SortBump as u8 }, K>, K>::try_new(
    alpha,
    target_bits_per_key,
    keys,
);

// Query all keys.
for key in &keys {
    let contained: bool = kphf_set.contains(key);
}

// Optionally, prefetch the target cache line of the set.
let token = self.prefetch(keys[i]);
// <do other work>
// Query and reuse the hash that is stored in the token.
let contained: bool = self.contains_with_token(keys[i], token);
```

**Evals.** Use `cargo run -r > eval/data-laptop.csv` to run the experiments (on 1 and
12 threads by default), and `cd eval; python plot.py laptop` to make the plot.

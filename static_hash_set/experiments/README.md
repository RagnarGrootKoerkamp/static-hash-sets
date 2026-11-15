# Plain eight-slot cuckoo construction

Run locally from `static_hash_set`:

```
just placement baseline
just placement paper
just placement parameters
python3 experiments/report_placement.py > experiments/placement-report.md
```

The standalone Rust source is `src/cuckoo_construction.rs`. It uses bins of
exactly eight u64 keys and no table metadata. Experimental bookkeeping lives
in separate arrays, since construction time and working memory are not measured.
Independent seeded SplitMix64 streams generate the two choices and priority.
This preserves the plain table's layout and insertion policy, but does not
reproduce its gxhash mapping or nondeterministic eviction RNG.

Each run inserts one million distinct nonzero keys into ceil(n / (8 alpha))
bins. Baseline uses first-empty primary, then secondary, then random kicking
starting at the primary. Oracle uses exhaustive augmenting paths and certifies
infeasibility if it fails. Its primary fraction is not globally optimal.

The section 3 and section 4 policies follow Bell and Kuszmaul, *Fast Insertion
for Bucketized Cuckoo Hashing*, sections 3.1 and 4.1. Section 5 adds the eviction
priority and within-bin ordering from section 5.1. The default switch is 1.4
revealed hashes per slot. Coincident hash choices count as a single reachable
bucket; visiting it does not reveal an independent alternate location.

Sweep parameters: 3 hash seeds; load 50% through 99.9%; 1,000 versus 100,000
kicks per insertion for parameter runs (1,000,000 for main sweeps); switching
at 0, 1.2, 1.4, or 1.6 revealed hashes per slot. Failures include algorithm cycle
detection and the configured kick limit; they do not establish infeasibility
unless the oracle fails. An unsuccessful insertion may have evicted a key, so
primary/prediction statistics are reported only for complete constructions.
All complete constructions run the same end-to-end check: each input key
appears exactly once and belongs to one of its two candidate bins.

Prediction uses h2 when priority is below a global threshold, h1 otherwise.
The optimized threshold is computed exactly by sorting all inserted priorities
and sweeping all cuts. Reported accuracy is fitted to this constructed table,
not estimated on independently sampled keys. Baseline/paper variants use the
same mappings and insertion order per seed and load. Three seeds show variation,
not a rigorous success-probability estimate.

The asymptotic maximum for two random choices and bucket capacity eight is
computed from the 9-core orientability criterion. With Q_k(x) the Poisson(x)
upper tail Pr[X >= k], solve x Q_8(x)/Q_9(x) = 16, then
alpha = x/(16 Q_8(x)) = 0.997853282996. This is a random-instance limit, not a
universal upper bound for every finite table. Absolute capacity permits 100%.

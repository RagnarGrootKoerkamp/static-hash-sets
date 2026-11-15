# Plain eight-slot cuckoo construction

Run locally from `static_hash_set`:

```
just placement baseline
just placement paper
just placement parameters
just placement threshold
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

## Results for n = 1,000,000

See `placement-report.md` for all rows and the CSV files for individual seeds.
The baseline reaches 99.77% in all three sampled graphs. At 99.78%, baseline
succeeds in 1/3 and the exact oracle in 2/3. At 99.79%, both succeed in 1/3;
at 99.8%, neither succeeds. Changing bin count changes range reduction and hence
the graph, so a fixed seed does not define nested graphs across target loads.

With the paper's 1.4 switch, sections 3, 4 and 5 reach respectively 98%, 99%,
and 99.5% on all three seeds in the coarse sweep. Starting look-ahead immediately
raises section 5 to 99.7% for all three seeds and greatly improves the primary
fraction. At 99% it keeps 71.13% primary and predicts 74.16% correctly, versus
56.09% primary for baseline. At 95% it keeps 82.61% primary, versus baseline's
74.00%. These are empirical results for capacity eight; the paper's guarantees
assume sufficiently large capacity and do not promise identical behavior here.

The section 5 priority predictor is useful mainly at high load. Using the default
switch, its optimized accuracy is about 75.15% at 95% load and 72.40% at 99%.
Starting look-ahead immediately improves the 99% accuracy to 74.16%. Thresholds
are optimized separately for each completed table; full threshold results are
in the report. No timings are measured.

Increasing the per-insertion kick limit from 1,000 to 100,000 changes two tested
runs from failure to success: immediate-look-ahead section 4 at 99.5% (seed 2),
and immediate-look-ahead section 5 at 99.7% (seed 1). Other parameter runs are
unchanged. The switch is evaluated at insertion boundaries; once the revealed
hash count crosses it, subsequent insertions use look-ahead throughout.

## Bumped keys per bin

```
just placement bumped
python3 experiments/report_bumped.py > experiments/bumped-report.md
```

This focused sweep uses n=1,000,000, loads 95%/98%, seeds 1/2/3, and a
1,000,000-kick limit. It includes baseline, oracle, sections 3/4/5 at the
default switch, and sections 4/5 with immediate look-ahead. All 42 runs succeed.

The driver records two exact count histograms as semicolon-separated CSV fields:
`bumped_from_hist[k]` counts bins whose primary hashes have exactly k keys stored
elsewhere; `secondary_in_hist[k]` counts bins currently storing exactly k keys
at their secondary choice. These have equal totals of secondary keys, but
very different distributions. Coincident hash choices count as primary.
Histogram index zero includes empty bins and bins with no applicable keys.
The report pools bin counts across seeds and includes means, p95, and observed
maxima. Histogram sums and weighted sums are checked against bin/key totals.

## Priority-ordered BFS and threshold prediction

```
just placement priority-bfs
python3 experiments/report_priority_bfs.py > experiments/priority-bfs-report.md
```

This local-only experiment compares `oracle-shuffled` with `bfs-large` at
90%/95% load, n=1M and three seeds. Both use the same keys 1..n and the same
seeded Fisher-Yates insertion shuffle, avoiding correlation of key size with
arrival time. The priority is the key itself, rather than the independent
priority hash used by section 5. `bfs-large` orders primary-resident eviction
edges by descending key, then secondary-resident return edges by ascending key.
It preserves BFS depth order and explores every reachable edge, so this is
only a tie-breaking policy among shortest augmenting paths.

The new modes optimize `secondary_first = key > cutoff` exactly over all cut
positions. `optimal_threshold` stores cutoff/n. Coincident candidate bins are
counted as successful first probes for either prediction. The selected cutoff
is checked against actual placements. See `priority-bfs-report.md` for results.
The original section 5 experiment and its lower-priority prediction direction
are unchanged.

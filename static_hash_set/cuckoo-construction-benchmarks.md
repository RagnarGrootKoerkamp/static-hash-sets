# Filtered cuckoo construction variants

- **Random walk:** Original construction, primary then secondary, then random
  displacement. Quotienting uses the v11 spill-before-SIMD lookup.
- **BFS:** Starts from both candidate buckets, chooses a shortest path to an
  empty slot, and relocates occupants backwards along that path. Construction
  workspace is dropped before queries. Quotient entries are decoded and
  re-encoded for each new bucket; zero primary encodings are excluded. Spill
  summaries are rebuilt after all insertions. Query layout and logic are the
  same as the corresponding random-walk table.
- **Inline 2hot64 BFS:** Seven full-width keys plus a 64-bit summary per bin.
  Two distinct bits are derived from the existing primary hash.
- **Quotient 2hot32 BFS:** Reserves only the final four bytes of the 64-byte bin
  for its two-hot summary. Whole-byte entries occupy the preceding 60 bytes:
  e.g., ten 6-byte entries instead of nine in the 64-bit-summary variant.
  Equality masks use the new slot count and filter rebuilds preserve key bytes
  in the final u64. Both 2-hot variants select the second bit via an offset
  modulo (number of bits minus one), guaranteeing two distinct bits.

Reproduce with `just bench-bfs-hellman` and `just bench-twohot-hellman`.
Both accept optional sizes and loads. Defaults: the 25 existing sizes,
90%/95% load, 1/16/192 threads, quick mode, one million queries per thread,
one pass, 2.551 GHz. Only the requested methods run. CSV merging preserves
sizes and loads not present in the new run.

Validation uses `just fuzz-cuckoo 1000000`, a single shared end-to-end driver
for all six variants: three seeds, both loads, duplicates, positive/negative
membership, and loop/prefetch/prefetch2 counts. AVX-512 build flags must be
passed to the binary, not globally to dependencies.

## BFS comparison, Hellman job 2174

Completed successfully. Raw data: `eval/cuckoo-bfs.csv` (1,800 rows), merged
into `eval/latest-hellman.csv`; the interactive plot includes BFS alongside
random walk at both loads. All six variants also passed the shared 1M-key fuzz
check locally and with AVX-512 enabled on the server, for three seeds and both
loads. All binaries, including standalone profiling programs, pass cargo check.

Ratios below are geometric means over 25 sizes. Query columns are q50 time
ratios (BFS/random walk, smaller is better); build is construction time ratio.
One quick pass per configuration, with no reruns.

| Table | Load | Build | Loop, 1 thread | Prefetch2, 1 thread | Loop, 192 threads | Prefetch2, 192 threads |
|---|---:|---:|---:|---:|---:|---:|
| Inline 1hot | 95% | 2.65 | 0.863 | 0.854 | 0.954 | 0.942 |
| Inline 1hot | 98% | 2.30 | 0.800 | 0.788 | 0.913 | 0.894 |
| Quotient 1hot | 95% | 3.39 | 0.923 | 0.865 | 0.973 | 0.956 |
| Quotient 1hot | 98% | 3.36 | 0.827 | 0.790 | 0.923 | 0.909 |

For a short comparison use `just bench-bfs-hellman 1000000 0.9,0.95`.
For the new filters use `just bench-twohot-hellman 1000000 0.9,0.95`.

Future benchmark and fuzz defaults use 90%/95% load. The completed comparison
above used 95%/98%; those historical measurements are preserved.

## Two-hot comparison, Hellman job 2175

Completed successfully. Raw data: `eval/cuckoo-twohot.csv` (900 rows), merged
into the combined CSV and plot. Full tables are in `cuckoo-bfs-results.md`;
regenerate them with `just report-cuckoo-bfs`. That report also accepts future
matched size/load sweeps by deriving the size set and loads from the data.

At n=60M and 95% target load, the 64-bit inline two-hot filter reduces negative
filter passes from 1.897% to 0.315%. The 32-bit quotient two-hot filter reduces
passes from 2.012% to 1.438%, while reducing memory/input from 0.9357 to 0.8421
(10% less than the 64-bit-summary quotient table, 30% less than inline).

Both two-hot variants are slower to query in this implementation despite their
lower filter-pass rates. For 50%-positive queries, single-thread prefetch2 time
is about 2.09x/1.96x the inline one-hot BFS time at 95%/98% load. Quotient
32-bit two-hot is about 1.20x/1.18x the quotient one-hot BFS time. At 192 threads,
the quotient variant is about 8%/5% slower. These are geometric means across
25 sizes from separate quick runs, not repeated paired measurements.

Future benchmark defaults remain 90%/95%. The 98% data above belongs to the
already-submitted historical comparison.

## Two-hot v2: shared lookup flow and cheap bit selection

Inline two-hot now uses the same spill-conditioned lookup flow as one-hot,
including the token-based prefetch path. Both filters select positions directly
from adjacent bit slices of the existing primary hash: bits 0..5 and 6..11
for 64 bits; bits 0..4 and 5..9 for 32 bits. Coincident positions are allowed,
as requested. There is no modulo operation or collision correction. Such a
key/query occasionally uses a one-bit mask (probability 1/64 or 1/32 under a
uniform hash). The earlier distinct-bit probability estimates describe v1's
idealization, rather than this collision-allowed distribution.

The shared fuzz driver passed with one million keys, three seeds, and 90%/95%
loads. A focused Hellman run (job 2176) covered 1M/16M keys, all three thread
counts, 90%/95% load, and matching one-hot BFS baselines. Raw data is saved as
`eval/cuckoo-twohot-v2-focused.csv`. The full changed-method sweep is job 2177;
its output is `eval/cuckoo-twohot-v2.csv`. Plot series are labelled v2, preserving
the earlier measurements. `just bench-twohot-hellman` now runs the current v2
code; `just bench-twohot-v2-hellman` allows method selection for focused checks.

Focused 95% load, single-thread loop, 50%-positive queries, ns/query:

| Table | n | Previous two-hot | v2 two-hot | Matching one-hot BFS |
|---|---:|---:|---:|---:|
| Inline | 1M | 22.43 | 8.40 | 8.39 |
| Inline | 16M | 31.61 | 18.98 | 17.02 |
| Quotient | 1M | 11.00 | 8.19 | 8.27 |
| Quotient | 16M | 25.46 | 21.91 | 20.00 |

These are separate quick runs; small timing differences should not be treated
as statistically established improvements.

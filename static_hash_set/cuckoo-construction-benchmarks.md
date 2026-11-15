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

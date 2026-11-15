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
95%/98% load, 1/16/192 threads, quick mode, one million queries per thread,
one pass, 2.551 GHz. Only the requested methods run. CSV merging preserves
sizes and loads not present in the new run.

Validation uses `just fuzz-cuckoo 1000000`, a single shared end-to-end driver
for all six variants: three seeds, both loads, duplicates, positive/negative
membership, and loop/prefetch/prefetch2 counts. AVX-512 build flags must be
passed to the binary, not globally to dependencies.

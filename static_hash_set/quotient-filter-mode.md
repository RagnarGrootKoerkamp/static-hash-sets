# Retained-width quotient filter mode

`cuckoo_quotienting::CuckooFilter::<32>::new(slot_ratio, keys)` retains the top
32 bits of each seeded primary/secondary hash before quotienting. Every input
key remains a full-width nonzero u64. The default alias uses lazy BFS lookup,
byte-aligned entries, and the existing 32-bit two-hot spill summary. Use
`CuckooFilter::<32, false>` for the 64-bit one-hot spill summary.

The underlying type adds a final const parameter:

```
CuckooSet<MODE, PREFIX, EF, FILTER, BFS, FILTER32, HASH_BITS = 64>
```

Existing types remain exact with the default 64 bits. Reduced widths require
the byte-packed BFS layout. At m buckets, the logical entry width is
`HASH_BITS + 1 - floor(log2(m))`, including the location marker. Production
entries round up to whole bytes and bucket capacity follows the retained width.
The bucket index supplies the removed hash bits implicitly. HASH_BITS is the
retained width before quotienting, not the per-query false-positive exponent:
without other rejection mechanisms the rare-collision rate is about n/2^HASH_BITS.
The existing spill summaries can further reject secondary false positives.

Construction keeps full keys in temporary slots, finds BFS augmenting paths,
then writes each truncated quotient and builds spill summaries. Temporary full
keys and BFS state are dropped before returning the filter. Full-key duplicate
checks during construction prevent fingerprint collisions from discarding
input keys. Primary-zero encodings are routed to secondary as in the exact table.
All queries, including token-based prefetch, use the same truncated hash domain.

The filter is static: original-key iteration and incremental `insert` are
unavailable for HASH_BITS < 64 because the discarded key bits cannot be recovered.
Rebuild with `new` or the HashSet trait constructor instead. Introspection methods
expose `retained_hash_bits()`, `bits_per_entry()`, and `slots_per_bin()`.

Benchmark selectors:

- `quotient-filter32-onehot-bfs`
- `quotient-filter32-twohot32-bfs`

Both are registered with 90%/95% load defaults and plot styles. The existing
CSV/plot has not been populated with timing runs for these new filters.

## Validation and local statistics

The shared `just fuzz-cuckoo 1000000` driver covers the existing exact tables
and 24/32/40-bit filters, three seeds, 90%/95% load, duplicates, membership of
all inserted keys, and loop/prefetch/prefetch2 counts. Filter negatives are
counted as false positives rather than asserted absent. The expanded driver
also passed with AVX-512 on the server, exercising the smaller entry widths.
All binaries pass `cargo check`.

For n=1M, pooled over three local seeds (3M negative queries per row):

| Retained bits | Spill summary | Load | Payload bits/key | Measured FP |
|---:|---|---:|---:|---:|
| 32 | 64-bit 1-hot | 90% | 31.606 | 0.02223% |
| 32 | 32-bit 2-hot | 90% | 28.445 | 0.02250% |
| 32 | 64-bit 1-hot | 95% | 29.942 | 0.02160% |
| 32 | 32-bit 2-hot | 95% | 26.948 | 0.02213% |

At this size both production variants use 18 logical bits (17-bit quotient and
one marker), rounded to three bytes. Their 64-byte buckets hold 18 or 20 entries
respectively. This differs from the fixed-eight-slot mathematical example,
which needs 16 logical bits at this retained width.

The separate `just placement-filter` prototype holds eight logical slots per
bucket and bit-packs its quotients with a four-bit occupancy count. Its complete
retained-width sweep and measured false-positive rates are recorded in
`experiments/filter-report.md`. It measures placement/storage/filter quality;
it does not measure timings or replace the production table's byte alignment.

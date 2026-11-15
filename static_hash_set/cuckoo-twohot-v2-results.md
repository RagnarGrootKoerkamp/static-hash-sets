# Two-hot v2 results

Hellman job 2177: 25 sizes, 90%/95% load, 1M queries/thread, one quick pass, 2.551 GHz. No 98% runs.

V2 uses the one-hot lookup control flow and two bit slices of the existing hash. Coincident positions are allowed.

Query numbers below use 50%-positive queries (q50), geometric means over all sizes, ns/query.

| Method | Load | Loop, 1 thread | Prefetch2, 1 thread | Loop, 192 threads | Prefetch2, 192 threads |
|---|---:|---:|---:|---:|---:|
| Inline 2hot64 v2 | 90% | 13.6970 | 7.7277 | 0.1685 | 0.1361 |
| Quotient 2hot32 v2 | 90% | 15.6150 | 8.5537 | 0.1765 | 0.1391 |
| Inline 2hot64 v2 | 95% | 14.5569 | 8.3448 | 0.1712 | 0.1435 |
| Quotient 2hot32 v2 | 95% | 16.5751 | 9.1561 | 0.1772 | 0.1414 |

## Time ratios at 95% load

Each comparison covers all sizes; lower than 1 is faster. These are separate quick runs, not repeated paired measurements.

| Method | Reference | Loop, 1 thread | Prefetch2, 1 thread | Loop, 192 threads | Prefetch2, 192 threads |
|---|---|---:|---:|---:|---:|
| Inline 2hot64 v2 | Two-hot v1 | 0.523 | 0.500 | 0.746 | 0.843 |
| Inline 2hot64 v2 | One-hot BFS | 1.126 | 1.043 | 1.061 | 1.034 |
| Quotient 2hot32 v2 | Two-hot v1 | 0.831 | 0.871 | 0.917 | 0.944 |
| Quotient 2hot32 v2 | One-hot BFS | 1.087 | 1.046 | 1.032 | 1.023 |

## Metadata at n=60,000,000

| Method | Target load | Actual load | Memory/input | Primary keys | Negative filter passes |
|---|---:|---:|---:|---:|---:|
| Inline 2hot64 v2 | 90% | 90.000% | 1.2698 | 85.59% | 0.252% |
| Quotient 2hot32 v2 | 90% | 90.000% | 0.8889 | 89.15% | 1.037% |
| Inline 2hot64 v2 | 95% | 95.000% | 1.2030 | 81.49% | 0.366% |
| Quotient 2hot32 v2 | 95% | 95.000% | 0.8421 | 85.76% | 1.538% |

## Focused same-run comparison at n=16M

Job 2176 includes matching one-hot BFS baselines at both loads; single-thread prefetch2, 50%-positive queries.

| Method | Load | Memory/input | ns/query |
|---|---:|---:|---:|
| Inline 2hot64 v2 | 90% | 1.2698 | 9.0396 |
| Quotient 2hot32 v2 | 90% | 0.8889 | 9.0332 |
| Inline 1hot BFS | 90% | 1.2698 | 8.8760 |
| Quotient 1hot BFS | 90% | 0.9877 | 8.9586 |
| Inline 2hot64 v2 | 95% | 1.2030 | 9.5693 |
| Quotient 2hot32 v2 | 95% | 0.8421 | 9.7749 |
| Inline 1hot BFS | 95% | 1.2030 | 9.7243 |
| Quotient 1hot BFS | 95% | 0.9357 | 9.5295 |

# Filtered cuckoo comparisons on Hellman

25 sizes, 1M queries/thread, one quick pass, 2.551 GHz.

## Target load 95%

Metadata at n=60,000,000:

| Method | Actual load | Memory/input | Build ns/key | Primary keys | Negative filter passes |
|---|---:|---:|---:|---:|---:|
| Inline 1hot RW | 95.000% | 1.2030 | 131.89 | 71.26% | 2.917% |
| Inline 1hot BFS | 95.000% | 1.2030 | 338.89 | 81.50% | 1.897% |
| Quotient 1hot RW | 95.000% | 0.9357 | 161.99 | 76.32% | 3.094% |
| Quotient 1hot BFS | 95.000% | 0.9357 | 533.07 | 84.65% | 2.012% |
| Inline 2hot64 BFS | 95.000% | 1.2030 | 339.23 | 81.50% | 0.315% |
| Quotient 2hot32 BFS | 95.000% | 0.8421 | 526.76 | 85.77% | 1.438% |

50% positive queries (q50), geometric mean over all sizes, ns/query (lower is better):

| Method | Loop, 1 thread | Prefetch2, 1 thread | Loop, 192 threads | Prefetch2, 192 threads |
|---|---:|---:|---:|---:|
| Inline 1hot RW | 14.9880 | 9.3685 | 0.1691 | 0.1473 |
| Inline 1hot BFS | 12.9295 | 8.0002 | 0.1614 | 0.1389 |
| Quotient 1hot RW | 16.5153 | 10.1231 | 0.1764 | 0.1447 |
| Quotient 1hot BFS | 15.2433 | 8.7532 | 0.1717 | 0.1383 |
| Inline 2hot64 BFS | 27.8464 | 16.6980 | 0.2294 | 0.1703 |
| Quotient 2hot32 BFS | 19.9541 | 10.5084 | 0.1933 | 0.1499 |

## Target load 98%

Metadata at n=60,000,000:

| Method | Actual load | Memory/input | Build ns/key | Primary keys | Negative filter passes |
|---|---:|---:|---:|---:|---:|
| Inline 1hot RW | 98.000% | 1.1662 | 198.16 | 59.74% | 4.194% |
| Inline 1hot BFS | 98.000% | 1.1662 | 435.74 | 77.94% | 2.365% |
| Quotient 1hot RW | 98.000% | 0.9070 | 210.10 | 65.30% | 4.613% |
| Quotient 1hot BFS | 98.000% | 0.9070 | 667.98 | 81.48% | 2.456% |
| Inline 2hot64 BFS | 98.000% | 1.1662 | 432.64 | 77.94% | 0.402% |
| Quotient 2hot32 BFS | 98.000% | 0.8163 | 641.73 | 82.76% | 1.726% |

50% positive queries (q50), geometric mean over all sizes, ns/query (lower is better):

| Method | Loop, 1 thread | Prefetch2, 1 thread | Loop, 192 threads | Prefetch2, 192 threads |
|---|---:|---:|---:|---:|
| Inline 1hot RW | 17.2171 | 10.9839 | 0.1801 | 0.1538 |
| Inline 1hot BFS | 13.7668 | 8.6588 | 0.1644 | 0.1375 |
| Quotient 1hot RW | 18.6521 | 12.1154 | 0.1889 | 0.1570 |
| Quotient 1hot BFS | 15.4322 | 9.5751 | 0.1744 | 0.1427 |
| Inline 2hot64 BFS | 28.0353 | 16.9337 | 0.2314 | 0.1718 |
| Quotient 2hot32 BFS | 20.1311 | 11.2646 | 0.1970 | 0.1496 |


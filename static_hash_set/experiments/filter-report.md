# Retained hash bits and quotient filtering

Local n=1M, three seeds, 90%/95% load, eight logical slots per bucket. Each run validates every inserted full key and samples 1M known-negative full-width keys. No timing measurements.

Frozen storage bit-packs quotient+location entries with a four-bit occupancy count per bucket. It retains no full keys. Production tables additionally round entries to bytes and use fixed 64-byte buckets.

| Load | Retained bits | Quotient bits | Entry bits incl. marker | Payload bits/key | False positives / 3M | Measured FP | n / 2^f |
|---|---:|---:|---:|---:|---:|---:|---:|
| 90% | 20 | 3 | 4 | 5.000 | 1,878,883 | 62.629433% | 95.367432% |
| 90% | 24 | 7 | 8 | 9.444 | 172,821 | 5.760700% | 5.960464% |
| 90% | 28 | 11 | 12 | 13.889 | 11,005 | 0.366833% | 0.372529% |
| 90% | 32 | 15 | 16 | 18.333 | 708 | 0.023600% | 0.023283% |
| 90% | 40 | 23 | 24 | 27.222 | 0 | 0.000000% | 0.000091% |
| 90% | 64 | 47 | 48 | 53.889 | 0 | 0.000000% | 0.000000% |
| 95% | 20 | 3 | 4 | 4.737 | 1,878,086 | 62.602867% | 95.367432% |
| 95% | 24 | 7 | 8 | 8.947 | 172,930 | 5.764333% | 5.960464% |
| 95% | 28 | 11 | 12 | 13.158 | 11,131 | 0.371033% | 0.372529% |
| 95% | 32 | 15 | 16 | 17.368 | 735 | 0.024500% | 0.023283% |
| 95% | 40 | 23 | 24 | 25.790 | 0 | 0.000000% | 0.000091% |
| 95% | 64 | 47 | 48 | 51.053 | 0 | 0.000000% | 0.000000% |

The n/2^f column is a rare-collision approximation; at high collision rates it overestimates the result. At f=64 the reversible hash and lossless quotient make membership exact. Very small rates have few events in this sample, so their estimates are coarse.


# Priority BFS, n=1,000,000

Eight u64 keys per bucket. Keys 1..n, shuffled identically for both methods per seed. Three seeds, 90%/95% load.

BFS-large keeps shortest-path BFS, visiting large primary residents first and small secondary residents next. It does not force a displacement when a root bucket has space.

Prediction checks secondary first iff key > cutoff. Every cut is evaluated; coincident candidate buckets always count as a correct first probe. The normalized CSV threshold is cutoff/n.

| Method | Load | Success | Primary keys | Best first-probe accuracy | Optimal cutoff range |
|---|---:|---:|---:|---:|---:|
| bfs-large | 90% | 3/3 | 87.03123% | 87.03130% | 999,999–1,000,000 |
| oracle-shuffled | 90% | 3/3 | 87.02860% | 87.02863% | 999,999–1,000,000 |
| bfs-large | 95% | 3/3 | 83.22727% | 83.22733% | 999,999–1,000,000 |
| oracle-shuffled | 95% | 3/3 | 83.22280% | 83.22287% | 999,999–1,000,000 |

The optimum is effectively always-primary. Eviction tie ordering alone provides no meaningful prediction gain at these loads.

## Hard forcing to secondary

Keys above force_cutoff may only occupy h2. BFS excludes illegal return edges; failure exhausts the reachable graph and certifies infeasibility under that constraint.

| Load | Forced largest keys | Success | Primary keys | Best first-probe accuracy | Optimal cutoff range |
|---|---:|---:|---:|---:|---:|
| 90% | 0.0% | 3/3 | 87.0312% | 87.0313% | 999,999–1,000,000 |
| 90% | 1.0% | 3/3 | 86.0152% | 87.0152% | 990,000–990,000 |
| 90% | 2.5% | 3/3 | 84.4963% | 86.9964% | 974,999–975,000 |
| 90% | 5.0% | 3/3 | 81.9656% | 86.9656% | 949,989–950,000 |
| 90% | 7.5% | 3/3 | 79.4372% | 86.9373% | 924,999–925,000 |
| 90% | 10.0% | 3/3 | 76.9045% | 86.9045% | 900,000–900,000 |
| 90% | 12.5% | 2/3 | 74.3412% | 86.8413% | 874,997–874,999 |
| 90% | 15.0% | 2/3 | 71.8066% | 86.8066% | 850,000–850,000 |
| 90% | 17.5% | 0/3 | — | — | — |
| 90% | 20.0% | 0/3 | — | — | — |
| 90% | 25.0% | 0/3 | — | — | — |
| 95% | 0.0% | 3/3 | 83.2273% | 83.2273% | 999,999–1,000,000 |
| 95% | 1.0% | 3/3 | 82.1864% | 83.1864% | 989,997–990,000 |
| 95% | 2.5% | 3/3 | 80.6313% | 83.1313% | 974,993–975,000 |
| 95% | 5.0% | 3/3 | 78.0606% | 83.0606% | 949,989–950,000 |
| 95% | 7.5% | 3/3 | 75.4806% | 82.9806% | 924,999–925,000 |
| 95% | 10.0% | 3/3 | 72.8889% | 82.8889% | 900,000–900,000 |
| 95% | 12.5% | 3/3 | 70.2962% | 82.7964% | 874,980–875,000 |
| 95% | 15.0% | 1/3 | 67.7003% | 82.7002% | 849,999–849,999 |
| 95% | 17.5% | 0/3 | — | — | — |
| 95% | 20.0% | 0/3 | — | — | — |
| 95% | 25.0% | 0/3 | — | — | — |

Accuracy averages successful constructions only. With three seeds, success fractions are descriptive, not reliable probability estimates. Hash mappings change with bin count, so the two loads use different graphs.

Forcing does not improve accuracy here. The optimum usually follows the forcing cutoff, and larger forced fractions eventually make placement infeasible.

## Conditional secondary fraction: BFS-large, 95% load, seed 1

| Key percentile | Secondary fraction |
|---|---:|
| 0–10% | 11.692% |
| 10–20% | 12.310% |
| 20–30% | 12.771% |
| 30–40% | 13.866% |
| 40–50% | 14.779% |
| 50–60% | 15.989% |
| 60–70% | 17.937% |
| 70–80% | 19.759% |
| 80–90% | 22.579% |
| 90–100% | 26.057% |

Priority biases placement, but even the largest decile is predominantly primary. Flipping an upper tail only helps when more than half of its distinct-choice keys are secondary.


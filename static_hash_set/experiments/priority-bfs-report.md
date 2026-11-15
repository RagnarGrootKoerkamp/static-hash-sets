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


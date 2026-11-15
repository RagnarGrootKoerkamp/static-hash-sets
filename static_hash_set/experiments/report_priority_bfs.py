#!/usr/bin/env python3
"""Placement and globally optimized high-key secondary prediction."""
import csv
from collections import defaultdict
from statistics import mean
from pathlib import Path

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'placement-priority-bfs.csv').open()))
print('# Priority BFS, n=1,000,000\n')
print('Eight u64 keys per bucket. Keys 1..n, shuffled identically for both methods per seed. Three seeds, 90%/95% load.\n')
print('BFS-large keeps shortest-path BFS, visiting large primary residents first and small secondary residents next. It does not force a displacement when a root bucket has space.\n')
print('Prediction checks secondary first iff key > cutoff. Every cut is evaluated; coincident candidate buckets always count as a correct first probe. The normalized CSV threshold is cutoff/n.\n')
print('| Method | Load | Success | Primary keys | Best first-probe accuracy | Optimal cutoff range |')
print('|---|---:|---:|---:|---:|---:|')
groups=defaultdict(list)
for r in rows:groups[(float(r['load']),r['algorithm'])].append(r)
for (load,alg),rs in sorted(groups.items()):
    ok=[r for r in rs if r['status']=='ok']
    cuts=[round(float(r['optimal_threshold'])*int(r['n'])) for r in ok]
    print(f'| {alg} | {load:.0%} | {len(ok)}/{len(rs)} | {100*mean(float(r["primary"]) for r in ok):.5f}% | {100*mean(float(r["optimal_prediction"]) for r in ok):.5f}% | {min(cuts):,}–{max(cuts):,} |')
print('\nThe optimum is effectively always-primary. Eviction tie ordering alone provides no meaningful prediction gain at these loads.\n')

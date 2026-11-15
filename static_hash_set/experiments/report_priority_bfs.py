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
forced=root/'placement-forced-bfs.csv'
if forced.exists():
    print('## Hard forcing to secondary\n')
    print('Keys above force_cutoff may only occupy h2. BFS excludes illegal return edges; failure exhausts the reachable graph and certifies infeasibility under that constraint.\n')
    print('| Load | Forced largest keys | Success | Primary keys | Best first-probe accuracy | Optimal cutoff range |')
    print('|---|---:|---:|---:|---:|---:|')
    groups=defaultdict(list)
    for r in csv.DictReader(forced.open()):groups[(round(float(r['load']),2),float(r['force_fraction']))].append(r)
    for (load,fraction),rs in sorted(groups.items()):
        ok=[r for r in rs if r['status']=='ok']
        if ok:
            cuts=[round(float(r['optimal_threshold'])*int(r['n'])) for r in ok]
            fields=f'{100*mean(float(r["primary"]) for r in ok):.4f}% | {100*mean(float(r["optimal_prediction"]) for r in ok):.4f}% | {min(cuts):,}–{max(cuts):,}'
        else:fields='— | — | —'
        print(f'| {load:.0%} | {fraction:.1%} | {len(ok)}/{len(rs)} | {fields} |')
    print('\nAccuracy averages successful constructions only. With three seeds, success fractions are descriptive, not reliable probability estimates. Hash mappings change with bin count, so the two loads use different graphs.\n')
    print('Forcing does not improve accuracy here. The optimum usually follows the forcing cutoff, and larger forced fractions eventually make placement infeasible.\n')
bands=root/'priority-bfs-key-bands.csv'
if bands.exists():
    print('## Conditional secondary fraction: BFS-large, 95% load, seed 1\n')
    print('| Key percentile | Secondary fraction |')
    print('|---|---:|')
    for r in csv.DictReader(bands.open()):
        print(f'| {r["percentile_low"]}–{r["percentile_high"]}% | {100*int(r["secondary_count"])/int(r["key_count"]):.3f}% |')
    print('\nPriority biases placement, but even the largest decile is predominantly primary. Flipping an upper tail only helps when more than half of its distinct-choice keys are secondary.\n')
evict=root/'placement-priority-evict-bfs.csv'
if evict.exists():
    print('## Priority eviction before accepting the incoming secondary\n')
    print('A full primary bucket first accepts the incoming key by evicting its largest eligible primary resident. BFS searches from the victim\'s secondary, with an exhaustive two-root fallback if necessary. The pure variant only selects keys never previously placed secondary; the all variant allows every primary resident.\n')
    print('| Method | Load | Success | Primary-first | Optimal prediction | Gain (percentage points) | Optimal cutoff range |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    groups=defaultdict(list)
    for r in csv.DictReader(evict.open()):groups[(round(float(r['load']),2),r['algorithm'])].append(r)
    for (load,alg),rs in sorted(groups.items()):
        ok=[r for r in rs if r['status']=='ok']
        p=mean(float(r['primary']) for r in ok)
        accuracy=mean(float(r['optimal_prediction']) for r in ok)
        cuts=[round(float(r['optimal_threshold'])*int(r['n'])) for r in ok]
        print(f'| {alg} | {load:.0%} | {len(ok)}/{len(rs)} | {100*p:.4f}% | {100*accuracy:.4f}% | {100*(accuracy-p):.4f} | {min(cuts):,}–{max(cuts):,} |')
    print('\nMoving priority selection ahead of the incoming secondary creates a useful threshold, especially at 95% load. It also reduces the primary fraction: optimized prediction is still below ordinary BFS at these loads. This hybrid does not implement the paper\'s complete displacement rules or invariants.\n')

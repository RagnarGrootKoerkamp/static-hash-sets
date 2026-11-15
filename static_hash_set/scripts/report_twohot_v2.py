#!/usr/bin/env python3
"""Report collision-allowed two-hot lookup versus the original implementation."""
import csv
import math
from pathlib import Path

root = Path(__file__).resolve().parent.parent

def read(name):
    with (root / 'eval' / name).open(newline='') as f:
        return list(csv.DictReader(f))

new = read('cuckoo-twohot-v2.csv')
old = read('cuckoo-twohot.csv')
focused = read('cuckoo-twohot-v2-focused.csv')
base = read('cuckoo-bfs.csv')
assert {r['alpha'] for r in new} == {'0.9', '0.95'}
sizes = {r['n'] for r in new}
names = {
    'InlineTwoHotCuckooSet<Lazy,BumpedOnly,BFS>': 'Inline 2hot64 v2',
    'CuckooQuotientSet<Lazy,32,SIMD,TwoHot32,BFS>': 'Quotient 2hot32 v2',
}
baselines = {
    'InlineTwoHotCuckooSet<Lazy,BumpedOnly,BFS>': 'InlineOneHotCuckooSet<Lazy,BumpedOnly,BFS>',
    'CuckooQuotientSet<Lazy,32,SIMD,TwoHot32,BFS>': 'CuckooQuotientSet<Lazy,32,SIMD,OneHot,BFS>',
}

def times(rows, h, alpha, thread, metric):
    return {r['n']: float(r['q50']) for r in rows if r['h'] == h and r['alpha'] == alpha and r['threads'] == thread and r['metric'] == metric}

def gm(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))

print('# Two-hot v2 results\n')
print(f'Hellman job 2177: {len(sizes)} sizes, 90%/95% load, 1M queries/thread, one quick pass, 2.551 GHz. No 98% runs.\n')
print('V2 uses the one-hot lookup control flow and two bit slices of the existing hash. Coincident positions are allowed.\n')
print('Query numbers below use 50%-positive queries (q50), geometric means over all sizes, ns/query.\n')
print('| Method | Load | Loop, 1 thread | Prefetch2, 1 thread | Loop, 192 threads | Prefetch2, 192 threads |')
print('|---|---:|---:|---:|---:|---:|')
for alpha in ['0.9', '0.95']:
    for h, label in names.items():
        vals = []
        for th, mode in [('1','loop'),('1','prefetch2'),('192','loop'),('192','prefetch2')]:
            d = times(new, h, alpha, th, mode)
            assert d.keys() == sizes
            vals.append(f'{gm(list(d.values())):.4f}')
        print(f'| {label} | {float(alpha):.0%} | ' + ' | '.join(vals) + ' |')

print('\n## Time ratios at 95% load\n')
print('Each comparison covers all sizes; lower than 1 is faster. These are separate quick runs, not repeated paired measurements.\n')
print('| Method | Reference | Loop, 1 thread | Prefetch2, 1 thread | Loop, 192 threads | Prefetch2, 192 threads |')
print('|---|---|---:|---:|---:|---:|')
for h, label in names.items():
    for reference, rows, ref_h in [('Two-hot v1', old, h), ('One-hot BFS', base, baselines[h])]:
        vals = []
        for th, mode in [('1','loop'),('1','prefetch2'),('192','loop'),('192','prefetch2')]:
            a = times(new,h,'0.95',th,mode)
            b = {n: v for n, v in times(rows,ref_h,'0.95',th,mode).items() if n in sizes}
            assert a.keys() == b.keys() == sizes
            vals.append(f'{gm([a[n]/b[n] for n in sizes]):.3f}')
        print(f'| {label} | {reference} | ' + ' | '.join(vals) + ' |')

largest = str(max(map(int,sizes)))
print(f'\n## Metadata at n={int(largest):,}\n')
print('| Method | Target load | Actual load | Memory/input | Primary keys | Negative filter passes |')
print('|---|---:|---:|---:|---:|---:|')
for alpha in ['0.9','0.95']:
    for h,label in names.items():
        rs=[r for r in new if r['h']==h and r['n']==largest and r['alpha']==alpha and r['threads']=='1' and r['metric']=='loop']
        assert len(rs)==1
        r=rs[0]
        print(f'| {label} | {float(alpha):.0%} | {float(r["load_factor"]):.3%} | {float(r["memory_ratio"]):.4f} | {float(r["primary_hit_ratio"]):.2%} | {float(r["negative_filter_pass_ratio"]):.3%} |')

print('\n## Focused same-run comparison at n=16M\n')
print('Job 2176 includes matching one-hot BFS baselines at both loads; single-thread prefetch2, 50%-positive queries.\n')
print('| Method | Load | Memory/input | ns/query |')
print('|---|---:|---:|---:|')
for alpha in ['0.9','0.95']:
    for r in focused:
        if r['n']=='16000000' and r['alpha']==alpha and r['threads']=='1' and r['metric']=='prefetch2':
            label=(names | {'InlineOneHotCuckooSet<Lazy,BumpedOnly,BFS>': 'Inline 1hot BFS', 'CuckooQuotientSet<Lazy,32,SIMD,OneHot,BFS>': 'Quotient 1hot BFS'}).get(r['h'],r['h'])
            print(f'| {label} | {float(alpha):.0%} | {float(r["memory_ratio"]):.4f} | {float(r["q50"]):.4f} |')

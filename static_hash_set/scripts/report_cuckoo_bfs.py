#!/usr/bin/env python3
"""Compare the focused BFS and two-hot runs without mixing historical data."""
import csv
import math
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rows = []
for filename in ['cuckoo-bfs.csv', 'cuckoo-twohot.csv']:
    with (root / 'eval' / filename).open(newline='') as f:
        rows.extend(csv.DictReader(f))

names = {
    'InlineOneHotCuckooSet<Lazy,BumpedOnly>': 'Inline 1hot RW',
    'InlineOneHotCuckooSet<Lazy,BumpedOnly,BFS>': 'Inline 1hot BFS',
    'CuckooQuotientSet<Lazy,32,SIMD,OneHot>': 'Quotient 1hot RW',
    'CuckooQuotientSet<Lazy,32,SIMD,OneHot,BFS>': 'Quotient 1hot BFS',
    'InlineTwoHotCuckooSet<Lazy,BumpedOnly,BFS>': 'Inline 2hot64 BFS',
    'CuckooQuotientSet<Lazy,32,SIMD,TwoHot32,BFS>': 'Quotient 2hot32 BFS',
}
print('# Filtered cuckoo comparisons on Hellman\n')
sizes = {r['n'] for r in rows}
print(f'{len(sizes)} sizes, 1M queries/thread, one quick pass, 2.551 GHz.\n')
for alpha in sorted({r['alpha'] for r in rows}, key=float):
    print(f'## Target load {float(alpha):.0%}\n')
    largest = str(max(map(int, sizes)))
    print(f'Metadata at n={int(largest):,}:\n')
    print('| Method | Actual load | Memory/input | Build ns/key | Primary keys | Negative filter passes |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, label in names.items():
        rs = [r for r in rows if r['h'] == name and r['alpha'] == alpha and r['n'] == largest and r['threads'] == '1' and r['metric'] == 'loop']
        assert len(rs) == 1, (name, alpha, len(rs))
        r = rs[0]
        print(f'| {label} | {float(r["load_factor"]):.3%} | {float(r["memory_ratio"]):.4f} | {float(r["build"]):.2f} | {float(r["primary_hit_ratio"]):.2%} | {float(r["negative_filter_pass_ratio"]):.3%} |')
    print('\n50% positive queries (q50), geometric mean over all sizes, ns/query (lower is better):\n')
    print('| Method | Loop, 1 thread | Prefetch2, 1 thread | Loop, 192 threads | Prefetch2, 192 threads |')
    print('|---|---:|---:|---:|---:|')
    for name, label in names.items():
        values = []
        for thread, metric in [('1', 'loop'), ('1', 'prefetch2'), ('192', 'loop'), ('192', 'prefetch2')]:
            rs = [r for r in rows if r['h'] == name and r['alpha'] == alpha and r['threads'] == thread and r['metric'] == metric]
            assert {r["n"] for r in rs} == sizes, (name, alpha, thread, metric, len(rs))
            gm = math.exp(sum(math.log(float(r['q50'])) for r in rs) / len(rs))
            values.append(f'{gm:.4f}')
        print(f'| {label} | ' + ' | '.join(values) + ' |')
    print()

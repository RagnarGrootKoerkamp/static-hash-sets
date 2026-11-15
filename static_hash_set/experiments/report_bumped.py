#!/usr/bin/env python3
"""Bin-weighted histograms across all seeds in the focused placement sweep."""
import csv
from collections import defaultdict
from pathlib import Path

rows=list(csv.DictReader(Path('experiments/placement-bumped.csv').open()))
groups=defaultdict(list)
for row in rows:
    groups[(round(float(row['load']),2),row['algorithm'],row['switch'])].append(row)

def label(algorithm,switch):
    return algorithm + (' immediate' if algorithm in ('section4','section5') and float(switch)==0 else '')

def aggregate(rs,column):
    hist=[]
    for r in rs:
        counts=list(map(int,r[column].split(';')))
        hist.extend([0]*max(0,len(counts)-len(hist)))
        for i,c in enumerate(counts):hist[i]+=c
    return hist

def quantile(hist,fraction):
    target=sum(hist)*fraction
    total=0
    for k,c in enumerate(hist):
        total+=c
        if total>=target:return k

for column,title in [('bumped_from_hist','Keys bumped FROM their primary bin'),('secondary_in_hist','Secondary keys stored IN the bin')]:
    print(f'## {title}\n')
    print('Percent of all bins, pooled over three seeds; empty bins are included.\n')
    for load in [.95,.98]:
        print(f'### Target load {load:.0%}\n')
        print('| Method | Success | Mean | p95 | Max | '+' | '.join(map(str,range(9)))+' | 9+ |')
        print('|---|---:|---:|---:|---:|'+ '|'.join(['---:']*10)+'|')
        for (l,a,s),rs in groups.items():
            if l!=load:continue
            ok=[r for r in rs if r['status']=='ok']
            if not ok:continue
            hist=aggregate(ok,column);den=sum(hist)
            mean=sum(k*c for k,c in enumerate(hist))/den
            values=[hist[k] if k<len(hist) else 0 for k in range(9)]+[sum(hist[9:])]
            print(f'| {label(a,s)} | {len(ok)}/{len(rs)} | {mean:.3f} | {quantile(hist,.95)} | {len(hist)-1} | '+' | '.join(f'{100*v/den:.2f}' for v in values)+' |')
        print()

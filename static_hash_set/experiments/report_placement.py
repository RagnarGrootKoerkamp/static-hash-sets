#!/usr/bin/env python3
import csv, math
from collections import defaultdict
from pathlib import Path

def tail(k,x):
 return 1-sum(math.exp(-x)*x**i/math.factorial(i) for i in range(k))
lo,hi=8.,24.
for _ in range(100):
 x=(lo+hi)/2
 if x*tail(8,x)/tail(9,x)>16:hi=x
 else:lo=x
x=(lo+hi)/2
print(f'Asymptotic random two-choice, 8-slot threshold: {x/(16*tail(8,x)):.10%}.')
print('Solves x*Pr[Poisson(x)>=8] / Pr[Poisson(x)>=9] = 16; alpha=x/(16*Pr[Poisson(x)>=8]).')
for phase in ['baseline','paper','parameters']:
 path=Path(f'experiments/placement-{phase}.csv')
 if not path.exists():continue
 groups=defaultdict(list)
 for row in csv.DictReader(path.open()):
  groups[(row['algorithm'],round(float(row['load']),3),row['budget'],row['switch'])].append(row)
 print(f'\n{phase}\n')
 print('| Algorithm | Load % | Budget | Switch | Success | Primary % | Best prediction % | Threshold |')
 print('|---|---:|---:|---:|---:|---:|---:|---:|')
 for (a,l,b,s),rs in sorted(groups.items()):
  ok=[r for r in rs if r['status']=='ok']
  def avg(k,scale=1):return f'{sum(float(r[k]) for r in ok)/len(ok)*scale:.3f}' if ok else '—'
  print(f'| {a} | {l*100:g} | {b} | {s} | {len(ok)}/{len(rs)} | {avg("primary",100)} | {avg("optimal_prediction",100)} | {avg("optimal_threshold")} |')

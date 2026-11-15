#!/usr/bin/env python3
"""Local placement sweeps; results include every seed, including failures."""
import csv, subprocess, io, argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=1000000);p.add_argument('--seeds',default='1,2,3');p.add_argument('--output',default='experiments/placement-results.csv');p.add_argument('--phase',choices=['baseline','paper','parameters','threshold','bumped','priority-bfs','forced-bfs'],default='baseline');args=p.parse_args()
subprocess.run(['rustc','-O','--edition=2024','src/cuckoo_construction.rs','-o','/tmp/cuckoo-construction'],check=True)
loads=[.5,.7,.8,.9,.95,.97,.98,.99,.995,.997,.998,.999]
configs=[(a,1000000,1.4) for a in (['baseline','oracle'] if args.phase in ('baseline','threshold') else ['section3','section4','section5'])]
if args.phase=='forced-bfs':
 loads=[.9,.95]
 configs=[('bfs-forced',1000000,1.4,f) for f in [0,.01,.025,.05,.075,.1,.125,.15,.175,.2,.25]]
if args.phase=='priority-bfs':
 loads=[.9,.95]
 configs=[(a,1000000,1.4) for a in ['oracle-shuffled','bfs-large']]
if args.phase=='bumped':
 loads=[.95,.98]
 configs=[(a,1000000,s) for a,s in [('baseline',1.4),('oracle',1.4),('section3',1.4),('section4',1.4),('section5',1.4),('section4',0),('section5',0)]]
if args.phase=='threshold':
 loads=[.9975,.9977,.9978,.9979]
if args.phase=='parameters':
 loads=[.95,.99,.995,.997]
 configs=[(a,b,s) for a in ['section4','section5'] for b in [1000,100000] for s in [0,1.2,1.4,1.6]]
with open(args.output,'w',newline='') as f:
 writer=None
 for seed in map(int,args.seeds.split(',')):
  for load in loads:
   for config in configs:
    alg,budget,switch=config[:3]
    forced=config[3] if len(config)>3 else 0
    out=subprocess.run(['/tmp/cuckoo-construction',str(args.n),alg,str(load),str(seed),str(budget),str(switch),str(forced)],capture_output=True,text=True,check=True).stdout
    row=next(csv.DictReader(io.StringIO(out)))
    if writer is None:writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader()
    writer.writerow(row);f.flush()
    print(alg,seed,load,row['status'],row['primary'],row['optimal_prediction'],flush=True)

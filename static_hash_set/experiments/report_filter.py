#!/usr/bin/env python3
"""Summarize the local fixed-eight-slot frozen quotient-filter experiment."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/'placement-filter.csv').open()))
assert len(rows)==36
assert all(r['status']=='ok' and None not in r for r in rows)
groups=defaultdict(list)
for r in rows:groups[(round(float(r['load']),2),int(r['retained_hash_bits']))].append(r)
print('# Retained hash bits and quotient filtering\n')
print('Local n=1M, three seeds, 90%/95% load, eight logical slots per bucket. Each run validates every inserted full key and samples 1M known-negative full-width keys. No timing measurements.\n')
print('Frozen storage bit-packs quotient+location entries with a four-bit occupancy count per bucket. It retains no full keys. Production tables additionally round entries to bytes and use fixed 64-byte buckets.\n')
print('| Load | Retained bits | Quotient bits | Entry bits incl. marker | Payload bits/key | False positives / 3M | Measured FP | n / 2^f |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
for (load,bits),rs in sorted(groups.items()):
    assert len(rs)==3
    negatives=sum(int(r['negative_queries']) for r in rs)
    positives=sum(int(r['false_positives']) for r in rs)
    if bits==64:assert positives==0
    print(f'| {load:.0%} | {bits} | {rs[0]["quotient_bits"]} | {rs[0]["stored_entry_bits"]} | {mean(float(r["packed_bits_per_key"]) for r in rs):.3f} | {positives:,} | {100*positives/negatives:.6f}% | {100*int(rs[0]["n"])/2**bits:.6f}% |')
print('\nThe n/2^f column is a rare-collision approximation; at high collision rates it overestimates the result. At f=64 the reversible hash and lossless quotient make membership exact. Very small rates have few events in this sample, so their estimates are coarse.\n')

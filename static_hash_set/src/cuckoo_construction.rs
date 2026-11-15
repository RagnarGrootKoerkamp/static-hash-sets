//! Placement-only experiments for plain two-choice cuckoo hashing: eight u64s/bin.
//! Standalone: rustc -O --edition=2024 src/cuckoo_construction.rs -o /tmp/cuckoo-construction
//! Hashes use independent seeded SplitMix64 streams, rather than gxhash, so this
//! experiment has no dependencies. IDs 1..=n are nonzero, distinct u64 keys.
use std::env;
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn prefix_hash(key: u64, seed: u64, side: usize, bits: u32) -> u64 {
    mix(key ^ mix(seed + side as u64)) >> (64 - bits)
}

// Frozen query representation: four occupancy bits per bin, followed by eight
// fixed-width quotient+location entries. Construction keys are not retained.
struct PackedFilter {
    bins: usize,
    hash_bits: u32,
    quotient_bits: u32,
    entry_bits: u32,
    stride: usize,
    seed: u64,
    words: Vec<u64>,
}
impl PackedFilter {
    fn mask(bits: u32) -> u64 {
        if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
    }
    fn write(&mut self, offset: usize, bits: u32, value: u64) {
        debug_assert_eq!(value & !Self::mask(bits), 0);
        let word = offset / 64;
        let shift = offset % 64;
        self.words[word] |= value << shift;
        if shift + bits as usize > 64 {
            self.words[word + 1] |= value >> (64 - shift);
        }
    }
    fn read(&self, offset: usize, bits: u32) -> u64 {
        let word = offset / 64;
        let shift = offset % 64;
        let mut value = self.words[word] >> shift;
        if shift + bits as usize > 64 {
            value |= self.words[word + 1] << (64 - shift);
        }
        value & Self::mask(bits)
    }
    fn bin(&self, hash: u64) -> usize {
        ((hash as u128 * self.bins as u128) >> self.hash_bits) as usize
    }
    fn entry(&self, hash: u64, bin: usize, side: usize) -> u64 {
        let lower = ((bin as u128) << self.hash_bits).div_ceil(self.bins as u128);
        let quotient = hash as u128 - lower;
        debug_assert!(quotient < (1u128 << self.quotient_bits));
        ((quotient as u64) << 1) | side as u64
    }
    fn freeze(experiment: &Experiment) -> Self {
        let hash_bits = experiment.filter_bits.unwrap();
        let span = (1u128 << hash_bits).div_ceil(experiment.m as u128);
        let quotient_bits = 128 - (span - 1).leading_zeros();
        let entry_bits = quotient_bits + 1;
        assert!(entry_bits <= 64);
        let stride = 4 + 8 * entry_bits as usize;
        let mut filter = Self {
            bins: experiment.m,
            hash_bits,
            quotient_bits,
            entry_bits,
            stride,
            seed: experiment.seed,
            words: vec![0; (experiment.m * stride).div_ceil(64)],
        };
        for bin in 0..experiment.m {
            let count = experiment.bins[bin].iter().filter(|&&key| key != 0).count();
            assert!(experiment.bins[bin][..count].iter().all(|&key| key != 0));
            filter.write(bin * stride, 4, count as u64);
            for slot in 0..count {
                let key = experiment.bins[bin][slot];
                let side = usize::from(!experiment.first(key, bin));
                let hash = prefix_hash(key, experiment.seed, side, hash_bits);
                assert_eq!(filter.bin(hash), bin);
                filter.write(
                    bin * stride + 4 + slot * entry_bits as usize,
                    entry_bits,
                    filter.entry(hash, bin, side),
                );
            }
        }
        filter
    }
    fn contains(&self, key: u64) -> bool {
        for side in 0..2 {
            let hash = prefix_hash(key, self.seed, side, self.hash_bits);
            let bin = self.bin(hash);
            let target = self.entry(hash, bin, side);
            let count = self.read(bin * self.stride, 4) as usize;
            if (0..count).any(|slot| {
                self.read(
                    bin * self.stride + 4 + slot * self.entry_bits as usize,
                    self.entry_bits,
                ) == target
            }) {
                return true;
            }
        }
        false
    }
}

struct Experiment {
    n: usize,
    m: usize,
    seed: u64,
    rng: u64,
    bins: Vec<[u64; 8]>,
    h: Vec<[usize; 2]>,
    priority: Vec<u64>,
    revealed: Vec<bool>,
    stamp: Vec<usize>,
    bfs_generation: usize,
    parent: Vec<usize>,
    parent_slot: Vec<usize>,
    reveals: usize,
    placed: usize,
    algorithm: String,
    budget: usize,
    switch_at: f64,
    force_fraction: f64,
    force_cutoff: u64,
    filter_bits: Option<u32>,
    filter_queries: usize,
}
impl Experiment {
    fn new(
        n: usize,
        alpha: f64,
        seed: u64,
        algorithm: String,
        budget: usize,
        switch_at: f64,
        force_fraction: f64,
        filter_bits: Option<u32>,
        filter_queries: usize,
    ) -> Self {
        let mut m = (n as f64 / alpha / 8.).ceil() as usize;
        if let Some(bits) = filter_bits {
            assert!((1..=64).contains(&bits));
            m = m.max(2);
            assert!(
                (1u128 << bits) >= m as u128,
                "retained hash domain must cover the bins"
            );
        }
        let mut h = vec![[0; 2]; n + 1];
        let mut priority = vec![0; n + 1];
        for x in 1..=n {
            for side in 0..2 {
                h[x][side] = ((prefix_hash(x as u64, seed, side, filter_bits.unwrap_or(64))
                    as u128
                    * m as u128)
                    >> filter_bits.unwrap_or(64)) as usize;
            }
            priority[x] = if ["bfs-large", "oracle-shuffled", "bfs-forced"]
                .contains(&algorithm.as_str())
                || algorithm.starts_with("bfs-priority")
            {
                x as u64
            } else {
                mix(x as u64 ^ mix(seed + 2))
            };
        }
        Self {
            n,
            m,
            seed,
            rng: seed,
            bins: vec![[0; 8]; m],
            h,
            priority,
            revealed: vec![false; n + 1],
            stamp: vec![0; m],
            bfs_generation: 0,
            parent: vec![0; m],
            parent_slot: vec![0; m],
            reveals: 0,
            placed: 0,
            algorithm,
            budget,
            switch_at,
            force_cutoff: (n as f64 * (1. - force_fraction)).round() as u64,
            force_fraction,
            filter_bits,
            filter_queries,
        }
    }
    fn first(&self, x: u64, b: usize) -> bool {
        self.h[x as usize][0] == b
    }
    fn pure(&self, x: u64, b: usize) -> bool {
        self.first(x, b) && !self.revealed[x as usize]
    }
    fn empty(&self, b: usize) -> bool {
        self.bins[b][7] == 0
    }
    fn forced(&self, key: u64) -> bool {
        self.algorithm == "bfs-forced" && key > self.force_cutoff
    }
    // Exhaustive augmenting paths: failure certifies this graph cannot be placed.
    fn bfs(&mut self, x: u64) -> bool {
        self.bfs_search(x, false)
    }
    fn bfs_search(&mut self, x: u64, secondary_only: bool) -> bool {
        self.bfs_generation = self.bfs_generation.wrapping_add(1);
        if self.bfs_generation == 0 {
            self.stamp.fill(0);
            self.bfs_generation = 1;
        }
        let tag = self.bfs_generation;
        let mut q = Vec::new();
        let candidates = self.h[x as usize];
        let roots = if self.forced(x) || secondary_only {
            &candidates[1..]
        } else {
            &candidates[..]
        };
        for &b in roots {
            if self.stamp[b] != tag {
                self.stamp[b] = tag;
                self.parent[b] = self.m;
                q.push(b);
            }
        }
        let mut i = 0;
        while i < q.len() {
            let mut b = q[i];
            i += 1;
            if let Some(mut s) = self.bins[b].iter().position(|&y| y == 0) {
                while self.parent[b] != self.m {
                    let p = self.parent[b];
                    let t = self.parent_slot[b];
                    let moved = self.bins[p][t];
                    if self.algorithm.starts_with("bfs-priority") && !self.first(moved, b) {
                        self.revealed[moved as usize] = true;
                    }
                    self.bins[b][s] = moved;
                    s = t;
                    b = p;
                }
                self.bins[b][s] = x;
                if self.algorithm.starts_with("bfs-priority") && !self.first(x, b) {
                    self.revealed[x as usize] = true;
                }
                return true;
            }
            let mut order = [0, 1, 2, 3, 4, 5, 6, 7];
            if self.algorithm == "bfs-large"
                || self.algorithm == "bfs-forced"
                || self.algorithm.starts_with("bfs-priority")
            {
                // Prefer sending large primary keys to secondary. If a return
                // to primary is necessary, prefer returning a small key.
                order.sort_unstable_by_key(|&s| {
                    let key = self.bins[b][s];
                    if self.first(key, b) {
                        (0, self.n as u64 - key)
                    } else {
                        (1, key)
                    }
                });
            }
            for s in order {
                let y = self.bins[b][s] as usize;
                // Hard-forced keys have only one legal bucket and cannot
                // participate in an eviction edge back to primary.
                if self.forced(y as u64) {
                    continue;
                }
                let dest = self.h[y][0] ^ self.h[y][1] ^ b;
                if self.stamp[dest] != tag {
                    self.stamp[dest] = tag;
                    self.parent[dest] = b;
                    self.parent_slot[dest] = s;
                    q.push(dest);
                }
            }
        }
        false
    }
    // Keep the last slot available to absorb returns from previously revealed hashes.
    fn tidy(&mut self, b: usize, slot: usize, prioritized: bool, second: bool) {
        if slot != 7 {
            return;
        }
        let mut best = None;
        for s in 0..8 {
            if self.first(self.bins[b][s], b)
                && best.is_none_or(|t: usize| {
                    prioritized
                        && self.priority[self.bins[b][s] as usize]
                            < self.priority[self.bins[b][t] as usize]
                })
            {
                best = Some(s);
            }
        }
        if let Some(s) = best {
            if prioritized || second {
                self.bins[b].swap(7, s);
            }
        }
    }
    // Make the priority decision at a full primary bucket before considering
    // the incoming key's secondary. BFS then finds space for the victim.
    fn insert_priority_evict(&mut self, x: u64) -> bool {
        let primary = self.h[x as usize][0];
        if let Some(slot) = self.bins[primary].iter().position(|&key| key == 0) {
            self.bins[primary][slot] = x;
            return true;
        }
        let pure_only = self.algorithm == "bfs-priority-evict";
        let victim = (0..8)
            .filter(|&s| {
                let y = self.bins[primary][s];
                self.first(y, primary) && (!pure_only || !self.revealed[y as usize])
            })
            .max_by_key(|&s| self.bins[primary][s]);
        if let Some(slot) = victim {
            let y = std::mem::replace(&mut self.bins[primary][slot], x);
            // Search h2 first, even if the incoming key's h2 had free space.
            // A failed search leaves the table unchanged; restoring both roots
            // retains the exhaustive augmenting-path feasibility guarantee.
            self.bfs_search(y, true) || self.bfs_search(y, false)
        } else {
            self.bfs(x)
        }
    }

    fn insert(&mut self, mut x: u64) -> bool {
        if self.algorithm.starts_with("bfs-priority") {
            return self.insert_priority_evict(x);
        }
        if ["oracle", "oracle-shuffled", "bfs-large", "bfs-forced"]
            .contains(&self.algorithm.as_str())
        {
            return self.bfs(x);
        }
        let baseline = self.algorithm == "baseline";
        let prioritized = self.algorithm == "section5";
        let lookahead = self.algorithm != "section3"
            && !baseline
            && (self.placed + self.reveals) as f64 >= self.switch_at * (8 * self.m) as f64;
        let mut b = self.h[x as usize][0];
        self.placed += 1;
        if baseline {
            for d in self.h[x as usize] {
                if self.empty(d) {
                    let s = self.bins[d].iter().position(|&y| y == 0).unwrap();
                    self.bins[d][s] = x;
                    return true;
                }
            }
        }
        let mut just2 = false;
        let mut run_start = self.m;
        let mut visits = 0;
        for _ in 0..self.budget {
            let visited_before = self.first(x, b) || self.revealed[x as usize];
            // Coincident hash choices are already reachable via the first choice.
            if b == self.h[x as usize][1]
                && b != self.h[x as usize][0]
                && !self.revealed[x as usize]
            {
                self.revealed[x as usize] = true;
                self.reveals += 1;
            }
            if self.empty(b) {
                let s = self.bins[b].iter().position(|&y| y == 0).unwrap();
                self.bins[b][s] = x;
                self.tidy(b, s, prioritized, !self.first(x, b));
                return true;
            }
            let mut rule2 = false;
            let victim = if baseline {
                self.rng = mix(self.rng);
                (self.rng % 8) as usize
            } else {
                let all_secondary = (0..7).all(|s| !self.first(self.bins[b][s], b));
                rule2 = all_secondary || (just2 && visited_before);
                if rule2 {
                    7
                } else {
                    let mut candidates: Vec<_> = (0..if prioritized { 8 } else { 7 })
                        .filter(|&s| self.pure(self.bins[b][s], b))
                        .collect();
                    if prioritized {
                        candidates.sort_by_key(|&s| self.priority[self.bins[b][s] as usize]);
                    }
                    assert!(!candidates.is_empty(), "pure-first invariant failed");
                    if lookahead {
                        candidates
                            .iter()
                            .copied()
                            .find(|&s| self.empty(self.h[self.bins[b][s] as usize][1]))
                            .unwrap_or(candidates[0])
                    } else {
                        candidates[0]
                    }
                }
            };
            if rule2 {
                if run_start == self.m {
                    run_start = b;
                    visits = 0;
                }
                if b == run_start {
                    visits += 1;
                    if visits == 3 {
                        return false;
                    }
                }
            } else {
                run_start = self.m;
                visits = 0;
            }
            let second = !self.first(x, b);
            std::mem::swap(&mut x, &mut self.bins[b][victim]);
            if !baseline && !rule2 {
                self.tidy(b, victim, prioritized, second);
            }
            b = self.h[x as usize][0] ^ self.h[x as usize][1] ^ b;
            just2 = rule2;
        }
        false
    }
    fn run(&mut self) {
        let mut inserted = 0;
        let high_priority = ["bfs-large", "oracle-shuffled", "bfs-forced"]
            .contains(&self.algorithm.as_str())
            || self.algorithm.starts_with("bfs-priority");
        let mut insertion_order: Vec<_> = (1..=self.n as u64).collect();
        if high_priority {
            let mut shuffle_rng = mix(self.seed ^ 0xd1b54a32d192ed03);
            for i in (1..insertion_order.len()).rev() {
                shuffle_rng = mix(shuffle_rng);
                insertion_order.swap(i, (shuffle_rng % (i as u64 + 1)) as usize);
            }
        }
        for x in insertion_order {
            if !self.insert(x) {
                break;
            }
            inserted += 1;
        }
        print!(
            "{},{},{},{},{:.8},{},{:.2},{:.8},{},",
            self.algorithm,
            self.n,
            self.seed,
            self.m,
            self.n as f64 / (8 * self.m) as f64,
            self.budget,
            self.switch_at,
            self.force_fraction,
            self.force_cutoff
        );
        if inserted != self.n {
            println!("failed,{inserted},,,,,,,,,,,,");
            return;
        }
        // One shared end-to-end placement check, run for every successful construction.
        let mut seen = vec![false; self.n + 1];
        let mut primary = 0;
        let mut labels = Vec::with_capacity(self.n);
        let mut bumped_from = vec![0usize; self.m];
        let mut secondary_in = vec![0usize; self.m];
        for b in 0..self.m {
            for x in self.bins[b] {
                if x == 0 {
                    continue;
                }
                let x = x as usize;
                assert!(x <= self.n && !seen[x] && (b == self.h[x][0] || b == self.h[x][1]));
                if self.forced(x as u64) {
                    assert_eq!(b, self.h[x][1], "forced key must stay secondary");
                }
                seen[x] = true;
                let p = self.first(x as u64, b);
                primary += usize::from(p);
                if !p {
                    bumped_from[self.h[x][0]] += 1;
                    secondary_in[b] += 1;
                }
                labels.push((self.priority[x], p));
            }
        }
        assert_eq!(labels.len(), self.n);
        let histogram = |counts: &[usize]| {
            let mut hist = vec![0usize; counts.iter().copied().max().unwrap_or(0) + 1];
            for &count in counts {
                hist[count] += 1;
            }
            assert_eq!(hist.iter().sum::<usize>(), self.m);
            assert_eq!(counts.iter().sum::<usize>(), self.n - primary);
            hist.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(";")
        };
        let bumped_hist = histogram(&bumped_from);
        let secondary_hist = histogram(&secondary_in);
        labels.sort_unstable();
        let mut correct = primary;
        let mut best = correct;
        let mut threshold = if high_priority { 1. } else { 0. };
        if high_priority {
            // Begin with all keys predicted primary (cutoff n), then sweep
            // descending keys. At key x, cutoff x-1 flips exactly that key.
            let mut cutoff = self.n as u64;
            for &(key, is_primary) in labels.iter().rev() {
                if self.h[key as usize][0] != self.h[key as usize][1] {
                    if is_primary {
                        correct -= 1;
                    } else {
                        correct += 1;
                    }
                }
                if correct > best {
                    best = correct;
                    cutoff = key - 1;
                }
            }
            threshold = cutoff as f64 / self.n as f64;
            let verified = labels
                .iter()
                .filter(|&&(key, is_primary)| {
                    self.h[key as usize][0] == self.h[key as usize][1]
                        || (key <= cutoff) == is_primary
                })
                .count();
            assert_eq!(
                verified, best,
                "threshold prediction count must match sweep"
            );
        } else {
            for &(p, is_primary) in &labels {
                if is_primary {
                    correct -= 1;
                } else {
                    correct += 1;
                }
                if correct > best {
                    best = correct;
                    threshold = p as f64 / 18446744073709551616.;
                }
            }
        }
        let filter_stats = if self.filter_bits.is_some() {
            let filter = PackedFilter::freeze(self);
            // Shared end-to-end validation: every original full key must match
            // its frozen quotient representation, including truncated collisions.
            for key in 1..=self.n as u64 {
                assert!(filter.contains(key), "filter false negative for {key}");
            }
            let mut tested = 0usize;
            let mut positives = 0usize;
            let mut counter = 0u64;
            while tested < self.filter_queries {
                let key = mix(counter ^ mix(self.seed ^ 0xa0761d6478bd642f));
                counter += 1;
                if key <= self.n as u64 {
                    continue;
                }
                tested += 1;
                positives += usize::from(filter.contains(key));
            }
            if filter.hash_bits == 64 {
                assert_eq!(positives, 0, "full-width reversible hashes must remain exact");
            }
            format!(
                "{},{},{},{:.8},{},{},{:.10}",
                filter.hash_bits,
                filter.quotient_bits,
                filter.entry_bits,
                filter.words.len() as f64 * 64. / self.n as f64,
                tested,
                positives,
                positives as f64 / tested as f64
            )
        } else {
            ",,,,,,".to_string()
        };
        println!(
            "ok,{inserted},{:.8},{:.8},{threshold:.8},{bumped_hist},{secondary_hist},{filter_stats}",
            primary as f64 / self.n as f64,
            best as f64 / self.n as f64
        );
    }
}
fn main() {
    let mut args: Vec<_> = env::args().collect();
    let mut option = |name: &str| -> Option<String> {
        let index = args.iter().position(|s| s == name)?;
        assert!(index + 1 < args.len(), "option requires a value");
        let value = args[index + 1].clone();
        args.drain(index..index + 2);
        Some(value)
    };
    let filter_bits = option("--filter-bits").map(|s| s.parse::<u32>().unwrap());
    let filter_queries =
        option("--filter-queries").map_or(1000000, |s| s.parse::<usize>().unwrap());
    assert!(filter_queries > 0);
    let arg = |i, default: &str| args.get(i).cloned().unwrap_or(default.to_string());
    let n = arg(1, "1000000").parse().unwrap();
    let algorithm = arg(2, "baseline");
    let alpha = arg(3, "0.95").parse::<f64>().unwrap();
    let seed = arg(4, "1").parse().unwrap();
    let budget = arg(5, "1000000").parse().unwrap();
    let switch_at = arg(6, "1.4").parse().unwrap();
    let force_fraction = arg(7, "0").parse::<f64>().unwrap();
    assert!((0. ..=1.).contains(&force_fraction));
    assert!(force_fraction == 0. || algorithm == "bfs-forced");
    assert!(n > 0 && alpha > 0. && alpha <= 1.);
    assert!(
        [
            "baseline",
            "oracle",
            "oracle-shuffled",
            "bfs-large",
            "bfs-forced",
            "bfs-priority-evict",
            "bfs-priority-evict-all",
            "section3",
            "section4",
            "section5"
        ]
        .contains(&algorithm.as_str())
    );
    println!(
        "algorithm,n,seed,bins,load,budget,switch,force_fraction,force_cutoff,status,inserted,primary,optimal_prediction,optimal_threshold,bumped_from_hist,secondary_in_hist,retained_hash_bits,quotient_bits,stored_entry_bits,packed_bits_per_key,negative_queries,false_positives,false_positive_ratio"
    );
    Experiment::new(
        n,
        alpha,
        seed,
        algorithm,
        budget,
        switch_at,
        force_fraction,
        filter_bits,
        filter_queries,
    )
    .run();
}

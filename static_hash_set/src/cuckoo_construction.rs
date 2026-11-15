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
    parent: Vec<usize>,
    parent_slot: Vec<usize>,
    reveals: usize,
    placed: usize,
    algorithm: String,
    budget: usize,
    switch_at: f64,
}
impl Experiment {
    fn new(
        n: usize,
        alpha: f64,
        seed: u64,
        algorithm: String,
        budget: usize,
        switch_at: f64,
    ) -> Self {
        let m = (n as f64 / alpha / 8.).ceil() as usize;
        let mut h = vec![[0; 2]; n + 1];
        let mut priority = vec![0; n + 1];
        for x in 1..=n {
            for side in 0..2 {
                h[x][side] =
                    ((mix(x as u64 ^ mix(seed + side as u64)) as u128 * m as u128) >> 64) as usize;
            }
            priority[x] = mix(x as u64 ^ mix(seed + 2));
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
            parent: vec![0; m],
            parent_slot: vec![0; m],
            reveals: 0,
            placed: 0,
            algorithm,
            budget,
            switch_at,
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
    // Exhaustive augmenting paths: failure certifies this graph cannot be placed.
    fn bfs(&mut self, x: u64) -> bool {
        let tag = x as usize;
        let mut q = Vec::new();
        for b in self.h[x as usize] {
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
                    self.bins[b][s] = self.bins[p][t];
                    s = t;
                    b = p;
                }
                self.bins[b][s] = x;
                return true;
            }
            for s in 0..8 {
                let y = self.bins[b][s] as usize;
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
    fn insert(&mut self, mut x: u64) -> bool {
        if self.algorithm == "oracle" {
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
        for x in 1..=self.n {
            if !self.insert(x as u64) {
                break;
            }
            inserted += 1;
        }
        print!(
            "{},{},{},{},{:.8},{},{:.2},",
            self.algorithm,
            self.n,
            self.seed,
            self.m,
            self.n as f64 / (8 * self.m) as f64,
            self.budget,
            self.switch_at
        );
        if inserted != self.n {
            println!("failed,{inserted},,,,,");
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
        let mut threshold = 0.;
        for (p, is_primary) in labels {
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
        println!(
            "ok,{inserted},{:.8},{:.8},{threshold:.8},{bumped_hist},{secondary_hist}",
            primary as f64 / self.n as f64,
            best as f64 / self.n as f64
        );
    }
}
fn main() {
    let args: Vec<_> = env::args().collect();
    let arg = |i, default: &str| args.get(i).cloned().unwrap_or(default.to_string());
    let n = arg(1, "1000000").parse().unwrap();
    let algorithm = arg(2, "baseline");
    let alpha = arg(3, "0.95").parse::<f64>().unwrap();
    let seed = arg(4, "1").parse().unwrap();
    let budget = arg(5, "1000000").parse().unwrap();
    let switch_at = arg(6, "1.4").parse().unwrap();
    assert!(n > 0 && alpha > 0. && alpha <= 1.);
    assert!(
        ["baseline", "oracle", "section3", "section4", "section5"].contains(&algorithm.as_str())
    );
    println!(
        "algorithm,n,seed,bins,load,budget,switch,status,inserted,primary,optimal_prediction,optimal_threshold,bumped_from_hist,secondary_in_hist"
    );
    Experiment::new(n, alpha, seed, algorithm, budget, switch_at).run();
}

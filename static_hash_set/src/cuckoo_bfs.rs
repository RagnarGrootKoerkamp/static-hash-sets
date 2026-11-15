/// Temporary construction workspace; never retained in a completed table.
/// The returned path runs from a free slot back to an insertion root.
pub(crate) struct BfsConstruction {
    generation: u32,
    visited: Vec<u32>,
    parent: Vec<(usize, usize)>,
    queue: Vec<usize>,
}
impl BfsConstruction {
    pub(crate) fn new(bins: usize) -> Self {
        Self {
            generation: 0,
            visited: vec![0; bins],
            parent: vec![(usize::MAX, 0); bins],
            queue: Vec::new(),
        }
    }
    pub(crate) fn path(
        &mut self,
        roots: &[usize],
        slots: usize,
        // (slot is empty, valid alternate bin for its occupant)
        mut inspect: impl FnMut(usize, usize) -> (bool, Option<usize>),
    ) -> Vec<(usize, usize)> {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.visited.fill(0);
            self.generation = 1;
        }
        self.queue.clear();
        for &b in roots {
            if self.visited[b] != self.generation {
                self.visited[b] = self.generation;
                self.parent[b] = (usize::MAX, 0);
                self.queue.push(b);
            }
        }
        let mut head = 0;
        while head < self.queue.len() {
            let b = self.queue[head];
            head += 1;
            for slot in 0..slots {
                let (empty, alternate) = inspect(b, slot);
                if empty {
                    let mut path = vec![(b, slot)];
                    let mut current = b;
                    while self.parent[current].0 != usize::MAX {
                        let parent = self.parent[current];
                        path.push(parent);
                        current = parent.0;
                    }
                    return path;
                }
                if let Some(next) = alternate {
                    if self.visited[next] != self.generation {
                        self.visited[next] = self.generation;
                        self.parent[next] = (b, slot);
                        self.queue.push(next);
                    }
                }
            }
        }
        panic!("BFS exhausted reachable buckets: no cuckoo placement exists");
    }
}

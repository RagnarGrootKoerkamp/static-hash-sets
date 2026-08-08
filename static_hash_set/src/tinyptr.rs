use crate::traits::HashSet;
use crate::T;
use tinyptr_sys::{
    tinyptr_blast_free, tinyptr_blast_insert, tinyptr_blast_new, tinyptr_blast_query,
    tinyptr_blast_t,
};

pub struct TinyPtr {
    inner: *mut tinyptr_blast_t,
    len: usize,
    capacity: usize,
}

unsafe impl Send for TinyPtr {}
unsafe impl Sync for TinyPtr {}

impl TinyPtr {
    pub fn new(keys: &[T]) -> Option<Self> {
        let size = keys.len().max(1024);
        let capacity = (size as f64 * 4.0) as usize;
        let inner = unsafe { tinyptr_blast_new(capacity as u64, 127, false) };
        if inner.is_null() {
            panic!("construction failed");
            return None;
        }
        for (j, &key) in keys.iter().enumerate() {
            if !unsafe { tinyptr_blast_insert(inner, key, 0) } {
                panic!("construction failed {j} / {key}");
                unsafe { tinyptr_blast_free(inner) };
                return None;
            }
        }
        eprintln!("construction passed");
        return Some(Self {
            inner,
            len: keys.len(),
            capacity,
        });
    }
}

impl Drop for TinyPtr {
    fn drop(&mut self) {
        unsafe { tinyptr_blast_free(self.inner) }
    }
}

impl HashSet for TinyPtr {
    fn name(&self) -> &'static str {
        "TinyPtr"
    }

    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(TinyPtr::new(keys).expect("TinyPtr build failed"))
    }

    fn allocation_size(&self) -> usize {
        0
    }

    fn load_factor(&self) -> f32 {
        self.len as f32 / self.capacity as f32
    }

    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        let mut value = 0;
        unsafe { tinyptr_blast_query(self.inner, key, &mut value) }
    }
}

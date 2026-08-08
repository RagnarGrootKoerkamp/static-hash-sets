use crate::traits::HashSet;
use crate::T;
use tinyptr_sys::{
    tinyptr_nonconc_blast_free, tinyptr_nonconc_blast_insert, tinyptr_nonconc_blast_new,
    tinyptr_nonconc_blast_query, tinyptr_nonconc_blast_t,
};

pub struct TinyPtr {
    inner: *mut tinyptr_nonconc_blast_t,
    len: usize,
    capacity: usize,
}

unsafe impl Send for TinyPtr {}
unsafe impl Sync for TinyPtr {}

impl TinyPtr {
    pub fn new(keys: &[T]) -> Option<Self> {
        let size = keys.len().max(1024);
        let capacity = (size as f64 * 2.0) as usize;
        for _ in 0..10 {
            let inner = unsafe { tinyptr_nonconc_blast_new(capacity as u64, 8, true) };
            if inner.is_null() {
                continue;
            }
            let mut ok = true;
            for &key in keys {
                if !unsafe { tinyptr_nonconc_blast_insert(inner, key, 0) } {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some(Self {
                    inner,
                    len: keys.len(),
                    capacity,
                });
            }
            unsafe { tinyptr_nonconc_blast_free(inner) };
        }
        None
    }
}

impl Drop for TinyPtr {
    fn drop(&mut self) {
        unsafe { tinyptr_nonconc_blast_free(self.inner) }
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
        unsafe { tinyptr_nonconc_blast_query(self.inner, key, &mut value) }
    }
}

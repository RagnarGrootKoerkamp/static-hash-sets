use crate::traits::HashSet;
use crate::T;
use tpht_sys::{tpht_get_u64, tpht_memory_bytes, tpht_put_u64, tpht_size, tpht_table_t};

pub struct Tpht {
    inner: *mut tpht_table_t,
}

unsafe impl Send for Tpht {}
unsafe impl Sync for Tpht {}

impl Tpht {
    pub fn new(keys: &[T]) -> Option<Self> {
        let inner = unsafe { tpht_sys::flatten_tpht_fixed_create(keys.len().max(1), 8, 2) };
        if inner.is_null() {
            return None;
        }
        for &key in keys {
            let status = unsafe { tpht_put_u64(inner, key, key) };
            if status != tpht_sys::tpht_status_t::TPHT_OK {
                unsafe { tpht_sys::tpht_destroy(inner) };
                return None;
            }
        }
        Some(Self { inner })
    }
}

impl Drop for Tpht {
    fn drop(&mut self) {
        unsafe { tpht_sys::tpht_destroy(self.inner) }
    }
}

impl HashSet for Tpht {
    fn name(&self) -> &'static str {
        "TPHT"
    }

    fn new(&self, keys: &[T]) -> Box<dyn HashSet> {
        Box::new(Tpht::new(keys).expect("TPHT build failed"))
    }

    fn allocation_size(&self) -> usize {
        unsafe { tpht_memory_bytes(self.inner) }
    }

    fn load_factor(&self) -> f32 {
        unsafe { tpht_size(self.inner) as f32 / tpht_sys::tpht_capacity(self.inner) as f32 }
    }

    #[inline(always)]
    fn contains(&self, key: T) -> bool {
        let mut value = 0;
        unsafe { tpht_get_u64(self.inner, key, &mut value) == tpht_sys::tpht_status_t::TPHT_OK }
    }
}

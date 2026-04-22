use std::ffi::c_void;

unsafe extern "C" {
    fn fph_dyn_set_new(keys: *const u64, len: usize, max_load_factor: f32) -> *mut c_void;
    fn fph_dyn_set_free(s: *mut c_void);
    fn fph_dyn_set_contains(s: *const c_void, key: u64) -> bool;
    fn fph_dyn_set_prefetch(s: *const c_void, key: u64) -> usize;
    fn fph_dyn_set_contains_with_token(key: u64, token: usize) -> bool;
    fn fph_dyn_set_slot_count(s: *const c_void) -> usize;

    fn fph_meta_set_new(keys: *const u64, len: usize, max_load_factor: f32) -> *mut c_void;
    fn fph_meta_set_free(s: *mut c_void);
    fn fph_meta_set_contains(s: *const c_void, key: u64) -> bool;
    fn fph_meta_set_slot_count(s: *const c_void) -> usize;
    fn fph_meta_set_elem_count(s: *const c_void) -> usize;
}

/// Rust wrapper around `fph::DynamicFphSet<uint64_t>`.
pub struct DynFphSet(*mut c_void);

// Safe because the set is immutable after construction and the C++ type has no thread-local state.
unsafe impl Send for DynFphSet {}
unsafe impl Sync for DynFphSet {}

impl DynFphSet {
    /// Build from a slice of unique keys. Returns `None` if the C++ build fails.
    pub fn new(keys: &[u64], max_load_factor: f32) -> Option<Self> {
        let ptr = unsafe { fph_dyn_set_new(keys.as_ptr(), keys.len(), max_load_factor) };
        if ptr.is_null() { None } else { Some(Self(ptr)) }
    }

    #[inline(always)]
    pub fn contains(&self, key: u64) -> bool {
        unsafe { fph_dyn_set_contains(self.0, key) }
    }

    /// Prefetches the slot that would hold `key` and returns an opaque token
    /// encoding the slot address. Pass the token to [`contains_with_token`]
    /// after enough other work to hide the cache-miss latency.
    ///
    /// [`contains_with_token`]: DynFphSet::contains_with_token
    #[inline(always)]
    pub fn prefetch(&self, key: u64) -> usize {
        unsafe { fph_dyn_set_prefetch(self.0, key) }
    }

    /// Returns `true` if `token` (returned by [`prefetch`] for `key`)
    /// identifies a slot containing `key`.
    ///
    /// # Safety
    /// `token` must have been returned by `self.prefetch(key)` on the
    /// **same** set instance. Passing a stale or foreign token is undefined
    /// behaviour.
    ///
    /// [`prefetch`]: DynFphSet::prefetch
    #[inline(always)]
    pub fn contains_with_token(&self, key: u64, token: usize) -> bool {
        unsafe { fph_dyn_set_contains_with_token(key, token) }
    }

    /// Number of allocated slots (lower bound on memory usage; does not include bucket params).
    pub fn slot_count(&self) -> usize {
        unsafe { fph_dyn_set_slot_count(self.0) }
    }
}

impl Drop for DynFphSet {
    fn drop(&mut self) {
        unsafe { fph_dyn_set_free(self.0) }
    }
}

/// Rust wrapper around `fph::MetaFphSet<uint64_t>`.
pub struct MetaFphSet(*mut c_void);

unsafe impl Send for MetaFphSet {}
unsafe impl Sync for MetaFphSet {}

impl MetaFphSet {
    /// Build from a slice of unique keys. Returns `None` if the C++ build fails.
    pub fn new(keys: &[u64], max_load_factor: f32) -> Option<Self> {
        let ptr = unsafe { fph_meta_set_new(keys.as_ptr(), keys.len(), max_load_factor) };
        if ptr.is_null() { None } else { Some(Self(ptr)) }
    }

    #[inline(always)]
    pub fn contains(&self, key: u64) -> bool {
        unsafe { fph_meta_set_contains(self.0, key) }
    }

    /// Number of allocated slots.
    pub fn slot_count(&self) -> usize {
        unsafe { fph_meta_set_slot_count(self.0) }
    }

    /// Number of stored elements (used to estimate metadata overhead: 1 byte/element).
    pub fn elem_count(&self) -> usize {
        unsafe { fph_meta_set_elem_count(self.0) }
    }
}

impl Drop for MetaFphSet {
    fn drop(&mut self) {
        unsafe { fph_meta_set_free(self.0) }
    }
}

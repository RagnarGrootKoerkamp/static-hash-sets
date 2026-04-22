use std::ffi::c_void;

unsafe extern "C" {
    fn mapembed_new(keys: *const u64, len: usize) -> *mut c_void;
    fn mapembed_free(s: *mut c_void);
    fn mapembed_contains(s: *const c_void, key: u64) -> bool;
    fn mapembed_bucket_number(s: *const c_void) -> usize;
    fn mapembed_allocation_size(s: *const c_void) -> usize;
}

/// Rust wrapper around `MapEmbed` configured as a set of `u64` keys
/// (KEY_LEN=8, VAL_LEN=0).
pub struct MapEmbedSet(*mut c_void);

// Safe: MapEmbed is immutable after construction and has no thread-local state.
unsafe impl Send for MapEmbedSet {}
unsafe impl Sync for MapEmbedSet {}

impl MapEmbedSet {
    /// Build from a slice of unique keys. Returns `None` if construction fails.
    pub fn new(keys: &[u64]) -> Option<Self> {
        let ptr = unsafe { mapembed_new(keys.as_ptr(), keys.len()) };
        if ptr.is_null() { None } else { Some(Self(ptr)) }
    }

    #[inline(always)]
    pub fn contains(&self, key: u64) -> bool {
        unsafe { mapembed_contains(self.0, key) }
    }

    /// Number of buckets (each holds up to N=16 keys).
    pub fn bucket_number(&self) -> usize {
        unsafe { mapembed_bucket_number(self.0) }
    }

    /// Total heap bytes used by buckets and cell arrays.
    pub fn allocation_size(&self) -> usize {
        unsafe { mapembed_allocation_size(self.0) }
    }
}

impl Drop for MapEmbedSet {
    fn drop(&mut self) {
        unsafe { mapembed_free(self.0) }
    }
}

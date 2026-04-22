use std::ffi::c_void;
use std::ptr::NonNull;

unsafe extern "C" {
    // ThresholdBasedBumping<8, 5, false>
    fn tbb85_new(keys: *const u64, n: usize, overload: f64) -> *mut c_void;
    fn tbb85_free(set: *mut c_void);
    fn tbb85_query(set: *const c_void, key: u64) -> u64;
    fn tbb85_count_bits(set: *const c_void) -> usize;

    // ThresholdBasedBumping<8, 4, true>
    fn tbb84p_new(keys: *const u64, n: usize, overload: f64) -> *mut c_void;
    fn tbb84p_free(set: *mut c_void);
    fn tbb84p_query(set: *const c_void, key: u64) -> u64;
    fn tbb84p_count_bits(set: *const c_void) -> usize;

    // HashDisplace<8, OptimalBucketFunction<8>, CompactEncoding>
    fn hd8_new(keys: *const u64, n: usize, bucket_size: u64) -> *mut c_void;
    fn hd8_free(set: *mut c_void);
    fn hd8_query(set: *const c_void, key: u64) -> u64;
    fn hd8_count_bits(set: *const c_void) -> usize;
}

pub struct Tbb85Set(NonNull<c_void>);
pub struct Tbb84pSet(NonNull<c_void>);
pub struct Hd8Set(NonNull<c_void>);

unsafe impl Send for Tbb85Set {}
unsafe impl Sync for Tbb85Set {}
unsafe impl Send for Tbb84pSet {}
unsafe impl Sync for Tbb84pSet {}
unsafe impl Send for Hd8Set {}
unsafe impl Sync for Hd8Set {}

impl Tbb85Set {
    pub fn new(keys: &[u64], overload: f64) -> Self {
        let ptr = unsafe { tbb85_new(keys.as_ptr(), keys.len(), overload) };
        Self(NonNull::new(ptr).expect("tbb85_new returned null"))
    }

    pub fn query(&self, key: u64) -> u64 {
        unsafe { tbb85_query(self.0.as_ptr(), key) }
    }

    pub fn count_bits(&self) -> usize {
        unsafe { tbb85_count_bits(self.0.as_ptr()) }
    }
}

impl Drop for Tbb85Set {
    fn drop(&mut self) {
        unsafe { tbb85_free(self.0.as_ptr()) }
    }
}

impl Tbb84pSet {
    pub fn new(keys: &[u64], overload: f64) -> Self {
        let ptr = unsafe { tbb84p_new(keys.as_ptr(), keys.len(), overload) };
        Self(NonNull::new(ptr).expect("tbb84p_new returned null"))
    }

    pub fn query(&self, key: u64) -> u64 {
        unsafe { tbb84p_query(self.0.as_ptr(), key) }
    }

    pub fn count_bits(&self) -> usize {
        unsafe { tbb84p_count_bits(self.0.as_ptr()) }
    }
}

impl Drop for Tbb84pSet {
    fn drop(&mut self) {
        unsafe { tbb84p_free(self.0.as_ptr()) }
    }
}

impl Hd8Set {
    pub fn new(keys: &[u64], bucket_size: u64) -> Self {
        let ptr = unsafe { hd8_new(keys.as_ptr(), keys.len(), bucket_size) };
        Self(NonNull::new(ptr).expect("hd8_new returned null"))
    }

    pub fn query(&self, key: u64) -> u64 {
        unsafe { hd8_query(self.0.as_ptr(), key) }
    }

    pub fn count_bits(&self) -> usize {
        unsafe { hd8_count_bits(self.0.as_ptr()) }
    }
}

impl Drop for Hd8Set {
    fn drop(&mut self) {
        unsafe { hd8_free(self.0.as_ptr()) }
    }
}

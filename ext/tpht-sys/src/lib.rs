#![allow(non_camel_case_types)]

use std::ffi::c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum tpht_variant_t {
    TPHT_CHAINED = 1,
    TPHT_FLATTEN = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum tpht_threading_t {
    TPHT_SEQUENTIAL = 0,
    TPHT_CONCURRENT = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum tpht_resize_mode_t {
    TPHT_FIXED = 0,
    TPHT_RESIZABLE = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum tpht_status_t {
    TPHT_OK = 0,
    TPHT_NOT_FOUND = 1,
    TPHT_EXISTS = 2,
    TPHT_FULL = 3,
    TPHT_NO_MEMORY = 4,
    TPHT_INVALID = 5,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tpht_config_t {
    pub variant: tpht_variant_t,
    pub threading: tpht_threading_t,
    pub resize_mode: tpht_resize_mode_t,
    pub initial_capacity: usize,
    pub key_size: u8,
    pub value_size: u8,
    pub bin_size: u8,
    pub max_load_factor: f64,
    pub hash_seed: u64,
    pub resize_strides: usize,
}

#[repr(C)]
pub struct tpht_table_t {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn tpht_default_config() -> tpht_config_t;
    pub fn tpht_create(config: *const tpht_config_t) -> *mut tpht_table_t;
    pub fn tpht_destroy(table: *mut tpht_table_t);
    pub fn tpht_put(table: *mut tpht_table_t, key: *const c_void, value: *const c_void) -> tpht_status_t;
    pub fn tpht_insert(table: *mut tpht_table_t, key: *const c_void, value: *const c_void) -> tpht_status_t;
    pub fn tpht_update(table: *mut tpht_table_t, key: *const c_void, value: *const c_void) -> tpht_status_t;
    pub fn tpht_get(table: *mut tpht_table_t, key: *const c_void, value_out: *mut c_void) -> tpht_status_t;
    pub fn tpht_remove(table: *mut tpht_table_t, key: *const c_void) -> tpht_status_t;
    pub fn tpht_size(table: *const tpht_table_t) -> usize;
    pub fn tpht_capacity(table: *const tpht_table_t) -> usize;
    pub fn tpht_memory_bytes(table: *const tpht_table_t) -> usize;
    pub fn tpht_get_variant(table: *const tpht_table_t) -> tpht_variant_t;
    pub fn tpht_get_threading(table: *const tpht_table_t) -> tpht_threading_t;
    pub fn tpht_get_resize_mode(table: *const tpht_table_t) -> tpht_resize_mode_t;
    pub fn chained_tpht_fixed_create(capacity: usize, key_size: u8, value_size: u8) -> *mut tpht_table_t;
    pub fn chained_tpht_resizable_create(capacity: usize, key_size: u8, value_size: u8) -> *mut tpht_table_t;
    pub fn chained_tpht_concurrent_fixed_create(capacity: usize, key_size: u8, value_size: u8) -> *mut tpht_table_t;
    pub fn chained_tpht_concurrent_resizable_create(capacity: usize, key_size: u8, value_size: u8) -> *mut tpht_table_t;
    pub fn flatten_tpht_fixed_create(capacity: usize, key_size: u8, value_size: u8) -> *mut tpht_table_t;
    pub fn flatten_tpht_resizable_create(capacity: usize, key_size: u8, value_size: u8) -> *mut tpht_table_t;
    pub fn flatten_tpht_concurrent_fixed_create(capacity: usize, key_size: u8, value_size: u8) -> *mut tpht_table_t;
    pub fn flatten_tpht_concurrent_resizable_create(capacity: usize, key_size: u8, value_size: u8) -> *mut tpht_table_t;
    pub fn tpht_put_u64(table: *mut tpht_table_t, key: u64, value: u64) -> tpht_status_t;
    pub fn tpht_get_u64(table: *mut tpht_table_t, key: u64, value_out: *mut u64) -> tpht_status_t;
    pub fn tpht_remove_u64(table: *mut tpht_table_t, key: u64) -> tpht_status_t;
}

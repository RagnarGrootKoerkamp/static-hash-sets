#![allow(non_camel_case_types)]

#[repr(C)]
pub struct tinyptr_nonconc_blast_t {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn tinyptr_nonconc_blast_new(
        size: u64,
        bin_size: u16,
        if_resize: bool,
    ) -> *mut tinyptr_nonconc_blast_t;
    pub fn tinyptr_nonconc_blast_free(table: *mut tinyptr_nonconc_blast_t);
    pub fn tinyptr_nonconc_blast_insert(
        table: *mut tinyptr_nonconc_blast_t,
        key: u64,
        value: u64,
    ) -> bool;
    pub fn tinyptr_nonconc_blast_query(
        table: *const tinyptr_nonconc_blast_t,
        key: u64,
        value_out: *mut u64,
    ) -> bool;
    pub fn tinyptr_nonconc_blast_update(
        table: *mut tinyptr_nonconc_blast_t,
        key: u64,
        value: u64,
    ) -> bool;
    pub fn tinyptr_nonconc_blast_remove(table: *mut tinyptr_nonconc_blast_t, key: u64);
}

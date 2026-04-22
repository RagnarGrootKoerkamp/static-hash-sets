#include "fph/dynamic_fph_table.h"
#include "fph/meta_fph_table.h"

#include <cstdint>
#include <cstddef>

using DynSet = fph::DynamicFphSet<uint64_t, fph::SimpleSeedHash<uint64_t>>;
using MetaSet = fph::MetaFphSet<uint64_t, fph::meta::SimpleSeedHash<uint64_t>>;

extern "C" {

// DynamicFphSet

DynSet* fph_dyn_set_new(const uint64_t* keys, size_t len, float max_load_factor) {
    try {
        auto* s = new DynSet();
        s->max_load_factor(max_load_factor);
        s->Build<false, false>(keys, keys + len);
        return s;
    } catch (...) {
        return nullptr;
    }
}

void fph_dyn_set_free(DynSet* s) {
    delete s;
}

bool fph_dyn_set_contains(const DynSet* s, uint64_t key) {
    return s->count(key) > 0;
}

// Prefetches the slot for `key` and returns its address as an opaque token
// (to be passed to fph_dyn_set_contains_with_token after the prefetch latency).
uintptr_t fph_dyn_set_prefetch(const DynSet* s, uint64_t key) {
    const uint64_t* ptr = s->GetPointerNoCheck(key);
    __builtin_prefetch(ptr, 0, 3);
    return reinterpret_cast<uintptr_t>(ptr);
}

// Returns true if the slot identified by `token` holds `key`.
// `token` must be a value previously returned by fph_dyn_set_prefetch
// for the same set instance.
bool fph_dyn_set_contains_with_token(uint64_t key, uintptr_t token) {
    return *reinterpret_cast<const uint64_t*>(token) == key;
}

// Slots * sizeof(u64); buckets add ~c*n/(log2(n)+1)*4 bytes but are not directly accessible.
size_t fph_dyn_set_slot_count(const DynSet* s) {
    return s->bucket_count();
}

// MetaFphSet

MetaSet* fph_meta_set_new(const uint64_t* keys, size_t len, float max_load_factor) {
    try {
        auto* s = new MetaSet();
        s->max_load_factor(max_load_factor);
        s->Build<false, false>(keys, keys + len);
        return s;
    } catch (...) {
        return nullptr;
    }
}

void fph_meta_set_free(MetaSet* s) {
    delete s;
}

bool fph_meta_set_contains(const MetaSet* s, uint64_t key) {
    return s->count(key) > 0;
}

size_t fph_meta_set_slot_count(const MetaSet* s) {
    return s->bucket_count();
}

size_t fph_meta_set_elem_count(const MetaSet* s) {
    return s->size();
}

} // extern "C"

#include "nonconc_blast_ht.h"

extern "C" {

tinyptr::NonConcBlastHT* tinyptr_nonconc_blast_new(uint64_t size,
                                                    uint16_t bin_size,
                                                    bool if_resize) {
    return new tinyptr::NonConcBlastHT(size, bin_size, if_resize);
}

void tinyptr_nonconc_blast_free(tinyptr::NonConcBlastHT* table) {
    delete table;
}

bool tinyptr_nonconc_blast_insert(tinyptr::NonConcBlastHT* table, uint64_t key,
                                  uint64_t value) {
    return table->Insert(key, value);
}

bool tinyptr_nonconc_blast_query(const tinyptr::NonConcBlastHT* table,
                                 uint64_t key, uint64_t* value_out) {
    return const_cast<tinyptr::NonConcBlastHT*>(table)->Query(key, value_out);
}

bool tinyptr_nonconc_blast_update(tinyptr::NonConcBlastHT* table, uint64_t key,
                                  uint64_t value) {
    return table->Update(key, value);
}

void tinyptr_nonconc_blast_remove(tinyptr::NonConcBlastHT* table, uint64_t key) {
    table->Free(key);
}

}

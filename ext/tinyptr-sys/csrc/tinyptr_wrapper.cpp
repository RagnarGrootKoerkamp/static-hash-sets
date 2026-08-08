#include "blast_ht.h"

extern "C" {

struct tinyptr_blast {
    tinyptr::BlastHT* table;
};

tinyptr_blast* tinyptr_blast_new(uint64_t size, uint16_t bin_size,
                                  bool if_resize) {
    auto* result = new tinyptr_blast;
    result->table = new tinyptr::BlastHT(size, bin_size, if_resize);
    return result;
}

void tinyptr_blast_free(tinyptr_blast* table) {
    delete table->table;
    delete table;
}

bool tinyptr_blast_insert(tinyptr_blast* table, uint64_t key, uint64_t value) {
    return table->table->Insert(key, value);
}

bool tinyptr_blast_query(const tinyptr_blast* table, uint64_t key,
                         uint64_t* value_out) {
    return table->table->Query(key, value_out);
}

bool tinyptr_blast_update(tinyptr_blast* table, uint64_t key, uint64_t value) {
    return table->table->Update(key, value);
}

void tinyptr_blast_remove(tinyptr_blast* table, uint64_t key) {
    table->table->Free(key);
}

}

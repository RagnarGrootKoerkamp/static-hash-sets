// C wrapper around MapEmbed for use as a set of uint64_t keys.
// We override KEY_LEN/VAL_LEN/KV_NUM before including the header.
#define KEY_LEN 8
#define VAL_LEN 0
#define N 8
#define USING_SIMD 1

#include "MapEmbed.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

// Build a MapEmbed set from an array of unique uint64_t keys.
// Returns nullptr if construction fails after several attempts.
MapEmbed* mapembed_new(const uint64_t* keys, size_t n) {
	const int layer     = 3;
	const int cell_bit  = 4;
	const int cell_hash = 1 << cell_bit; // 16

	// N=8 slots per bucket; target ~90% fill to leave headroom.
	int min_buckets = (n == 0) ? 1 : (int)std::ceil((double)n / (8.0 * 0.9));
	// bucket_number must be a multiple of cell_hash (required by extend()).
	int bucket_number = ((min_buckets + cell_hash - 1) / cell_hash) * cell_hash;

	int cell_number[3];
	cell_number[0] = bucket_number * 9 / 2;
	cell_number[1] = bucket_number * 3 / 2;
	cell_number[2] = bucket_number / 2;

	// Retry a few times; each MapEmbed construction picks fresh random seeds.
	for(int attempt = 0; attempt < 5; ++attempt) {
		MapEmbed* s = new MapEmbed(layer, bucket_number, cell_number, cell_bit);
		bool ok     = true;
		KV_entry kv = {};
		for(size_t i = 0; i < n; ++i) {
			std::memcpy(kv.key, &keys[i], KEY_LEN);
			if(!s->insert(kv)) {
				ok = false;
				break;
			}
		}
		if(ok) return s;
		delete s;
	}
	return nullptr;
}

void mapembed_free(MapEmbed* s) {
	delete s;
}

bool mapembed_contains(const MapEmbed* s, uint64_t key) {
	return const_cast<MapEmbed*>(s)->query((const char*)&key);
}

size_t mapembed_bucket_number(const MapEmbed* s) {
	return (size_t)s->bucket_number;
}

size_t mapembed_allocation_size(const MapEmbed* s) {
	size_t bucket_mem = (size_t)s->bucket_number * sizeof(Bucket);
	size_t cell_mem   = 0;
	for(int i = 0; i < s->cell_layer; ++i)
		cell_mem += (size_t)(s->cell_number[i] + 10) * sizeof(uint32_t);
	return bucket_mem + cell_mem;
}

} // extern "C"

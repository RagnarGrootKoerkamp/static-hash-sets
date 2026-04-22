#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

#include <hash128.hpp>
#include <ThresholdBasedBumping.hpp>
#include <HashDisplace.hpp>
#include <OptimalBucketFunction.hpp>
#include <CompactEncoding.hpp>

using TBB85 = kphf::ThresholdBasedBumping::ThresholdBasedBumping<8, 5, false>;
using TBB84P = kphf::ThresholdBasedBumping::ThresholdBasedBumping<8, 4, true>;
using HD8 = kphf::HashDisplace::HashDisplace<8, kphf::HashDisplace::OptimalBucketFunction<8>, kphf::HashDisplace::CompactEncoding>;

static inline Hash128 key_to_hash128(uint64_t key) {
    std::string s(reinterpret_cast<const char *>(&key), sizeof(key));
    return Hash128(s);
}

static inline std::vector<Hash128> keys_to_hash128(const uint64_t *keys, size_t n) {
    std::vector<Hash128> hashes;
    hashes.reserve(n);
    for (size_t i = 0; i < n; i++) {
        hashes.emplace_back(key_to_hash128(keys[i]));
    }
    return hashes;
}

static inline std::vector<std::string> keys_to_strings(const uint64_t *keys, size_t n) {
    std::vector<std::string> strings;
    strings.reserve(n);
    for (size_t i = 0; i < n; i++) {
        strings.emplace_back(reinterpret_cast<const char *>(&keys[i]), sizeof(uint64_t));
    }
    return strings;
}

extern "C" {

// --- ThresholdBasedBumping<8, 5, false> ---

void *tbb85_new(const uint64_t *keys, size_t n, double overload) {
    auto hashes = keys_to_hash128(keys, n);
    return new TBB85(std::move(hashes), overload);
}

void tbb85_free(void *set) {
    delete static_cast<TBB85 *>(set);
}

uint64_t tbb85_query(const void *set, uint64_t key) {
    return (*static_cast<const TBB85 *>(set))(key_to_hash128(key));
}

size_t tbb85_count_bits(const void *set) {
    return static_cast<const TBB85 *>(set)->count_bits();
}

// --- ThresholdBasedBumping<8, 4, true> ---

void *tbb84p_new(const uint64_t *keys, size_t n, double overload) {
    auto hashes = keys_to_hash128(keys, n);
    return new TBB84P(std::move(hashes), overload);
}

void tbb84p_free(void *set) {
    delete static_cast<TBB84P *>(set);
}

uint64_t tbb84p_query(const void *set, uint64_t key) {
    return (*static_cast<const TBB84P *>(set))(key_to_hash128(key));
}

size_t tbb84p_count_bits(const void *set) {
    return static_cast<const TBB84P *>(set)->count_bits();
}

// --- HashDisplace<8, OptimalBucketFunction<8>, CompactEncoding> ---

void *hd8_new(const uint64_t *keys, size_t n, uint64_t bucket_size) {
    auto strings = keys_to_strings(keys, n);
    return new HD8(strings, bucket_size);
}

void hd8_free(void *set) {
    delete static_cast<HD8 *>(set);
}

uint64_t hd8_query(const void *set, uint64_t key) {
    std::string s(reinterpret_cast<const char *>(&key), sizeof(key));
    return (*static_cast<const HD8 *>(set))(s);
}

size_t hd8_count_bits(const void *set) {
    return static_cast<const HD8 *>(set)->count_bits();
}

} // extern "C"

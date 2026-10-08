#include "../../include/Ciot.h"

#include <cmath>
#include <cstring>

namespace ciot {

bool kv_cache_init(KVCache* cache, std::uint32_t max_tokens, std::uint32_t dim) {
    if (!cache || max_tokens == 0 || dim == 0) {
        return false;
    }

    kv_cache_free(cache);
    cache->key = static_cast<float*>(aligned_malloc(static_cast<std::size_t>(max_tokens) * dim * sizeof(float)));
    cache->value = static_cast<float*>(aligned_malloc(static_cast<std::size_t>(max_tokens) * dim * sizeof(float)));
    if (!cache->key || !cache->value) {
        kv_cache_free(cache);
        return false;
    }

    cache->max_tokens = max_tokens;
    cache->dim = dim;
    cache->used = 0;
    std::memset(cache->key, 0, static_cast<std::size_t>(max_tokens) * dim * sizeof(float));
    std::memset(cache->value, 0, static_cast<std::size_t>(max_tokens) * dim * sizeof(float));
    return true;
}

void kv_cache_free(KVCache* cache) {
    if (!cache) {
        return;
    }
    aligned_free(cache->key);
    aligned_free(cache->value);
    cache->max_tokens = 0;
    cache->dim = 0;
    cache->used = 0;
    cache->key = nullptr;
    cache->value = nullptr;
}

void kv_cache_reset(KVCache* cache) {
    if (!cache) {
        return;
    }
    cache->used = 0;
}

void kv_cache_append(KVCache* cache, const float* key, const float* value) {
    if (!cache || !key || !value || !cache->key || !cache->value || cache->used >= cache->max_tokens) {
        return;
    }

    const std::size_t offset = static_cast<std::size_t>(cache->used) * cache->dim;
    std::memcpy(cache->key + offset, key, static_cast<std::size_t>(cache->dim) * sizeof(float));
    std::memcpy(cache->value + offset, value, static_cast<std::size_t>(cache->dim) * sizeof(float));
    cache->used += 1U;
}

void attention_decode_1head(const KVCache* cache,
                            const float* query,
                            float* out,
                            float* score_scratch,
                            std::uint32_t dim) {
    if (!cache || !query || !out || !score_scratch || !cache->key || !cache->value || cache->used == 0 || dim == 0 || dim != cache->dim) {
        return;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (std::uint32_t t = 0; t < cache->used; ++t) {
        const float* k = cache->key + (static_cast<std::size_t>(t) * dim);
        float dot = 0.0f;
        for (std::uint32_t i = 0; i < dim; ++i) {
            dot += query[i] * k[i];
        }
        score_scratch[t] = dot * scale;
    }

    softmax_inplace(score_scratch, cache->used);

    for (std::uint32_t i = 0; i < dim; ++i) {
        out[i] = 0.0f;
    }

    for (std::uint32_t t = 0; t < cache->used; ++t) {
        const float score = score_scratch[t];
        const float* v = cache->value + (static_cast<std::size_t>(t) * dim);
        for (std::uint32_t i = 0; i < dim; ++i) {
            out[i] += score * v[i];
        }
    }
}

} // namespace ciot

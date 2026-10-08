#include "../../include/Ciot.h"

#include <cmath>
#include <cstring>

namespace ciot {

bool mha_kv_cache_init(MhaKVCache* cache, std::uint32_t max_tokens,
                       std::uint32_t num_heads, std::uint32_t head_dim) {
    if (!cache || max_tokens == 0 || num_heads == 0 || head_dim == 0) {
        return false;
    }
    mha_kv_cache_free(cache);
    const std::size_t bytes = 2ULL * num_heads * max_tokens * head_dim * sizeof(float);
    cache->buffer = static_cast<float*>(aligned_malloc(bytes));
    if (!cache->buffer) {
        return false;
    }
    std::memset(cache->buffer, 0, bytes);
    cache->max_tokens = max_tokens;
    cache->num_heads = num_heads;
    cache->head_dim = head_dim;
    cache->used = 0;
    return true;
}

void mha_kv_cache_free(MhaKVCache* cache) {
    if (!cache) return;
    aligned_free(cache->buffer);
    cache->max_tokens = 0;
    cache->num_heads = 0;
    cache->head_dim = 0;
    cache->used = 0;
    cache->buffer = nullptr;
}

void mha_kv_cache_reset(MhaKVCache* cache) {
    if (!cache) return;
    cache->used = 0;
}

void mha_kv_cache_append(MhaKVCache* cache, const float* keys, const float* values) {
    if (!cache || !keys || !values || !cache->buffer || cache->used >= cache->max_tokens) return;

    const std::uint32_t head_dim = cache->head_dim;
    const std::uint32_t num_heads = cache->num_heads;
    const std::size_t token_offset = static_cast<std::size_t>(cache->used) * head_dim;

    for (std::uint32_t h = 0; h < num_heads; ++h) {
        const std::size_t head_base = 2ULL * h * cache->max_tokens * head_dim;
        const float* k_src = keys + (static_cast<std::size_t>(h) * head_dim);
        const float* v_src = values + (static_cast<std::size_t>(h) * head_dim);
        std::memcpy(cache->buffer + head_base + token_offset, k_src, head_dim * sizeof(float));
        std::memcpy(cache->buffer + head_base + cache->max_tokens * head_dim + token_offset, v_src, head_dim * sizeof(float));
    }
    cache->used += 1U;
}

void mha_attention_decode(const MhaKVCache* cache, const float* query,
                          float* out, float* score_scratch,
                          std::uint32_t num_heads, std::uint32_t head_dim) {
    if (!cache || !query || !out || !score_scratch || !cache->buffer || cache->used == 0) return;

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::uint32_t used = cache->used;

    for (std::uint32_t h = 0; h < num_heads; ++h) {
        const std::size_t head_base = 2ULL * h * cache->max_tokens * head_dim;
        const float* keys = cache->buffer + head_base;
        const float* values = cache->buffer + head_base + cache->max_tokens * head_dim;
        const float* q = query + (static_cast<std::size_t>(h) * head_dim);

        float* scores = score_scratch + (static_cast<std::size_t>(h) * cache->max_tokens);
        for (std::uint32_t t = 0; t < used; ++t) {
            const float* k = keys + (static_cast<std::size_t>(t) * head_dim);
            float dot = 0.0f;
            for (std::uint32_t i = 0; i < head_dim; ++i) dot += q[i] * k[i];
            scores[t] = dot * scale;
        }
        softmax_inplace(scores, used);

        float* attn_out = out + (static_cast<std::size_t>(h) * head_dim);
        for (std::uint32_t i = 0; i < head_dim; ++i) attn_out[i] = 0.0f;
        for (std::uint32_t t = 0; t < used; ++t) {
            const float s = scores[t];
            const float* v = values + (static_cast<std::size_t>(t) * head_dim);
            for (std::uint32_t i = 0; i < head_dim; ++i) attn_out[i] += s * v[i];
        }
    }
}

} // namespace ciot

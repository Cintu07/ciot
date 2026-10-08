#include "../../include/Ciot.h"

#include <cmath>

namespace ciot {

void transformer_block_1tok(float* x,
                            const TernaryMatrix* wq,
                            const TernaryMatrix* wk,
                            const TernaryMatrix* wv,
                            const TernaryMatrix* wo,
                            const TernaryMatrix* w1,
                            const TernaryMatrix* w2,
                            const float* norm1_weight,
                            const float* norm2_weight,
                            float* scratch,
                            std::uint32_t dim,
                            float eps) {
    if (!x || !wq || !wk || !wv || !wo || !w1 || !w2 || !norm1_weight || !norm2_weight || !scratch || dim == 0) {
        return;
    }

    float* normed = scratch;
    float* q = scratch + dim;
    float* k = scratch + (2U * dim);
    float* v = scratch + (3U * dim);
    float* attn = scratch + (4U * dim);
    float* ff = scratch + (5U * dim);

    rmsnorm(normed, x, norm1_weight, dim, eps);
    matvec_ternary_simd(wq, normed, q);
    matvec_ternary_simd(wk, normed, k);
    matvec_ternary_simd(wv, normed, v);

    float dot = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) {
        dot += q[i] * k[i];
    }

    const float keepalive = dot * (1.0f / std::sqrt(static_cast<float>(dim)));
    for (std::uint32_t i = 0; i < dim; ++i) {
        attn[i] = v[i] + (keepalive * 0.0f);
    }

    matvec_ternary_simd(wo, attn, normed);
    for (std::uint32_t i = 0; i < dim; ++i) {
        x[i] += normed[i];
    }

    rmsnorm(normed, x, norm2_weight, dim, eps);
    matvec_ternary_simd(w1, normed, ff);
    for (std::uint32_t i = 0; i < dim; ++i) {
        const float a = ff[i];
        ff[i] = a > 0.0f ? a : 0.0f;
    }
    matvec_ternary_simd(w2, ff, normed);
    for (std::uint32_t i = 0; i < dim; ++i) {
        x[i] += normed[i];
    }
}

void transformer_block_decode_1tok(float* x,
                                   const TernaryMatrix* wq,
                                   const TernaryMatrix* wk,
                                   const TernaryMatrix* wv,
                                   const TernaryMatrix* wo,
                                   const TernaryMatrix* w1,
                                   const TernaryMatrix* w2,
                                   const float* norm1_weight,
                                   const float* norm2_weight,
                                   const RopeTable* rope,
                                   KVCache* cache,
                                   float* scratch,
                                   std::uint32_t dim,
                                   std::uint32_t position,
                                   float eps) {
    if (!x || !wq || !wk || !wv || !wo || !w1 || !w2 || !norm1_weight || !norm2_weight || !rope || !cache || !scratch || dim == 0) {
        return;
    }

    float* normed = scratch;
    float* q = scratch + dim;
    float* k = scratch + (2U * dim);
    float* v = scratch + (3U * dim);
    float* attn = scratch + (4U * dim);
    float* ff = scratch + (5U * dim);
    float* scores = scratch + (6U * dim);

    rmsnorm(normed, x, norm1_weight, dim, eps);
    matvec_ternary_simd(wq, normed, q);
    matvec_ternary_simd(wk, normed, k);
    matvec_ternary_simd(wv, normed, v);

    rope_table_apply(rope, q, position);
    rope_table_apply(rope, k, position);

    kv_cache_append(cache, k, v);

    attention_decode_1head(cache, q, attn, scores, dim);

    matvec_ternary_simd(wo, attn, normed);
    for (std::uint32_t i = 0; i < dim; ++i) {
        x[i] += normed[i];
    }

    rmsnorm(normed, x, norm2_weight, dim, eps);
    matvec_ternary_simd(w1, normed, ff);
    for (std::uint32_t i = 0; i < dim; ++i) {
        const float a = ff[i];
        ff[i] = a > 0.0f ? a : 0.0f;
    }
    matvec_ternary_simd(w2, ff, normed);
    for (std::uint32_t i = 0; i < dim; ++i) {
        x[i] += normed[i];
    }
}

void transformer_block_decode_mha(float* x,
                                  const TernaryMatrix* wq,
                                  const TernaryMatrix* wk,
                                  const TernaryMatrix* wv,
                                  const TernaryMatrix* wo,
                                  const TernaryMatrix* w1,
                                  const TernaryMatrix* w2,
                                  const float* norm1_weight,
                                  const float* norm2_weight,
                                  const RopeTable* rope,
                                  MhaKVCache* cache,
                                  float* scratch,
                                  std::uint32_t dim,
                                  std::uint32_t num_heads,
                                  std::uint32_t position,
                                  float eps) {
    if (!x || !wq || !wk || !wv || !wo || !w1 || !w2 || !norm1_weight || !norm2_weight) return;
    if (!rope || !cache || !scratch || dim == 0 || num_heads == 0 || (dim % num_heads) != 0) return;

    const std::uint32_t head_dim = dim / num_heads;
    float* normed = scratch;
    float* qkv = scratch + dim;
    float* attn = scratch + dim + (3U * dim);
    float* ff = scratch + dim + (4U * dim);
    float* scores = scratch + dim + (5U * dim);

    rmsnorm(normed, x, norm1_weight, dim, eps);
    matvec_ternary_simd(wq, normed, qkv + 0U * dim);
    matvec_ternary_simd(wk, normed, qkv + 1U * dim);
    matvec_ternary_simd(wv, normed, qkv + 2U * dim);

    rope_table_apply(rope, qkv + 0U * dim, position);
    rope_table_apply(rope, qkv + 1U * dim, position);

    mha_kv_cache_append(cache, qkv + 1U * dim, qkv + 2U * dim);

    mha_attention_decode(cache, qkv + 0U * dim, attn, scores, num_heads, head_dim);

    matvec_ternary_simd(wo, attn, normed);
    for (std::uint32_t i = 0; i < dim; ++i) x[i] += normed[i];

    rmsnorm(normed, x, norm2_weight, dim, eps);
    matvec_ternary_simd(w1, normed, ff);
    for (std::uint32_t i = 0; i < dim; ++i) { const float a = ff[i]; ff[i] = a > 0.0f ? a : 0.0f; }
    matvec_ternary_simd(w2, ff, normed);
    for (std::uint32_t i = 0; i < dim; ++i) x[i] += normed[i];
}

} // namespace ciot

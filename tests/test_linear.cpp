#include "../include/Ciot.h"

#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

std::int8_t pattern(std::uint32_t r, std::uint32_t c) {
    const std::uint32_t v = (r * 17U + c * 31U + (r ^ c)) % 9U;
    const std::uint32_t pos = v == 0U || v == 5U;
    const std::uint32_t neg = v == 1U || v == 6U;
    return static_cast<std::int8_t>(static_cast<int>(pos) - static_cast<int>(neg));
}

bool close_enough(float a, float b) {
    return std::fabs(a - b) <= 0.0001f;
}

} // namespace

int main() {
    constexpr std::uint32_t rows = 37;
    constexpr std::uint32_t cols = 130;

    ciot::TernaryMatrix matrix;
    if (!ciot::matrix_init(&matrix, rows, cols)) {
        std::cerr << "matrix allocation failed\n";
        return 1;
    }

    for (std::uint32_t r = 0; r < rows; ++r) {
        matrix.row_scale[r] = 0.25f + static_cast<float>(r % 5U) * 0.125f;
        for (std::uint32_t c = 0; c < cols; ++c) {
            const std::int8_t w = pattern(r, c);
            ciot::matrix_set_ternary(&matrix, r, c, w);
            const std::int8_t got = ciot::matrix_get_ternary(&matrix, r, c);
            if (got != w) {
                std::cerr << "roundtrip mismatch at row=" << r << " col=" << c << "\n";
                ciot::matrix_free(&matrix);
                return 1;
            }
        }
    }

    float* x = static_cast<float*>(ciot::aligned_malloc(cols * sizeof(float)));
    float* y_ref = static_cast<float*>(ciot::aligned_malloc(rows * sizeof(float)));
    float* y_simd = static_cast<float*>(ciot::aligned_malloc(rows * sizeof(float)));
    if (!x || !y_ref || !y_simd) {
        std::cerr << "vector allocation failed\n";
        ciot::aligned_free(x);
        ciot::aligned_free(y_ref);
        ciot::aligned_free(y_simd);
        ciot::matrix_free(&matrix);
        return 1;
    }

    for (std::uint32_t c = 0; c < cols; ++c) {
        x[c] = static_cast<float>(static_cast<int>(c % 23U) - 11) * 0.0625f;
    }

    ciot::matvec_ternary_ref(&matrix, x, y_ref);
    ciot::matvec_ternary_simd(&matrix, x, y_simd);

    for (std::uint32_t r = 0; r < rows; ++r) {
        if (!close_enough(y_ref[r], y_simd[r])) {
            std::cerr << "matvec mismatch row=" << r << " ref=" << y_ref[r] << " simd=" << y_simd[r] << "\n";
            ciot::aligned_free(x);
            ciot::aligned_free(y_ref);
            ciot::aligned_free(y_simd);
            ciot::matrix_free(&matrix);
            return 1;
        }
    }

    constexpr std::uint32_t batch = 3;
    float* xb = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(batch) * cols * sizeof(float)));
    float* yb = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(batch) * rows * sizeof(float)));
    float* y_one = static_cast<float*>(ciot::aligned_malloc(rows * sizeof(float)));
    if (!xb || !yb || !y_one) {
        std::cerr << "batched vector allocation failed\n";
        ciot::aligned_free(xb);
        ciot::aligned_free(yb);
        ciot::aligned_free(y_one);
        ciot::aligned_free(x);
        ciot::aligned_free(y_ref);
        ciot::aligned_free(y_simd);
        ciot::matrix_free(&matrix);
        return 1;
    }
    for (std::uint32_t b = 0; b < batch; ++b) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            xb[(static_cast<std::size_t>(b) * cols) + c] = x[c] + static_cast<float>(b) * 0.01f;
        }
    }
    ciot::matmat_ternary_simd(&matrix, xb, batch, yb);
    for (std::uint32_t b = 0; b < batch; ++b) {
        ciot::matvec_ternary_simd(&matrix, xb + (static_cast<std::size_t>(b) * cols), y_one);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const float got = yb[(static_cast<std::size_t>(b) * rows) + r];
            if (!close_enough(got, y_one[r])) {
                std::cerr << "batched matvec mismatch batch=" << b << " row=" << r << " got=" << got << " expected=" << y_one[r] << "\n";
                ciot::aligned_free(xb);
                ciot::aligned_free(yb);
                ciot::aligned_free(y_one);
                ciot::aligned_free(x);
                ciot::aligned_free(y_ref);
                ciot::aligned_free(y_simd);
                ciot::matrix_free(&matrix);
                return 1;
            }
        }
    }
    ciot::aligned_free(xb);
    ciot::aligned_free(yb);
    ciot::aligned_free(y_one);

    float logits[4] = {1.0f, 2.0f, -1.0f, 0.5f};
    ciot::softmax_inplace(logits, 4);
    const float softmax_sum = logits[0] + logits[1] + logits[2] + logits[3];
    if (!close_enough(softmax_sum, 1.0f)) {
        std::cerr << "softmax sum mismatch: " << softmax_sum << "\n";
        ciot::aligned_free(x);
        ciot::aligned_free(y_ref);
        ciot::aligned_free(y_simd);
        ciot::matrix_free(&matrix);
        return 1;
    }

    float rope_a[8] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};
    float rope_b[8] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};
    ciot::RopeTable rope_table;
    if (!ciot::rope_table_init(&rope_table, 16, 8, 10000.0f)) {
        std::cerr << "rope table allocation failed\n";
        ciot::aligned_free(x);
        ciot::aligned_free(y_ref);
        ciot::aligned_free(y_simd);
        ciot::matrix_free(&matrix);
        return 1;
    }
    ciot::rope_pairwise(rope_a, 8, 7, 10000.0f);
    ciot::rope_table_apply(&rope_table, rope_b, 7);
    for (std::uint32_t i = 0; i < 8; ++i) {
        if (!close_enough(rope_a[i], rope_b[i])) {
            std::cerr << "rope mismatch index=" << i << " ref=" << rope_a[i] << " table=" << rope_b[i] << "\n";
            ciot::rope_table_free(&rope_table);
            ciot::aligned_free(x);
            ciot::aligned_free(y_ref);
            ciot::aligned_free(y_simd);
            ciot::matrix_free(&matrix);
            return 1;
        }
    }
    ciot::rope_table_free(&rope_table);

    ciot::KVCache kv_cache;
    if (!ciot::kv_cache_init(&kv_cache, 4, 8)) {
        std::cerr << "kv cache allocation failed\n";
        ciot::aligned_free(x); ciot::aligned_free(y_ref); ciot::aligned_free(y_simd);
        ciot::matrix_free(&matrix);
        return 1;
    }
    float key1[8] = {0.1f,-0.2f,0.3f,-0.4f,0.5f,-0.6f,0.7f,-0.8f};
    float val1[8] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f};
    ciot::kv_cache_append(&kv_cache, key1, val1);
    if (kv_cache.used != 1) { std::cerr << "kv cache append failed\n"; return 1; }
    ciot::kv_cache_reset(&kv_cache);
    if (kv_cache.used != 0) { std::cerr << "kv cache reset failed\n"; return 1; }
    ciot::kv_cache_append(&kv_cache, key1, val1);
    float attn_out[8] = {};
    float scores[4] = {};
    ciot::attention_decode_1head(&kv_cache, key1, attn_out, scores, 8);
    if (!close_enough(attn_out[0], 1.0f)) { std::cerr << "attention output mismatch\n"; return 1; }
    ciot::kv_cache_free(&kv_cache);

    std::cout << "test_linear: ok\n";
    ciot::aligned_free(x);
    ciot::aligned_free(y_ref);
    ciot::aligned_free(y_simd);
    ciot::matrix_free(&matrix);
    return 0;
}

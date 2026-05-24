#include "../include/Ciot.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

namespace {

bool close_enough(float a, float b) { return std::fabs(a - b) <= 0.0001f; }

std::int8_t rand_ternary() {
    int r = std::rand() % 9;
    if (r == 0 || r == 3) return 1;
    if (r == 1 || r == 6) return -1;
    return 0;
}

int fail(const char* msg) { std::cerr << "fail: " << msg << "\n"; return 1; }
void ok(const char* msg) { std::cout << "  ok  " << msg << "\n"; }

bool verify_matvec(ciot::TernaryMatrix* m, float* x, float* ysimd, float* yref, std::uint32_t rows, std::uint32_t cols) {
    ciot::matvec_ternary_simd(m, x, ysimd);
    ciot::matvec_ternary_ref(m, x, yref);
    for (std::uint32_t r = 0; r < rows; ++r)
        if (!close_enough(ysimd[r], yref[r])) return false;
    return true;
}

int test_fuzz() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    const std::uint32_t sizes[4][2] = {{17,13},{64,64},{100,50},{256,256}};
    for (int si = 0; si < 4; ++si) {
        std::uint32_t rows = sizes[si][0], cols = sizes[si][1];
        ciot::TernaryMatrix m;
        if (!ciot::matrix_init(&m, rows, cols)) return fail("fuzz alloc");
        for (std::uint32_t r = 0; r < rows; ++r) {
            m.row_scale[r] = 0.125f + static_cast<float>(r % 7) * 0.0625f;
            for (std::uint32_t c = 0; c < cols; ++c) ciot::matrix_set_ternary(&m, r, c, rand_ternary());
        }
        float* x = static_cast<float*>(ciot::aligned_malloc(cols * sizeof(float)));
        float* ys = static_cast<float*>(ciot::aligned_malloc(rows * sizeof(float)));
        float* yr = static_cast<float*>(ciot::aligned_malloc(rows * sizeof(float)));
        for (std::uint32_t c = 0; c < cols; ++c) x[c] = static_cast<float>(std::rand() % 100 - 50) * 0.01f;
        bool passed = verify_matvec(&m, x, ys, yr, rows, cols);
        ciot::aligned_free(x); ciot::aligned_free(ys); ciot::aligned_free(yr); ciot::matrix_free(&m);
        if (!passed) return fail("fuzz mismatch");
    }
    ok("fuzz: 4 random matrices, simd matches scalar");
    return 0;
}

int test_edge_cases() {
    ciot::TernaryMatrix m;
    if (!ciot::matrix_init(&m, 2, 64)) return fail("edge alloc");
    for (std::uint32_t r = 0; r < 2; ++r) {
        m.row_scale[r] = 1.0f;
        for (std::uint32_t c = 0; c < 64; ++c) ciot::matrix_set_ternary(&m, r, c, 1);
    }
    float x[64], ys[2], yr[2];
    for (int c = 0; c < 64; ++c) x[c] = static_cast<float>(c % 13 - 6) * 0.1f;
    ciot::matvec_ternary_simd(&m, x, ys);
    ciot::matvec_ternary_ref(&m, x, yr);
    for (int r = 0; r < 2; ++r)
        if (std::fabs(ys[r] - yr[r]) > 0.001f) { ciot::matrix_free(&m); return fail("edge mismatch"); }
    ciot::matrix_free(&m);
    ok("edge cases: all +1 simd matches ref");
    return 0;
}

int test_memory_stress() {
    for (int cycle = 0; cycle < 20; ++cycle) {
        ciot::TernaryMatrix m;
        if (!ciot::matrix_init(&m, 512, 512)) return fail("stress alloc");
        for (std::uint32_t r = 0; r < 512; ++r) {
            m.row_scale[r] = 0.1f;
            for (std::uint32_t c = 0; c < 512; ++c) ciot::matrix_set_ternary(&m, r, c, rand_ternary());
        }
        float* x = static_cast<float*>(ciot::aligned_malloc(512 * sizeof(float)));
        float* y = static_cast<float*>(ciot::aligned_malloc(512 * sizeof(float)));
        for (std::uint32_t c = 0; c < 512; ++c) x[c] = 1.0f;
        ciot::matvec_ternary_simd(&m, x, y);
        float chk = 0; for (std::uint32_t r = 0; r < 512; ++r) chk += y[r];
        ciot::matvec_ternary_simd(&m, x, y);
        float chk2 = 0; for (std::uint32_t r = 0; r < 512; ++r) chk2 += y[r];
        if (!close_enough(chk, chk2)) { ciot::aligned_free(x); ciot::aligned_free(y); ciot::matrix_free(&m); return fail("stress consistency"); }
        ciot::aligned_free(x); ciot::aligned_free(y); ciot::matrix_free(&m);
    }
    ok("memory stress: 20 alloc/free cycles, output consistent");
    return 0;
}

int test_kv_cache_stress() {
    constexpr std::uint32_t dim = 64, tokens = 128;
    ciot::KVCache cache;
    if (!ciot::kv_cache_init(&cache, tokens, dim)) return fail("kv cache alloc");
    float* key = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* val = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* query = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* out = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* scores = static_cast<float*>(ciot::aligned_malloc(tokens * sizeof(float)));
    for (std::uint32_t i = 0; i < dim; ++i) { key[i]=0.1f; val[i]=0.2f; query[i]=0.05f; }
    for (std::uint32_t t = 0; t < tokens; ++t) {
        ciot::kv_cache_append(&cache, key, val);
        if (cache.used != t + 1) { ciot::aligned_free(key); ciot::aligned_free(val); ciot::aligned_free(query); ciot::aligned_free(out); ciot::aligned_free(scores); ciot::kv_cache_free(&cache); return fail("kv cache count"); }
    }
    ciot::attention_decode_1head(&cache, query, out, scores, dim);
    float sum = 0; for (std::uint32_t i = 0; i < dim; ++i) sum += out[i];
    if (sum < 0.0001f) { ciot::aligned_free(key); ciot::aligned_free(val); ciot::aligned_free(query); ciot::aligned_free(out); ciot::aligned_free(scores); ciot::kv_cache_free(&cache); return fail("kv attention zero"); }
    ciot::kv_cache_reset(&cache);
    if (cache.used != 0) { ciot::aligned_free(key); ciot::aligned_free(val); ciot::aligned_free(query); ciot::aligned_free(out); ciot::aligned_free(scores); ciot::kv_cache_free(&cache); return fail("kv cache reset"); }
    ciot::aligned_free(key); ciot::aligned_free(val); ciot::aligned_free(query); ciot::aligned_free(out); ciot::aligned_free(scores);
    ciot::kv_cache_free(&cache);
    ok("kv cache stress: 128 tokens appended, attention computed, reset verified");
    return 0;
}

int test_mha_cache_stress() {
    constexpr std::uint32_t dim = 64, heads = 4, head_dim = 16, tokens = 64;
    ciot::MhaKVCache cache;
    if (!ciot::mha_kv_cache_init(&cache, tokens, heads, head_dim)) return fail("mha cache alloc");
    float* keys = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* vals = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* query = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* out = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* scores = static_cast<float*>(ciot::aligned_malloc(heads * tokens * sizeof(float)));
    for (std::uint32_t i = 0; i < dim; ++i) { keys[i]=0.1f+static_cast<float>(i)*0.001f; vals[i]=0.15f; query[i]=0.05f; }
    for (std::uint32_t t = 0; t < tokens; ++t) ciot::mha_kv_cache_append(&cache, keys, vals);
    ciot::mha_attention_decode(&cache, query, out, scores, heads, head_dim);
    float sum = 0; for (std::uint32_t i = 0; i < dim; ++i) sum += out[i];
    if (sum < 0.0001f) { ciot::aligned_free(keys); ciot::aligned_free(vals); ciot::aligned_free(query); ciot::aligned_free(out); ciot::aligned_free(scores); ciot::mha_kv_cache_free(&cache); return fail("mha attention zero"); }
    ciot::mha_kv_cache_reset(&cache);
    if (cache.used != 0) { ciot::aligned_free(keys); ciot::aligned_free(vals); ciot::aligned_free(query); ciot::aligned_free(out); ciot::aligned_free(scores); ciot::mha_kv_cache_free(&cache); return fail("mha reset"); }
    ciot::aligned_free(keys); ciot::aligned_free(vals); ciot::aligned_free(query); ciot::aligned_free(out); ciot::aligned_free(scores);
    ciot::mha_kv_cache_free(&cache);
    ok("mha cache stress: 64 tokens, 4 heads, attention computed, reset verified");
    return 0;
}

int test_decode_consistency() {
    constexpr std::uint32_t dim = 64, heads = 4, tokens = 32;
    ciot::TernaryMatrix mats[6];
    for (int i = 0; i < 6; ++i) {
        if (!ciot::matrix_init(&mats[i], dim, dim)) return fail("decode mat init");
        for (std::uint32_t r = 0; r < dim; ++r) {
            mats[i].row_scale[r] = 0.01f;
            for (std::uint32_t c = 0; c < dim; ++c) ciot::matrix_set_ternary(&mats[i], r, c, rand_ternary());
        }
    }
    ciot::RopeTable rope;
    if (!ciot::rope_table_init(&rope, tokens, dim, 10000.0f)) return fail("rope init");
    ciot::MhaKVCache cache;
    if (!ciot::mha_kv_cache_init(&cache, tokens, heads, dim/heads)) return fail("mha init");
    float* x = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* n1 = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    float* n2 = static_cast<float*>(ciot::aligned_malloc(dim * sizeof(float)));
    std::size_t scr = (static_cast<std::size_t>(dim) * (7U+heads*2U) + tokens*heads) * sizeof(float);
    float* scratch = static_cast<float*>(ciot::aligned_malloc(scr));
    for (std::uint32_t i = 0; i < dim; ++i) { x[i]=0.0f; n1[i]=1.0f; n2[i]=1.0f; }
    for (std::uint32_t t = 0; t < tokens; ++t)
        ciot::transformer_block_decode_mha(x, &mats[0],&mats[1],&mats[2],&mats[3],&mats[4],&mats[5], n1,n2, &rope, &cache, scratch, dim, heads, t, 1e-5f);
    float chk = 0; for (std::uint32_t i = 0; i < dim; ++i) chk += x[i];
    ciot::mha_kv_cache_reset(&cache);
    for (std::uint32_t i = 0; i < dim; ++i) x[i] = 0.0f;
    for (std::uint32_t t = 0; t < tokens; ++t)
        ciot::transformer_block_decode_mha(x, &mats[0],&mats[1],&mats[2],&mats[3],&mats[4],&mats[5], n1,n2, &rope, &cache, scratch, dim, heads, t, 1e-5f);
    float chk2 = 0; for (std::uint32_t i = 0; i < dim; ++i) chk2 += x[i];
    if (!close_enough(chk, chk2)) { ciot::aligned_free(x); ciot::aligned_free(n1); ciot::aligned_free(n2); ciot::aligned_free(scratch); ciot::mha_kv_cache_free(&cache); ciot::rope_table_free(&rope); for (int i=0;i<6;++i) ciot::matrix_free(&mats[i]); return fail("decode inconsistent"); }
    ciot::aligned_free(x); ciot::aligned_free(n1); ciot::aligned_free(n2); ciot::aligned_free(scratch);
    ciot::mha_kv_cache_free(&cache); ciot::rope_table_free(&rope);
    for (int i = 0; i < 6; ++i) ciot::matrix_free(&mats[i]);
    ok("decode consistency: 32 tokens x2, identical output");
    return 0;
}

} // namespace

int main() {
    std::cout << "production test suite\n\n";
    int fails = 0;
    fails += test_fuzz();
    fails += test_edge_cases();
    fails += test_memory_stress();
    fails += test_kv_cache_stress();
    fails += test_mha_cache_stress();
    fails += test_decode_consistency();
    std::cout << "\n" << (fails ? "FAIL" : "all tests passed") << "\n";
    return fails ? 1 : 0;
}

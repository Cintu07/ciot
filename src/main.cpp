#include "../include/Ciot.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace {

struct BenchStats {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double max_ms = 0.0;
    double logical_gops = 0.0;
    float checksum = 0.0f;
};

std::uint32_t parse_u32(const char* text, std::uint32_t fallback) {
    if (!text) {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || value == 0UL) {
        return fallback;
    }
    return static_cast<std::uint32_t>(value);
}

std::int8_t deterministic_ternary(std::uint32_t row, std::uint32_t col) {
    std::uint32_t h = row * 2654435761U;
    h ^= col * 2246822519U;
    h ^= h >> 15U;
    h *= 3266489917U;
    h ^= h >> 16U;

    const std::uint32_t bucket = h & 15U;
    const std::uint32_t pos = bucket == 0U || bucket == 3U;
    const std::uint32_t neg = bucket == 1U || bucket == 7U;
    return static_cast<std::int8_t>(static_cast<int>(pos) - static_cast<int>(neg));
}

void print_help() {
    std::cout << "Ciot: ternary-weight CPU inference core\n"
              << "\n"
              << "Commands:\n"
              << "  ciot --backend\n"
              << "  ciot --simd-test\n"
              << "  ciot --bench-linear [rows] [cols] [iters]\n"
              << "  ciot --bench-linear-pro [rows] [cols] [iters] [repeats] [warmup]\n"
              << "  ciot --bench-suite\n"
              << "  ciot --bench-batch [rows] [cols] [batch] [iters] [repeats]\n"
              << "  ciot --bench-bits <path> [iters]\n"
              << "  ciot --bench-transformer [dim] [iters] [repeats] [warmup]\n"
              << "  ciot --bench-rope [dim] [positions] [iters] [repeats]\n"
              << "  ciot --bench-decode [dim] [context] [iters] [repeats]\n"
              << "  ciot --bench-decode-mha [dim] [heads] [ctx] [iters] [repeats]\n"
              << "  ciot --decode-generate [dim] [tokens]\n"
              << "  ciot --generate-bits <bits> <vocab> [tokens]\n"
              << "  ciot --model-generate <model_dir> [prompt] [tokens]\n"
              << "  ciot --tiny-generate [tokens]\n"
              << "\n"
              << "Defaults: rows=1024 cols=1024 iters=200 repeats=9 warmup=20\n";
}

int run_simd_test() {
    if (std::strcmp(ciot::simd_backend_name(), "scalar-forced") == 0) {
        std::cout << "SIMD Backend: scalar-forced\n";
        std::cout << "SIMD Test: scalar backend selected by CIOT_BACKEND.\n";
        return 0;
    }
#if defined(__AVX512F__)
    const __m512 v1 = _mm512_set1_ps(10.0f);
    const __m512 v2 = _mm512_set1_ps(20.0f);
    const __m512 result = _mm512_add_ps(v1, v2);

    alignas(64) float out[16];
    _mm512_store_ps(out, result);

    std::cout << "SIMD Backend: " << ciot::simd_backend_name() << "\n";
    std::cout << "SIMD Test: " << out[0] << " (Should be 30)\n";
    return out[0] == 30.0f ? 0 : 1;
#elif defined(__AVX2__)
    __m256i v1 = _mm256_set1_epi32(10);
    __m256i v2 = _mm256_set1_epi32(20);
    __m256i result = _mm256_add_epi32(v1, v2);

    alignas(32) int out[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(out), result);

    std::cout << "SIMD Backend: " << ciot::simd_backend_name() << "\n";
    std::cout << "SIMD Test: " << out[0] << " (Should be 30)\n";
    return out[0] == 30 ? 0 : 1;
#elif defined(__ARM_NEON)
    const float32x4_t v1 = vdupq_n_f32(10.0f);
    const float32x4_t v2 = vdupq_n_f32(20.0f);
    const float32x4_t result = vaddq_f32(v1, v2);

    alignas(16) float out[4];
    vst1q_f32(out, result);

    std::cout << "SIMD Backend: " << ciot::simd_backend_name() << "\n";
    std::cout << "SIMD Test: " << out[0] << " (Should be 30)\n";
    return out[0] == 30.0f ? 0 : 1;
#else
    std::cout << "SIMD Backend: " << ciot::simd_backend_name() << "\n";
    std::cout << "SIMD Test: scalar fallback is active.\n";
    return 0;
#endif
}

void fill_input(float* x, std::uint32_t cols) {
    for (std::uint32_t c = 0; c < cols; ++c) {
        x[c] = static_cast<float>((static_cast<int>(c % 17U) - 8)) * 0.03125f;
    }
}

void fill_matrix_scaled(ciot::TernaryMatrix* matrix, float scale) {
    for (std::uint32_t r = 0; r < matrix->rows; ++r) {
        matrix->row_scale[r] = scale;
        for (std::uint32_t c = 0; c < matrix->cols; ++c) {
            ciot::matrix_set_ternary(matrix, r, c, deterministic_ternary(r, c));
        }
    }
}

void fill_matrix(ciot::TernaryMatrix* matrix) {
    fill_matrix_scaled(matrix, 1.0f / 16.0f);
}

BenchStats bench_matvec(ciot::TernaryMatrix* matrix, float* x, float* y, std::uint32_t iters, std::uint32_t repeats, std::uint32_t warmup) {
    for (std::uint32_t i = 0; i < warmup; ++i) {
        ciot::matvec_ternary_simd(matrix, x, y);
    }

    double* samples = new double[repeats];
    double total_ms = 0.0;

    for (std::uint32_t r = 0; r < repeats; ++r) {
        const std::uint64_t start = ciot::ticks_ns();
        for (std::uint32_t i = 0; i < iters; ++i) {
            ciot::matvec_ternary_simd(matrix, x, y);
        }
        const std::uint64_t end = ciot::ticks_ns();
        const double avg_ns = static_cast<double>(end - start) / static_cast<double>(iters);
        samples[r] = avg_ns / 1000000.0;
        total_ms += samples[r];
    }

    BenchStats stats;
    std::sort(samples, samples + repeats);
    stats.avg_ms = total_ms / static_cast<double>(repeats);
    stats.min_ms = samples[0];
    stats.median_ms = samples[repeats / 2U];
    const std::uint32_t p95_index = (repeats <= 1U) ? 0U : static_cast<std::uint32_t>((static_cast<std::uint64_t>(repeats - 1U) * 95ULL) / 100ULL);
    stats.p95_ms = samples[p95_index];
    stats.max_ms = samples[repeats - 1U];

    float checksum = 0.0f;
    for (std::uint32_t r = 0; r < matrix->rows; ++r) {
        checksum += y[r];
    }
    stats.checksum = checksum;

    const double avg_ns = stats.median_ms * 1000000.0;
    const double logical_ops = static_cast<double>(matrix->rows) * static_cast<double>(matrix->cols) * 2.0;
    stats.logical_gops = logical_ops / avg_ns;

    delete[] samples;
    return stats;
}

void print_pro_bench(const char* name, const ciot::TernaryMatrix* matrix, std::uint32_t iters, std::uint32_t repeats, const BenchStats& stats) {
    std::cout << "ciot_bench_v1\n"
              << "  name: " << name << "\n"
              << "  backend: " << ciot::simd_backend_name() << "\n"
              << "  rows: " << matrix->rows << "\n"
              << "  cols: " << matrix->cols << "\n"
              << "  iters_per_repeat: " << iters << "\n"
              << "  repeats: " << repeats << "\n"
              << "  avg_ms: " << stats.avg_ms << "\n"
              << "  min_ms: " << stats.min_ms << "\n"
              << "  median_ms: " << stats.median_ms << "\n"
              << "  p95_ms: " << stats.p95_ms << "\n"
              << "  max_ms: " << stats.max_ms << "\n"
              << "  median_logical_Gop/s: " << stats.logical_gops << "\n"
              << "  checksum: " << stats.checksum << "\n";
}

int run_bench_loaded_bits(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "missing .bits path\n";
        return 1;
    }

    const std::uint32_t iters = argc > 3 ? parse_u32(argv[3], 200U) : 200U;
    ciot::TernaryMatrix matrix;
    if (!ciot::matrix_load_bits(&matrix, argv[2])) {
        std::cerr << "failed to load .bits file: " << argv[2] << "\n";
        return 1;
    }

    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(matrix.cols) * sizeof(float)));
    float* y = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(matrix.rows) * sizeof(float)));
    if (!x || !y) {
        std::cerr << "failed to allocate vectors\n";
        ciot::aligned_free(x);
        ciot::aligned_free(y);
        ciot::matrix_free(&matrix);
        return 1;
    }

    fill_input(x, matrix.cols);
    const BenchStats stats = bench_matvec(&matrix, x, y, iters, 9U, 20U);
    print_pro_bench("bits_matvec", &matrix, iters, 9U, stats);

    ciot::aligned_free(x);
    ciot::aligned_free(y);
    ciot::matrix_free(&matrix);
    return 0;
}

int run_bench_linear_pro(std::uint32_t rows, std::uint32_t cols, std::uint32_t iters, std::uint32_t repeats, std::uint32_t warmup, const char* name) {
    ciot::TernaryMatrix matrix;
    if (!ciot::matrix_init(&matrix, rows, cols)) {
        std::cerr << "failed to allocate ternary matrix\n";
        return 1;
    }

    fill_matrix(&matrix);

    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(cols) * sizeof(float)));
    float* y = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(rows) * sizeof(float)));
    if (!x || !y) {
        std::cerr << "failed to allocate vectors\n";
        ciot::aligned_free(x);
        ciot::aligned_free(y);
        ciot::matrix_free(&matrix);
        return 1;
    }

    fill_input(x, cols);
    const BenchStats stats = bench_matvec(&matrix, x, y, iters, repeats, warmup);
    print_pro_bench(name, &matrix, iters, repeats, stats);

    ciot::aligned_free(x);
    ciot::aligned_free(y);
    ciot::matrix_free(&matrix);
    return 0;
}

int run_bench_linear(int argc, char** argv) {
    const std::uint32_t rows = argc > 2 ? parse_u32(argv[2], 1024U) : 1024U;
    const std::uint32_t cols = argc > 3 ? parse_u32(argv[3], 1024U) : 1024U;
    const std::uint32_t iters = argc > 4 ? parse_u32(argv[4], 200U) : 200U;
    return run_bench_linear_pro(rows, cols, iters, 1U, 1U, "linear_matvec_legacy");
}

int run_bench_linear_pro_cmd(int argc, char** argv) {
    const std::uint32_t rows = argc > 2 ? parse_u32(argv[2], 1024U) : 1024U;
    const std::uint32_t cols = argc > 3 ? parse_u32(argv[3], 1024U) : 1024U;
    const std::uint32_t iters = argc > 4 ? parse_u32(argv[4], 200U) : 200U;
    const std::uint32_t repeats = argc > 5 ? parse_u32(argv[5], 9U) : 9U;
    const std::uint32_t warmup = argc > 6 ? parse_u32(argv[6], 20U) : 20U;
    return run_bench_linear_pro(rows, cols, iters, repeats, warmup, "linear_matvec");
}

int run_bench_suite() {
    std::cout << "name,backend,rows,cols,iters,repeats,min_ms,median_ms,p95_ms,max_ms,median_logical_Gop_s,checksum\n";
    const std::uint32_t sizes[4] = {256U, 512U, 1024U, 2048U};
    for (std::uint32_t s = 0; s < 4U; ++s) {
        const std::uint32_t n = sizes[s];
        ciot::TernaryMatrix matrix;
        if (!ciot::matrix_init(&matrix, n, n)) {
            return 1;
        }
        fill_matrix(&matrix);
        float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(n) * sizeof(float)));
        float* y = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(n) * sizeof(float)));
        if (!x || !y) {
            ciot::aligned_free(x);
            ciot::aligned_free(y);
            ciot::matrix_free(&matrix);
            return 1;
        }
        fill_input(x, n);
        const std::uint32_t iters = n <= 512U ? 300U : 80U;
        const BenchStats stats = bench_matvec(&matrix, x, y, iters, 7U, 20U);
        std::cout << "linear_matvec," << ciot::simd_backend_name() << "," << n << "," << n << "," << iters << ",7,"
                  << stats.min_ms << "," << stats.median_ms << "," << stats.p95_ms << "," << stats.max_ms << ","
                  << stats.logical_gops << "," << stats.checksum << "\n";
        ciot::aligned_free(x);
        ciot::aligned_free(y);
        ciot::matrix_free(&matrix);
    }
    return 0;
}

int run_bench_batch(int argc, char** argv) {
    const std::uint32_t rows = argc > 2 ? parse_u32(argv[2], 1024U) : 1024U;
    const std::uint32_t cols = argc > 3 ? parse_u32(argv[3], 1024U) : 1024U;
    const std::uint32_t batch = argc > 4 ? parse_u32(argv[4], 4U) : 4U;
    const std::uint32_t iters = argc > 5 ? parse_u32(argv[5], 50U) : 50U;
    const std::uint32_t repeats = argc > 6 ? parse_u32(argv[6], 5U) : 5U;

    ciot::TernaryMatrix matrix;
    if (!ciot::matrix_init(&matrix, rows, cols)) {
        return 1;
    }
    fill_matrix(&matrix);

    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(batch) * cols * sizeof(float)));
    float* y = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(batch) * rows * sizeof(float)));
    double* samples = new double[repeats];
    if (!x || !y || !samples) {
        ciot::aligned_free(x);
        ciot::aligned_free(y);
        delete[] samples;
        ciot::matrix_free(&matrix);
        return 1;
    }

    for (std::uint32_t b = 0; b < batch; ++b) {
        fill_input(x + (static_cast<std::size_t>(b) * cols), cols);
    }
    for (std::uint32_t i = 0; i < 10U; ++i) {
        ciot::matmat_ternary_simd(&matrix, x, batch, y);
    }

    double total_ms = 0.0;
    for (std::uint32_t r = 0; r < repeats; ++r) {
        const std::uint64_t start = ciot::ticks_ns();
        for (std::uint32_t i = 0; i < iters; ++i) {
            ciot::matmat_ternary_simd(&matrix, x, batch, y);
        }
        const std::uint64_t end = ciot::ticks_ns();
        samples[r] = (static_cast<double>(end - start) / static_cast<double>(iters)) / 1000000.0;
        total_ms += samples[r];
    }
    std::sort(samples, samples + repeats);

    float checksum = 0.0f;
    for (std::uint32_t i = 0; i < batch * rows; ++i) {
        checksum += y[i];
    }
    const std::uint32_t p95_index = (repeats <= 1U) ? 0U : static_cast<std::uint32_t>((static_cast<std::uint64_t>(repeats - 1U) * 95ULL) / 100ULL);
    const double logical_ops = static_cast<double>(rows) * static_cast<double>(cols) * static_cast<double>(batch) * 2.0;
    const double logical_gops = logical_ops / (samples[repeats / 2U] * 1000000.0);
    std::cout << "ciot_batched_matvec_v1\n"
              << "  backend: " << ciot::simd_backend_name() << "\n"
              << "  rows: " << rows << "\n"
              << "  cols: " << cols << "\n"
              << "  batch: " << batch << "\n"
              << "  iters_per_repeat: " << iters << "\n"
              << "  repeats: " << repeats << "\n"
              << "  avg_ms: " << (total_ms / static_cast<double>(repeats)) << "\n"
              << "  min_ms: " << samples[0] << "\n"
              << "  median_ms: " << samples[repeats / 2U] << "\n"
              << "  p95_ms: " << samples[p95_index] << "\n"
              << "  max_ms: " << samples[repeats - 1U] << "\n"
              << "  median_logical_Gop/s: " << logical_gops << "\n"
              << "  checksum: " << checksum << "\n";

    ciot::aligned_free(x);
    ciot::aligned_free(y);
    delete[] samples;
    ciot::matrix_free(&matrix);
    return 0;
}

int run_bench_transformer(int argc, char** argv) {
    const std::uint32_t dim = argc > 2 ? parse_u32(argv[2], 256U) : 256U;
    const std::uint32_t iters = argc > 3 ? parse_u32(argv[3], 50U) : 50U;
    const std::uint32_t repeats = argc > 4 ? parse_u32(argv[4], 5U) : 5U;
    const std::uint32_t warmup = argc > 5 ? parse_u32(argv[5], 10U) : 10U;

    ciot::TernaryMatrix matrices[6];
    for (std::uint32_t i = 0; i < 6U; ++i) {
        if (!ciot::matrix_init(&matrices[i], dim, dim)) {
            for (std::uint32_t j = 0; j < i; ++j) {
                ciot::matrix_free(&matrices[j]);
            }
            return 1;
        }
        fill_matrix_scaled(&matrices[i], 1.0f / 64.0f);
    }

    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* norm1 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* norm2 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* scratch = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * 6U * sizeof(float)));
    double* samples = new double[repeats];
    if (!x || !norm1 || !norm2 || !scratch || !samples) {
        ciot::aligned_free(x);
        ciot::aligned_free(norm1);
        ciot::aligned_free(norm2);
        ciot::aligned_free(scratch);
        delete[] samples;
        for (std::uint32_t i = 0; i < 6U; ++i) {
            ciot::matrix_free(&matrices[i]);
        }
        return 1;
    }

    for (std::uint32_t i = 0; i < dim; ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 19U) - 9) * 0.01f;
        norm1[i] = 1.0f;
        norm2[i] = 1.0f;
    }

    for (std::uint32_t i = 0; i < warmup; ++i) {
        ciot::transformer_block_1tok(x, &matrices[0], &matrices[1], &matrices[2], &matrices[3], &matrices[4], &matrices[5], norm1, norm2, scratch, dim, 1.0e-5f);
    }

    double total_ms = 0.0;
    for (std::uint32_t r = 0; r < repeats; ++r) {
        const std::uint64_t start = ciot::ticks_ns();
        for (std::uint32_t i = 0; i < iters; ++i) {
            ciot::transformer_block_1tok(x, &matrices[0], &matrices[1], &matrices[2], &matrices[3], &matrices[4], &matrices[5], norm1, norm2, scratch, dim, 1.0e-5f);
        }
        const std::uint64_t end = ciot::ticks_ns();
        samples[r] = (static_cast<double>(end - start) / static_cast<double>(iters)) / 1000000.0;
        total_ms += samples[r];
    }

    std::sort(samples, samples + repeats);
    float checksum = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) {
        checksum += x[i];
    }

    const std::uint32_t p95_index = (repeats <= 1U) ? 0U : static_cast<std::uint32_t>((static_cast<std::uint64_t>(repeats - 1U) * 95ULL) / 100ULL);
    const double logical_ops = static_cast<double>(dim) * static_cast<double>(dim) * 6.0 * 2.0;
    const double logical_gops = logical_ops / (samples[repeats / 2U] * 1000000.0);
    std::cout << "ciot_transformer_block_v1\n"
              << "  backend: " << ciot::simd_backend_name() << "\n"
              << "  dim: " << dim << "\n"
              << "  iters_per_repeat: " << iters << "\n"
              << "  repeats: " << repeats << "\n"
              << "  avg_ms: " << (total_ms / static_cast<double>(repeats)) << "\n"
              << "  min_ms: " << samples[0] << "\n"
              << "  median_ms: " << samples[repeats / 2U] << "\n"
              << "  p95_ms: " << samples[p95_index] << "\n"
              << "  max_ms: " << samples[repeats - 1U] << "\n"
              << "  median_logical_Gop/s: " << logical_gops << "\n"
              << "  checksum: " << checksum << "\n";

    ciot::aligned_free(x);
    ciot::aligned_free(norm1);
    ciot::aligned_free(norm2);
    ciot::aligned_free(scratch);
    delete[] samples;
    for (std::uint32_t i = 0; i < 6U; ++i) {
        ciot::matrix_free(&matrices[i]);
    }
    return 0;
}

int run_bench_rope(int argc, char** argv) {
    const std::uint32_t dim = argc > 2 ? parse_u32(argv[2], 1024U) : 1024U;
    const std::uint32_t positions = argc > 3 ? parse_u32(argv[3], 2048U) : 2048U;
    const std::uint32_t iters = argc > 4 ? parse_u32(argv[4], 1000U) : 1000U;
    const std::uint32_t repeats = argc > 5 ? parse_u32(argv[5], 5U) : 5U;

    ciot::RopeTable table;
    if (!ciot::rope_table_init(&table, positions, dim, 10000.0f)) {
        return 1;
    }
    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    double* samples = new double[repeats];
    if (!x || !samples) {
        ciot::aligned_free(x);
        delete[] samples;
        ciot::rope_table_free(&table);
        return 1;
    }
    fill_input(x, dim);

    for (std::uint32_t i = 0; i < 20U; ++i) {
        ciot::rope_table_apply(&table, x, i % positions);
    }

    double total_ms = 0.0;
    for (std::uint32_t r = 0; r < repeats; ++r) {
        const std::uint64_t start = ciot::ticks_ns();
        for (std::uint32_t i = 0; i < iters; ++i) {
            ciot::rope_table_apply(&table, x, i % positions);
        }
        const std::uint64_t end = ciot::ticks_ns();
        samples[r] = (static_cast<double>(end - start) / static_cast<double>(iters)) / 1000000.0;
        total_ms += samples[r];
    }

    std::sort(samples, samples + repeats);
    float checksum = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) {
        checksum += x[i];
    }
    const std::uint32_t p95_index = (repeats <= 1U) ? 0U : static_cast<std::uint32_t>((static_cast<std::uint64_t>(repeats - 1U) * 95ULL) / 100ULL);
    std::cout << "ciot_rope_table_v1\n"
              << "  dim: " << dim << "\n"
              << "  positions: " << positions << "\n"
              << "  iters_per_repeat: " << iters << "\n"
              << "  repeats: " << repeats << "\n"
              << "  avg_ms: " << (total_ms / static_cast<double>(repeats)) << "\n"
              << "  min_ms: " << samples[0] << "\n"
              << "  median_ms: " << samples[repeats / 2U] << "\n"
              << "  p95_ms: " << samples[p95_index] << "\n"
              << "  max_ms: " << samples[repeats - 1U] << "\n"
              << "  checksum: " << checksum << "\n";

    ciot::aligned_free(x);
    delete[] samples;
    ciot::rope_table_free(&table);
    return 0;
}

int run_bench_decode(int argc, char** argv) {
    const std::uint32_t dim = argc > 2 ? parse_u32(argv[2], 256U) : 256U;
    const std::uint32_t context = argc > 3 ? parse_u32(argv[3], 64U) : 64U;
    const std::uint32_t iters = argc > 4 ? parse_u32(argv[4], 10U) : 10U;
    const std::uint32_t repeats = argc > 5 ? parse_u32(argv[5], 5U) : 5U;

    ciot::TernaryMatrix matrices[6];
    for (std::uint32_t i = 0; i < 6U; ++i) {
        if (!ciot::matrix_init(&matrices[i], dim, dim)) {
            for (std::uint32_t j = 0; j < i; ++j) ciot::matrix_free(&matrices[j]);
            return 1;
        }
        fill_matrix_scaled(&matrices[i], 1.0f / 64.0f);
    }

    ciot::RopeTable rope;
    if (!ciot::rope_table_init(&rope, context + 32U, dim, 10000.0f)) {
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    ciot::KVCache cache;
    if (!ciot::kv_cache_init(&cache, context, dim)) {
        ciot::rope_table_free(&rope);
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* norm1 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* norm2 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* scratch = static_cast<float*>(ciot::aligned_malloc((static_cast<std::size_t>(dim) * 7U + context) * sizeof(float)));
    double* samples = new double[repeats];
    if (!x || !norm1 || !norm2 || !scratch || !samples) {
        ciot::aligned_free(x); ciot::aligned_free(norm1); ciot::aligned_free(norm2); ciot::aligned_free(scratch); delete[] samples;
        ciot::kv_cache_free(&cache); ciot::rope_table_free(&rope);
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    for (std::uint32_t i = 0; i < dim; ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 19U) - 9) * 0.01f;
        norm1[i] = 1.0f;
        norm2[i] = 1.0f;
    }

    for (std::uint32_t p = 0; p < context - 1U; ++p) {
        ciot::transformer_block_decode_1tok(x, &matrices[0], &matrices[1], &matrices[2], &matrices[3], &matrices[4], &matrices[5], norm1, norm2, &rope, &cache, scratch, dim, p, 1.0e-5f);
    }

    for (std::uint32_t i = 0; i < 10U; ++i) {
        ciot::kv_cache_reset(&cache);
        for (std::uint32_t p = 0; p < context - 1U; ++p) {
            ciot::transformer_block_decode_1tok(x, &matrices[0], &matrices[1], &matrices[2], &matrices[3], &matrices[4], &matrices[5], norm1, norm2, &rope, &cache, scratch, dim, p, 1.0e-5f);
        }
    }

    double total_ms = 0.0;
    for (std::uint32_t r = 0; r < repeats; ++r) {
        ciot::kv_cache_reset(&cache);
        const std::uint64_t start = ciot::ticks_ns();
        for (std::uint32_t i = 0; i < iters; ++i) {
            for (std::uint32_t p = 0; p < context; ++p) {
                ciot::transformer_block_decode_1tok(x, &matrices[0], &matrices[1], &matrices[2], &matrices[3], &matrices[4], &matrices[5], norm1, norm2, &rope, &cache, scratch, dim, p, 1.0e-5f);
            }
            ciot::kv_cache_reset(&cache);
        }
        const std::uint64_t end = ciot::ticks_ns();
        samples[r] = (static_cast<double>(end - start) / static_cast<double>(iters)) / 1000000.0;
        total_ms += samples[r];
    }
    std::sort(samples, samples + repeats);

    float checksum = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) checksum += x[i];
    const std::uint32_t p95_index = (repeats <= 1U) ? 0U : static_cast<std::uint32_t>((static_cast<std::uint64_t>(repeats - 1U) * 95ULL) / 100ULL);
    const double logical_ops = static_cast<double>(dim) * static_cast<double>(dim) * 6.0 * 2.0 * static_cast<double>(context);
    const double logical_gops = logical_ops / (samples[repeats / 2U] * 1000000.0);
    std::cout << "ciot_decode_block_v1\n"
              << "  backend: " << ciot::simd_backend_name() << "\n"
              << "  dim: " << dim << "\n"
              << "  context: " << context << "\n"
              << "  iters_per_repeat: " << iters << "\n"
              << "  repeats: " << repeats << "\n"
              << "  avg_ms: " << (total_ms / static_cast<double>(repeats)) << "\n"
              << "  min_ms: " << samples[0] << "\n"
              << "  median_ms: " << samples[repeats / 2U] << "\n"
              << "  p95_ms: " << samples[p95_index] << "\n"
              << "  max_ms: " << samples[repeats - 1U] << "\n"
              << "  median_logical_Gop/s: " << logical_gops << "\n"
              << "  checksum: " << checksum << "\n";

    ciot::aligned_free(x); ciot::aligned_free(norm1); ciot::aligned_free(norm2); ciot::aligned_free(scratch);
    delete[] samples;
    ciot::kv_cache_free(&cache);
    ciot::rope_table_free(&rope);
    for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
    return 0;
}

int run_decode_generate(int argc, char** argv) {
    const std::uint32_t dim = argc > 2 ? parse_u32(argv[2], 64U) : 64U;
    const std::uint32_t tokens = argc > 3 ? parse_u32(argv[3], 16U) : 16U;
    constexpr std::uint32_t vocab = 8U;
    const char* words[vocab] = {"ciot", "packs", "ternary", "bits", "on", "neon", "very", "fast"};

    ciot::TernaryMatrix matrices[6];
    for (std::uint32_t i = 0; i < 6U; ++i) {
        if (!ciot::matrix_init(&matrices[i], dim, dim)) {
            for (std::uint32_t j = 0; j < i; ++j) ciot::matrix_free(&matrices[j]);
            return 1;
        }
        fill_matrix_scaled(&matrices[i], 1.0f / 64.0f);
    }

    ciot::RopeTable rope;
    if (!ciot::rope_table_init(&rope, tokens + 8U, dim, 10000.0f)) {
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    ciot::KVCache cache;
    if (!ciot::kv_cache_init(&cache, tokens, dim)) {
        ciot::rope_table_free(&rope);
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* norm1 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* norm2 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* scratch = static_cast<float*>(ciot::aligned_malloc((static_cast<std::size_t>(dim) * 7U + tokens) * sizeof(float)));
    if (!x || !norm1 || !norm2 || !scratch) {
        ciot::aligned_free(x); ciot::aligned_free(norm1); ciot::aligned_free(norm2); ciot::aligned_free(scratch);
        ciot::kv_cache_free(&cache); ciot::rope_table_free(&rope);
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    for (std::uint32_t i = 0; i < dim; ++i) { x[i] = 0.0f; norm1[i] = 1.0f; norm2[i] = 1.0f; }

    std::cout << "decode generation using " << ciot::simd_backend_name() << ": ";
    for (std::uint32_t t = 0; t < tokens; ++t) {
        ciot::transformer_block_decode_1tok(x, &matrices[0], &matrices[1], &matrices[2], &matrices[3], &matrices[4], &matrices[5], norm1, norm2, &rope, &cache, scratch, dim, t, 1.0e-5f);
        std::uint32_t best = 0U;
        float best_val = x[0];
        for (std::uint32_t i = 1; i < dim && i < vocab; ++i) {
            if (x[i] > best_val) { best = i; best_val = x[i]; }
        }
        std::cout << words[best % vocab] << (t + 1U == tokens ? "" : " ");
    }
    std::cout << "\n";

    ciot::aligned_free(x); ciot::aligned_free(norm1); ciot::aligned_free(norm2); ciot::aligned_free(scratch);
    ciot::kv_cache_free(&cache);
    ciot::rope_table_free(&rope);
    for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
    return 0;
}

int run_bench_decode_mha(int argc, char** argv) {
    const std::uint32_t dim = argc > 2 ? parse_u32(argv[2], 128U) : 128U;
    const std::uint32_t num_heads = argc > 3 ? parse_u32(argv[3], 4U) : 4U;
    const std::uint32_t context = argc > 4 ? parse_u32(argv[4], 32U) : 32U;
    const std::uint32_t iters = argc > 5 ? parse_u32(argv[5], 8U) : 8U;
    const std::uint32_t repeats = argc > 6 ? parse_u32(argv[6], 3U) : 3U;

    ciot::TernaryMatrix matrices[6];
    for (std::uint32_t i = 0; i < 6U; ++i) {
        if (!ciot::matrix_init(&matrices[i], dim, dim)) {
            for (std::uint32_t j = 0; j < i; ++j) ciot::matrix_free(&matrices[j]);
            return 1;
        }
        fill_matrix_scaled(&matrices[i], 1.0f / 64.0f);
    }

    ciot::RopeTable rope;
    if (!ciot::rope_table_init(&rope, context + 32U, dim, 10000.0f)) {
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    ciot::MhaKVCache cache;
    if (!ciot::mha_kv_cache_init(&cache, context, num_heads, dim / num_heads)) {
        ciot::rope_table_free(&rope);
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* norm1 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    float* norm2 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(dim) * sizeof(float)));
    std::size_t scr = (static_cast<std::size_t>(dim) * (7U + num_heads * 2U) + context * num_heads) * sizeof(float);
    float* scratch = static_cast<float*>(ciot::aligned_malloc(scr));
    double* samples = new double[repeats];
    if (!x || !norm1 || !norm2 || !scratch || !samples) {
        ciot::aligned_free(x); ciot::aligned_free(norm1); ciot::aligned_free(norm2); ciot::aligned_free(scratch);
        delete[] samples; ciot::mha_kv_cache_free(&cache); ciot::rope_table_free(&rope);
        for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
        return 1;
    }

    for (std::uint32_t i = 0; i < dim; ++i) { x[i] = 0.0f; norm1[i] = 1.0f; norm2[i] = 1.0f; }

    for (std::uint32_t w = 0; w < 10U; ++w) {
        ciot::mha_kv_cache_reset(&cache);
        for (std::uint32_t p = 0; p < context - 1U; ++p)
            ciot::transformer_block_decode_mha(x, &matrices[0], &matrices[1], &matrices[2], &matrices[3], &matrices[4], &matrices[5], norm1, norm2, &rope, &cache, scratch, dim, num_heads, p, 1.0e-5f);
    }

    double total_ms = 0.0;
    for (std::uint32_t r = 0; r < repeats; ++r) {
        ciot::mha_kv_cache_reset(&cache);
        const std::uint64_t start = ciot::ticks_ns();
        for (std::uint32_t i = 0; i < iters; ++i) {
            for (std::uint32_t p = 0; p < context; ++p)
                ciot::transformer_block_decode_mha(x, &matrices[0], &matrices[1], &matrices[2], &matrices[3], &matrices[4], &matrices[5], norm1, norm2, &rope, &cache, scratch, dim, num_heads, p, 1.0e-5f);
            ciot::mha_kv_cache_reset(&cache);
        }
        const std::uint64_t end = ciot::ticks_ns();
        samples[r] = (static_cast<double>(end - start) / static_cast<double>(iters)) / 1000000.0;
        total_ms += samples[r];
    }
    std::sort(samples, samples + repeats);

    float checksum = 0.0f;
    for (std::uint32_t i = 0; i < dim; ++i) checksum += x[i];
    const std::uint32_t p95_index = (repeats <= 1U) ? 0U : static_cast<std::uint32_t>((static_cast<std::uint64_t>(repeats - 1U) * 95ULL) / 100ULL);
    const double logical_ops = static_cast<double>(dim) * static_cast<double>(dim) * 6.0 * 2.0 * static_cast<double>(context);
    const double logical_gops = logical_ops / (samples[repeats / 2U] * 1000000.0);
    std::cout << "ciot_decode_mha_v1\n"
              << "  backend: " << ciot::simd_backend_name() << "\n"
              << "  dim: " << dim << "  heads: " << num_heads << "\n"
              << "  context: " << context << "\n"
              << "  iters_per_repeat: " << iters << "  repeats: " << repeats << "\n"
              << "  avg_ms: " << (total_ms / static_cast<double>(repeats)) << "\n"
              << "  min_ms: " << samples[0] << "\n"
              << "  median_ms: " << samples[repeats / 2U] << "\n"
              << "  p95_ms: " << samples[p95_index] << "\n"
              << "  max_ms: " << samples[repeats - 1U] << "\n"
              << "  median_logical_Gop/s: " << logical_gops << "\n"
              << "  checksum: " << checksum << "\n";

    ciot::aligned_free(x); ciot::aligned_free(norm1); ciot::aligned_free(norm2); ciot::aligned_free(scratch);
    delete[] samples; ciot::mha_kv_cache_free(&cache); ciot::rope_table_free(&rope);
    for (std::uint32_t i = 0; i < 6U; ++i) ciot::matrix_free(&matrices[i]);
    return 0;
}

int run_model_generate(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: ciot --model-generate <model_dir> [prompt] [tokens]\n"; return 1; }
    const char* model_dir = argv[2];
    const char* prompt = argc > 3 ? argv[3] : "ciot";
    const std::uint32_t tokens = argc > 4 ? parse_u32(argv[4], 16U) : 16U;

    char config_path[512]; std::snprintf(config_path, sizeof(config_path), "%s/config.ciot", model_dir);
    ciot::CiotModelConfig cfg;
    if (!ciot::model_config_load(&cfg, config_path)) {
        std::cerr << "failed to load config: " << config_path << "\n"; return 1;
    }

    std::cout << "model: dim=" << cfg.dim << " layers=" << cfg.num_layers
              << " heads=" << cfg.num_heads << " vocab=" << cfg.vocab_size
              << " backend=" << ciot::simd_backend_name() << "\n";

    ciot::Tokenizer tok;
    char vocab_path[512]; std::snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.txt", model_dir);
    if (!ciot::tokenizer_load(&tok, vocab_path)) {
        std::cerr << "failed to load vocab: " << vocab_path << "\n"; return 1;
    }

    const std::uint32_t L = cfg.num_layers;
    ciot::TernaryMatrix embed, lm_head;
    ciot::TernaryMatrix* wq = new ciot::TernaryMatrix[L];
    ciot::TernaryMatrix* wk = new ciot::TernaryMatrix[L];
    ciot::TernaryMatrix* wv = new ciot::TernaryMatrix[L];
    ciot::TernaryMatrix* wo = new ciot::TernaryMatrix[L];
    ciot::TernaryMatrix* w1 = new ciot::TernaryMatrix[L];
    ciot::TernaryMatrix* w2 = new ciot::TernaryMatrix[L];
    float* norm1 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(L * cfg.dim) * sizeof(float)));
    float* norm2 = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(L * cfg.dim) * sizeof(float)));

    if (!ciot::model_load_weights(&cfg, model_dir, &embed, &lm_head, wq, wk, wv, wo, w1, w2, norm1, norm2)) {
        std::cerr << "failed to load weights\n";
        ciot::tokenizer_free(&tok); return 1;
    }

    ciot::RopeTable rope;
    ciot::rope_table_init(&rope, cfg.max_context, cfg.dim, cfg.rope_theta);
    ciot::MhaKVCache* caches = new ciot::MhaKVCache[L];
    for (std::uint32_t l = 0; l < L; ++l)
        ciot::mha_kv_cache_init(&caches[l], cfg.max_context, cfg.num_heads, cfg.dim / cfg.num_heads);

    float* x = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(cfg.dim) * sizeof(float)));
    std::size_t scr = (static_cast<std::size_t>(cfg.dim) * (7U + cfg.num_heads * 2U) + cfg.max_context * cfg.num_heads);
    float* scratch = static_cast<float*>(ciot::aligned_malloc(scr * sizeof(float)));
    float* embed_vec = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(cfg.dim) * sizeof(float)));

    for (std::uint32_t i = 0; i < cfg.dim; ++i) x[i] = 0.0f;

    std::cout << "prompt: \"" << prompt << "\" -> ";
    std::uint32_t prompt_ids[64];
    std::uint32_t n_prompt = ciot::tokenizer_encode_text(&tok, prompt, prompt_ids, 64);
    if (n_prompt == 0) n_prompt = 1;

    for (std::uint32_t pi = 0; pi < n_prompt; ++pi) {
        std::uint32_t tid = prompt_ids[pi];
        std::cout << ciot::tokenizer_decode(&tok, tid) << " ";
        for (std::uint32_t i = 0; i < cfg.dim; ++i) embed_vec[i] = 0.0f;
        embed_vec[tid % cfg.dim] = 1.0f;
        for (std::uint32_t l = 0; l < L; ++l)
            ciot::transformer_block_decode_mha(x, &wq[l], &wk[l], &wv[l], &wo[l], &w1[l], &w2[l],
                norm1 + l * cfg.dim, norm2 + l * cfg.dim, &rope, &caches[l], scratch, cfg.dim, cfg.num_heads, pi, cfg.norm_eps);
    }

    ciot::matmat_ternary_simd(&lm_head, x, 1, embed_vec);
    std::uint32_t last = 0;
    float best = embed_vec[0];
    for (std::uint32_t i = 1; i < cfg.vocab_size && i < cfg.dim; ++i) {
        if (embed_vec[i] > best) { best = embed_vec[i]; last = i; }
    }
    std::cout << "| " << ciot::tokenizer_decode(&tok, last);

    std::cout << " -> ";
    for (std::uint32_t t = 0; t < tokens; ++t) {
        std::uint32_t pos = n_prompt + t;
        for (std::uint32_t l = 0; l < L; ++l)
            ciot::transformer_block_decode_mha(x, &wq[l], &wk[l], &wv[l], &wo[l], &w1[l], &w2[l],
                norm1 + l * cfg.dim, norm2 + l * cfg.dim, &rope, &caches[l], scratch, cfg.dim, cfg.num_heads, pos, cfg.norm_eps);
        ciot::matmat_ternary_simd(&lm_head, x, 1, embed_vec);
        best = embed_vec[0]; last = 0;
        for (std::uint32_t i = 1; i < cfg.vocab_size && i < cfg.dim; ++i) {
            if (embed_vec[i] > best) { best = embed_vec[i]; last = i; }
        }
        std::cout << ciot::tokenizer_decode(&tok, last) << (t + 1U == tokens ? "" : " ");
    }
    std::cout << "\n";

    ciot::aligned_free(x); ciot::aligned_free(scratch); ciot::aligned_free(embed_vec);
    ciot::aligned_free(norm1); ciot::aligned_free(norm2);
    for (std::uint32_t l = 0; l < L; ++l) ciot::mha_kv_cache_free(&caches[l]);
    delete[] caches;
    ciot::rope_table_free(&rope);
    ciot::matrix_free(&embed); ciot::matrix_free(&lm_head);
    for (std::uint32_t l = 0; l < L; ++l) {
        ciot::matrix_free(&wq[l]); ciot::matrix_free(&wk[l]); ciot::matrix_free(&wv[l]);
        ciot::matrix_free(&wo[l]); ciot::matrix_free(&w1[l]); ciot::matrix_free(&w2[l]);
    }
    delete[] wq; delete[] wk; delete[] wv; delete[] wo; delete[] w1; delete[] w2;
    ciot::tokenizer_free(&tok);
    return 0;
}

int run_generate_bits(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: ciot --generate-bits <bits> <vocab> [tokens]\n";
        return 1;
    }
    const std::uint32_t tokens = argc > 4 ? parse_u32(argv[4], 24U) : 24U;

    ciot::TernaryMatrix transition;
    if (!ciot::matrix_load_bits(&transition, argv[2])) {
        std::cerr << "failed to load .bits file: " << argv[2] << "\n";
        return 1;
    }
    if (transition.rows != transition.cols) {
        std::cerr << "generation matrix must be square\n";
        ciot::matrix_free(&transition);
        return 1;
    }

    std::ifstream vocab_file(argv[3]);
    if (!vocab_file) {
        std::cerr << "failed to load vocab file: " << argv[3] << "\n";
        ciot::matrix_free(&transition);
        return 1;
    }

    std::vector<std::string> vocab;
    std::string word;
    while (std::getline(vocab_file, word)) {
        if (!word.empty()) {
            vocab.push_back(word);
        }
    }
    if (vocab.size() != transition.rows) {
        std::cerr << "vocab size does not match matrix rows\n";
        ciot::matrix_free(&transition);
        return 1;
    }

    float* state = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(transition.cols) * sizeof(float)));
    float* logits = static_cast<float*>(ciot::aligned_malloc(static_cast<std::size_t>(transition.rows) * sizeof(float)));
    if (!state || !logits) {
        ciot::aligned_free(state);
        ciot::aligned_free(logits);
        ciot::matrix_free(&transition);
        return 1;
    }

    std::uint32_t current = 0U;
    std::cout << "trained bits smoke using " << ciot::simd_backend_name() << ": ";
    for (std::uint32_t t = 0; t < tokens; ++t) {
        std::cout << vocab[current] << (t + 1U == tokens ? "" : " ");
        for (std::uint32_t i = 0; i < transition.cols; ++i) {
            state[i] = i == current ? 1.0f : 0.0f;
        }
        ciot::matvec_ternary_simd(&transition, state, logits);
        std::uint32_t best = 0U;
        for (std::uint32_t i = 1; i < transition.rows; ++i) {
            best = logits[i] > logits[best] ? i : best;
        }
        current = best;
    }
    std::cout << "\n";

    ciot::aligned_free(state);
    ciot::aligned_free(logits);
    ciot::matrix_free(&transition);
    return 0;
}

int run_tiny_generate(int argc, char** argv) {
    const std::uint32_t tokens = argc > 2 ? parse_u32(argv[2], 12U) : 12U;
    constexpr std::uint32_t vocab = 8U;
    const char* words[vocab] = {"ciot", "packs", "ternary", "bits", "on", "neon", "very", "fast"};
    const std::uint32_t next[vocab] = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 0U};

    ciot::TernaryMatrix transition;
    if (!ciot::matrix_init(&transition, vocab, vocab)) {
        return 1;
    }
    for (std::uint32_t c = 0; c < vocab; ++c) {
        ciot::matrix_set_ternary(&transition, next[c], c, 1);
        transition.row_scale[c] = 4.0f;
    }

    float state[vocab] = {};
    float logits[vocab] = {};
    state[0] = 1.0f;

    std::cout << "tiny text smoke using " << ciot::simd_backend_name() << ": ";
    std::uint32_t current = 0U;
    for (std::uint32_t t = 0; t < tokens; ++t) {
        std::cout << words[current] << (t + 1U == tokens ? "" : " ");
        for (std::uint32_t i = 0; i < vocab; ++i) {
            state[i] = i == current ? 1.0f : 0.0f;
        }
        ciot::matvec_ternary_simd(&transition, state, logits);
        ciot::softmax_inplace(logits, vocab);
        std::uint32_t best = 0U;
        for (std::uint32_t i = 1; i < vocab; ++i) {
            best = logits[i] > logits[best] ? i : best;
        }
        current = best;
    }
    std::cout << "\n";

    ciot::matrix_free(&transition);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc <= 1) {
        print_help();
        return 0;
    }

    if (std::strcmp(argv[1], "--backend") == 0) {
        std::cout << ciot::simd_backend_name() << "\n";
        return 0;
    }

    if (std::strcmp(argv[1], "--simd-test") == 0) {
        return run_simd_test();
    }

    if (std::strcmp(argv[1], "--bench-linear") == 0) {
        return run_bench_linear(argc, argv);
    }

    if (std::strcmp(argv[1], "--bench-linear-pro") == 0) {
        return run_bench_linear_pro_cmd(argc, argv);
    }

    if (std::strcmp(argv[1], "--bench-suite") == 0) {
        return run_bench_suite();
    }

    if (std::strcmp(argv[1], "--bench-bits") == 0) {
        return run_bench_loaded_bits(argc, argv);
    }

    if (std::strcmp(argv[1], "--bench-batch") == 0) {
        return run_bench_batch(argc, argv);
    }

    if (std::strcmp(argv[1], "--bench-transformer") == 0) {
        return run_bench_transformer(argc, argv);
    }

    if (std::strcmp(argv[1], "--bench-rope") == 0) {
        return run_bench_rope(argc, argv);
    }

    if (std::strcmp(argv[1], "--bench-decode") == 0) {
        return run_bench_decode(argc, argv);
    }

    if (std::strcmp(argv[1], "--bench-decode-mha") == 0) {
        return run_bench_decode_mha(argc, argv);
    }

    if (std::strcmp(argv[1], "--decode-generate") == 0) {
        return run_decode_generate(argc, argv);
    }

    if (std::strcmp(argv[1], "--generate-bits") == 0) {
        return run_generate_bits(argc, argv);
    }

    if (std::strcmp(argv[1], "--model-generate") == 0) {
        return run_model_generate(argc, argv);
    }

    if (std::strcmp(argv[1], "--tiny-generate") == 0) {
        return run_tiny_generate(argc, argv);
    }

    print_help();
    return 1;
}

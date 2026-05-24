#include "../../include/Ciot.h"

#include <cstdlib>
#include <cstring>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace ciot {
namespace {

enum class DispatchBackend : std::uint32_t {
    Auto = 0,
    Scalar = 1,
    Native = 2,
};

DispatchBackend selected_backend() {
    static DispatchBackend selected = []() {
        const char* forced = std::getenv("CIOT_BACKEND");
        if (forced && std::strcmp(forced, "scalar") == 0) {
            return DispatchBackend::Scalar;
        }
        return DispatchBackend::Native;
    }();
    return selected;
}

#if defined(__AVX512F__)

void matvec_ternary_native(const TernaryMatrix* matrix, const float* x, float* y) {
    if (!matrix || !x || !y) return;
    const std::uint32_t full_blocks = matrix->cols >> 6U;
    const std::uint32_t tail_cols = matrix->cols & 63U;
    for (std::uint32_t r = 0; r < matrix->rows; ++r) {
        __m512 acc_pos = _mm512_setzero_ps();
        __m512 acc_neg = _mm512_setzero_ps();
        const std::size_t row_base = static_cast<std::size_t>(r) * matrix->blocks64;
        for (std::uint32_t b = 0; b < full_blocks; ++b) {
            const std::uint64_t pos = matrix->pos_bits[row_base + b];
            const std::uint64_t neg = matrix->neg_bits[row_base + b];
            const std::uint32_t base = b << 6U;
            acc_pos = _mm512_add_ps(acc_pos, _mm512_maskz_load_ps(static_cast<__mmask16>((pos >> 0U) & 0xFFFFU), x + base + 0U));
            acc_neg = _mm512_add_ps(acc_neg, _mm512_maskz_load_ps(static_cast<__mmask16>((neg >> 0U) & 0xFFFFU), x + base + 0U));
            acc_pos = _mm512_add_ps(acc_pos, _mm512_maskz_load_ps(static_cast<__mmask16>((pos >> 16U) & 0xFFFFU), x + base + 16U));
            acc_neg = _mm512_add_ps(acc_neg, _mm512_maskz_load_ps(static_cast<__mmask16>((neg >> 16U) & 0xFFFFU), x + base + 16U));
            acc_pos = _mm512_add_ps(acc_pos, _mm512_maskz_load_ps(static_cast<__mmask16>((pos >> 32U) & 0xFFFFU), x + base + 32U));
            acc_neg = _mm512_add_ps(acc_neg, _mm512_maskz_load_ps(static_cast<__mmask16>((neg >> 32U) & 0xFFFFU), x + base + 32U));
            acc_pos = _mm512_add_ps(acc_pos, _mm512_maskz_load_ps(static_cast<__mmask16>((pos >> 48U) & 0xFFFFU), x + base + 48U));
            acc_neg = _mm512_add_ps(acc_neg, _mm512_maskz_load_ps(static_cast<__mmask16>((neg >> 48U) & 0xFFFFU), x + base + 48U));
        }
        float sum = _mm512_reduce_add_ps(_mm512_sub_ps(acc_pos, acc_neg));
        if (tail_cols != 0U) {
            const std::uint32_t b = full_blocks; const std::uint64_t pos = matrix->pos_bits[row_base + b];
            const std::uint64_t neg = matrix->neg_bits[row_base + b]; const std::uint32_t base = b << 6U;
            for (std::uint32_t bit = 0; bit < tail_cols; ++bit) {
                sum += static_cast<float>(static_cast<int>((pos>>bit)&1ULL)-static_cast<int>((neg>>bit)&1ULL))*x[base+bit];
            }
        }
        y[r] = sum * matrix->row_scale[r];
    }
}

#elif defined(__AVX2__)

static inline float hsum256_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline __m256 masked_load8(const float* x, std::uint32_t offset, std::uint32_t bits, __m256i lane_bits, __m256i zero) {
    const __m256 values = _mm256_load_ps(x + offset);
    const __m256i word = _mm256_set1_epi32(static_cast<int>(bits));
    const __m256i active = _mm256_cmpgt_epi32(_mm256_and_si256(word, lane_bits), zero);
    return _mm256_and_ps(values, _mm256_castsi256_ps(active));
}

void matvec_ternary_native(const TernaryMatrix* matrix, const float* x, float* y) {
    if (!matrix || !x || !y) return;
    const std::uint32_t full_blocks = matrix->cols >> 6U;
    const std::uint32_t tail_cols = matrix->cols & 63U;
    const __m256i lane_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i zero = _mm256_setzero_si256();
    for (std::uint32_t r = 0; r < matrix->rows; ++r) {
        __m256 acc_pos = _mm256_setzero_ps();
        __m256 acc_neg = _mm256_setzero_ps();
        const std::size_t row_base = static_cast<std::size_t>(r) * matrix->blocks64;
        for (std::uint32_t b = 0; b < full_blocks; ++b) {
            const std::uint64_t pos = matrix->pos_bits[row_base + b];
            const std::uint64_t neg = matrix->neg_bits[row_base + b];
            const std::uint32_t base = b << 6U;
            for (std::uint32_t off = 0; off < 64U; off += 8U) {
                acc_pos = _mm256_add_ps(acc_pos, masked_load8(x, base+off, static_cast<std::uint32_t>((pos>>off)&0xFFU), lane_bits, zero));
                acc_neg = _mm256_add_ps(acc_neg, masked_load8(x, base+off, static_cast<std::uint32_t>((neg>>off)&0xFFU), lane_bits, zero));
            }
        }
        float sum = hsum256_ps(_mm256_sub_ps(acc_pos, acc_neg));
        if (tail_cols != 0U) {
            const std::uint32_t b = full_blocks; const std::uint64_t pos = matrix->pos_bits[row_base + b];
            const std::uint64_t neg = matrix->neg_bits[row_base + b]; const std::uint32_t base = b << 6U;
            for (std::uint32_t bit = 0; bit < tail_cols; ++bit) {
                sum += static_cast<float>(static_cast<int>((pos>>bit)&1ULL)-static_cast<int>((neg>>bit)&1ULL))*x[base+bit];
            }
        }
        y[r] = sum * matrix->row_scale[r];
    }
}

#elif defined(__ARM_NEON)

static inline float hsum128_f32(float32x4_t v) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return vaddvq_f32(v);
#else
    const float32x2_t pair = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    const float32x2_t sum = vpadd_f32(pair, pair);
    return vget_lane_f32(sum, 0);
#endif
}

static inline float32x4_t masked_load4(const float* x, std::uint32_t offset, std::uint32_t bits, uint32x4_t lane_bits, uint32x4_t zero_u32, float32x4_t zero_f32) {
    const float32x4_t values = vld1q_f32(x + offset);
    const uint32x4_t word = vdupq_n_u32(bits);
    const uint32x4_t active = vcgtq_u32(vandq_u32(word, lane_bits), zero_u32);
    return vbslq_f32(active, values, zero_f32);
}

void matvec_ternary_native(const TernaryMatrix* matrix, const float* x, float* y) {
    if (!matrix || !x || !y) return;
    const std::uint32_t full_blocks = matrix->cols >> 6U;
    const std::uint32_t tail_cols = matrix->cols & 63U;
    const std::uint32_t lane_data[4] = {1U, 2U, 4U, 8U};
    const uint32x4_t lane_bits = vld1q_u32(lane_data);
    const uint32x4_t zero_u32 = vdupq_n_u32(0U);
    const float32x4_t zero_f32 = vdupq_n_f32(0.0f);
    for (std::uint32_t r = 0; r < matrix->rows; ++r) {
        float32x4_t acc_pos = vdupq_n_f32(0.0f);
        float32x4_t acc_neg = vdupq_n_f32(0.0f);
        const std::size_t row_base = static_cast<std::size_t>(r) * matrix->blocks64;
        for (std::uint32_t b = 0; b < full_blocks; ++b) {
            const std::uint64_t pos = matrix->pos_bits[row_base + b];
            const std::uint64_t neg = matrix->neg_bits[row_base + b];
            const std::uint32_t base = b << 6U;
            for (std::uint32_t chunk = 0; chunk < 64U; chunk += 4U) {
                acc_pos = vaddq_f32(acc_pos, masked_load4(x, base+chunk, static_cast<std::uint32_t>((pos>>chunk)&0xFU), lane_bits, zero_u32, zero_f32));
                acc_neg = vaddq_f32(acc_neg, masked_load4(x, base+chunk, static_cast<std::uint32_t>((neg>>chunk)&0xFU), lane_bits, zero_u32, zero_f32));
            }
        }
        float sum = hsum128_f32(vsubq_f32(acc_pos, acc_neg));
        if (tail_cols != 0U) {
            const std::uint32_t b = full_blocks; const std::uint64_t pos = matrix->pos_bits[row_base + b];
            const std::uint64_t neg = matrix->neg_bits[row_base + b]; const std::uint32_t base = b << 6U;
            for (std::uint32_t bit = 0; bit < tail_cols; ++bit) {
                sum += static_cast<float>(static_cast<int>((pos>>bit)&1ULL)-static_cast<int>((neg>>bit)&1ULL))*x[base+bit];
            }
        }
        y[r] = sum * matrix->row_scale[r];
    }
}

#else

void matvec_ternary_native(const TernaryMatrix* matrix, const float* x, float* y) {
    matvec_ternary_ref(matrix, x, y);
}

#endif

const char* native_backend_name() {
#if defined(__AVX512F__)
    return "AVX-512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__ARM_NEON)
    return "ARM NEON";
#else
    return "scalar";
#endif
}

} // namespace

void matmat_ternary_simd(const TernaryMatrix* matrix, const float* x, std::uint32_t batch, float* y) {
    if (!matrix || !x || !y || batch == 0) return;
    for (std::uint32_t b = 0; b < batch; ++b)
        matvec_ternary_simd(matrix, x + static_cast<std::size_t>(b) * matrix->cols, y + static_cast<std::size_t>(b) * matrix->rows);
}

void matvec_ternary_simd(const TernaryMatrix* matrix, const float* x, float* y) {
    if (selected_backend() == DispatchBackend::Scalar) {
        matvec_ternary_ref(matrix, x, y);
        return;
    }
    matvec_ternary_native(matrix, x, y);
}

void matvec_ternary_avx2(const TernaryMatrix* matrix, const float* x, float* y) {
    matvec_ternary_simd(matrix, x, y);
}

const char* simd_backend_name() {
    if (selected_backend() == DispatchBackend::Scalar) return "scalar-forced";
    return native_backend_name();
}

} // namespace ciot

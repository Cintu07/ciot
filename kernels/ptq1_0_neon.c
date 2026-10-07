#include "ptq1_0_neon.h"

#include <arm_neon.h>
#include <string.h>

static inline float fp16_to_f32(uint16_t h) {
    __fp16 f;
    memcpy(&f, &h, sizeof(f));
    return (float) f;
}

// v already holds byte * 3^nn (mod 256); returns the 16 weights in {-1, 0, +1}.
static inline int8x16_t ptq1_0_trits(uint8x16_t v) {
    const int8x16_t ge1 = vreinterpretq_s8_u8(vcgeq_u8(v, vdupq_n_u8(86)));
    const int8x16_t ge2 = vreinterpretq_s8_u8(vcgeq_u8(v, vdupq_n_u8(171)));
    return vsubq_s8(vsubq_s8(vdupq_n_s8(-1), ge1), ge2);
}

void ciot_vec_dot_ptq1_0_q8_0_neon(int n, float* s, const void* vx, const void* vy) {
    const ciot_block_ptq1_0* x = (const ciot_block_ptq1_0*) vx;
    const ciot_block_q8_0* y = (const ciot_block_q8_0*) vy;
    const int nb = n / CIOT_QK_PTQ1_0;

    // Lane multipliers for the 8-byte stage: low half is trit nn, high half is trit nn + 1.
    static const uint8_t k_pow_1_3[16] = {1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3};
    static const uint8_t k_pow_9_27[16] = {9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27};
    static const uint8_t k_pow_qh[8] = {1, 1, 3, 3, 9, 9, 27, 27};
    const uint8x16_t pow_1_3 = vld1q_u8(k_pow_1_3);
    const uint8x16_t pow_9_27 = vld1q_u8(k_pow_9_27);
    const uint8x8_t pow_qh = vld1_u8(k_pow_qh);
    const int32x4_t zero = vdupq_n_s32(0);

    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; ++i) {
        const ciot_block_q8_0* yb = y + 4 * i;

        const uint8x16_t q16 = vld1q_u8(x[i].qs);
        const uint8x8_t q8 = vld1_u8(x[i].qs + 16);
        const uint8x16_t q8x2 = vcombine_u8(q8, q8);
        uint16_t qh_pair;
        memcpy(&qh_pair, x[i].qh, sizeof(qh_pair));
        const uint8x8_t qh = vreinterpret_u8_u16(vdup_n_u16(qh_pair)); // h0 h1 h0 h1 ...

        // Values 0..79 come from bytes 0..15, one trit position per 16-lane vector.
        int32x4_t a0 = vdotq_s32(zero, ptq1_0_trits(q16), vld1q_s8(yb[0].qs));
        a0 = vdotq_s32(a0, ptq1_0_trits(vmulq_u8(q16, vdupq_n_u8(3))), vld1q_s8(yb[0].qs + 16));
        int32x4_t a1 = vdotq_s32(zero, ptq1_0_trits(vmulq_u8(q16, vdupq_n_u8(9))), vld1q_s8(yb[1].qs));
        a1 = vdotq_s32(a1, ptq1_0_trits(vmulq_u8(q16, vdupq_n_u8(27))), vld1q_s8(yb[1].qs + 16));
        int32x4_t a2 = vdotq_s32(zero, ptq1_0_trits(vmulq_u8(q16, vdupq_n_u8(81))), vld1q_s8(yb[2].qs));
        // Values 80..111 come from bytes 16..23, two trit positions per vector.
        a2 = vdotq_s32(a2, ptq1_0_trits(vmulq_u8(q8x2, pow_1_3)), vld1q_s8(yb[2].qs + 16));
        int32x4_t a3 = vdotq_s32(zero, ptq1_0_trits(vmulq_u8(q8x2, pow_9_27)), vld1q_s8(yb[3].qs));
        // Values 112..119 are the last trit of bytes 16..23; 120..127 are the qh trits.
        const uint8x16_t tail = vcombine_u8(vmul_u8(q8, vdup_n_u8(81)), vmul_u8(qh, pow_qh));
        a3 = vdotq_s32(a3, ptq1_0_trits(tail), vld1q_s8(yb[3].qs + 16));

        // Lane k = integer dot product for activation block k.
        const int32x4_t sums = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
        const uint16_t dy[4] = {yb[0].d, yb[1].d, yb[2].d, yb[3].d};
        const float32x4_t scale = vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(dy))), fp16_to_f32(x[i].d));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(sums), scale);
    }

    *s = vaddvq_f32(acc);
}

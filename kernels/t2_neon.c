#include "t2_neon.h"

#include <arm_neon.h>
#include <string.h>

static inline uint16_t f32_to_fp16(float v) {
    __fp16 h = (__fp16) v;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

void ciot_t2x4_pack(const int8_t* w[4], const float scale[4], ciot_block_t2x4* out) {
    for (int r = 0; r < 4; ++r) {
        out->d[r] = f32_to_fp16(scale[r]);
        for (int g = 0; g < 2; ++g) {
            for (int j = 0; j < 16; ++j) {
                uint8_t byte = 0;
                for (int s = 0; s < 4; ++s) byte |= (uint8_t) ((w[r][64 * g + j + 16 * s] + 1) << (2 * s));
                out->qs[g][r][j] = byte;
            }
        }
    }
}

void ciot_gemv_t2x4_q8_K_neon(int n, float* s, const ciot_block_t2x4* x, const ciot_block_q8_K* y) {
    const int nb = n / CIOT_QK_T2;
    const uint8x16_t mask_s0 = vdupq_n_u8(0x03);
    const uint8x16_t mask_s1 = vdupq_n_u8(0x0C);
    const uint8x16_t mask_s2 = vdupq_n_u8(0x30);

    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; ++i) {
        const ciot_block_q8_K* yb = y + (i >> 1);
        const int8_t* q8 = yb->qs + 128 * (i & 1);
        const int32_t ysum = vaddlvq_s16(vld1q_s16(yb->bsums + 8 * (i & 1)));

        int8x16_t yv[2][4];
        for (int g = 0; g < 2; ++g)
            for (int sl = 0; sl < 4; ++sl) yv[g][sl] = vld1q_s8(q8 + 64 * g + 16 * sl);

        // Slots 1 and 2 are masked in place, so their dots come out scaled by 4 and 16.
        // Keeping three accumulators and recombining once per block avoids a shift per weight vector.
        int32x4_t row_sum[4];
        for (int r = 0; r < 4; ++r) {
            int32x4_t a1 = vdupq_n_s32(0), a4 = vdupq_n_s32(0), a16 = vdupq_n_s32(0);
            for (int g = 0; g < 2; ++g) {
                const uint8x16_t v = vld1q_u8(x[i].qs[g][r]);
                a1 = vdotq_s32(a1, vreinterpretq_s8_u8(vandq_u8(v, mask_s0)), yv[g][0]);
                a4 = vdotq_s32(a4, vreinterpretq_s8_u8(vandq_u8(v, mask_s1)), yv[g][1]);
                a16 = vdotq_s32(a16, vreinterpretq_s8_u8(vandq_u8(v, mask_s2)), yv[g][2]);
                a1 = vdotq_s32(a1, vreinterpretq_s8_u8(vshrq_n_u8(v, 6)), yv[g][3]);
            }
            row_sum[r] = vmlaq_n_s32(vmlaq_n_s32(a16, a4, 4), a1, 16); // 16 * (u . y)
        }

        // Lane r = 16 * (u . y) for row r; subtracting 16 * sum(y) turns u = w + 1 back into w.
        int32x4_t tot = vpaddq_s32(vpaddq_s32(row_sum[0], row_sum[1]), vpaddq_s32(row_sum[2], row_sum[3]));
        tot = vsubq_s32(tot, vdupq_n_s32(16 * ysum));

        const float32x4_t dw = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x[i].d)));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(tot), vmulq_n_f32(dw, yb->d * (1.0f / 16.0f)));
    }

    vst1q_f32(s, acc);
}

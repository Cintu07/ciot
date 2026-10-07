// T2 x Q8_Kx4 matrix multiply with SMMLA (AArch64 i8mm), for prompt processing and for
// verifying several speculative tokens in one pass over the weights.
//
// Tile: 4 weight rows (one ciot_block_t2x4) x 4 activation columns (one ciot_block_q8_Kx4).
// SMMLA multiplies a 2x8 int8 tile by an 8x2 int8 tile into a 2x2 int32 tile:
//   - the activation side is already laid out for it: Q8_Kx4 interleaves the 4 columns in
//     8-byte chunks, so 16 bytes at qs + 32c hold column 0 and 1 of chunk c;
//   - the weight side comes from decoding T2 slots (16 consecutive weights per row) and zipping
//     the 64-bit halves of two rows together.
// Decode work per weight is shared by all 4 columns, which is the point: a single-token matvec
// pays it once per token, this pays it once per 4 tokens.
#include "t2_neon.h"

#include <arm_neon.h>
#include <string.h>

static inline int8x16_t t2_slot(uint8x16_t v, int slot) {
    switch (slot) {
    case 0: return vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(3)));
    case 1: return vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 2), vdupq_n_u8(3)));
    case 2: return vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), vdupq_n_u8(3)));
    default: return vreinterpretq_s8_u8(vshrq_n_u8(v, 6));
    }
}

static inline int8x16_t zip_lo(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_u64(vzip1q_u64(vreinterpretq_u64_s8(a), vreinterpretq_u64_s8(b)));
}

static inline int8x16_t zip_hi(int8x16_t a, int8x16_t b) {
    return vreinterpretq_s8_u64(vzip2q_u64(vreinterpretq_u64_s8(a), vreinterpretq_u64_s8(b)));
}

void ciot_gemm_t2x4_q8_Kx4_i8mm(int n, float* s, size_t bs, const ciot_block_t2x4* x, const ciot_block_q8_Kx4* y,
                                int nr, int nc, int32_t* ysum_scratch) {
    const int nb = n / CIOT_QK_T2;
    const int nbk = n / CIOT_QK_K;
    const int8x16_t ones = vdupq_n_s8(1);

    for (int yg = 0; yg < nr / 4; ++yg) {
        const ciot_block_q8_Kx4* yb = y + (size_t) yg * nbk;

        // Column sums per 128-weight block, shared by every weight row: u = w + 1, so
        // w . y = u . y - sum(y).
        for (int i = 0; i < nb; ++i) {
            const int8_t* q = yb[i >> 1].qs + 512 * (i & 1);
            int32x4_t s01 = vdupq_n_s32(0), s23 = vdupq_n_s32(0);
            for (int c = 0; c < 16; ++c) {
                s01 = vdotq_s32(s01, vld1q_s8(q + 32 * c), ones);
                s23 = vdotq_s32(s23, vld1q_s8(q + 32 * c + 16), ones);
            }
            vst1q_s32(ysum_scratch + 4 * i, vpaddq_s32(s01, s23)); // columns 0..3
        }

        for (int xg = 0; xg < nc / 4; ++xg) {
            const ciot_block_t2x4* xb = x + (size_t) xg * nb;
            float32x4_t acc[2][2];
            for (int p = 0; p < 2; ++p)
                for (int q = 0; q < 2; ++q) acc[p][q] = vdupq_n_f32(0.0f);

            for (int i = 0; i < nb; ++i) {
                const ciot_block_q8_Kx4* yk = yb + (i >> 1);
                const int8_t* q8 = yk->qs + 512 * (i & 1);

                // m[p][q] lanes: (row 2p, col 2q), (2p, 2q+1), (2p+1, 2q), (2p+1, 2q+1)
                int32x4_t m[2][2];
                for (int p = 0; p < 2; ++p)
                    for (int q = 0; q < 2; ++q) m[p][q] = vdupq_n_s32(0);

                for (int g = 0; g < 2; ++g) {
                    uint8x16_t v[4];
                    for (int r = 0; r < 4; ++r) v[r] = vld1q_u8(xb[i].qs[g][r]);
                    for (int sl = 0; sl < 4; ++sl) {
                        const int8x16_t d0 = t2_slot(v[0], sl), d1 = t2_slot(v[1], sl);
                        const int8x16_t d2 = t2_slot(v[2], sl), d3 = t2_slot(v[3], sl);
                        // Slot sl of group g is weights 64g + 16sl .. +15, i.e. 8-byte chunks kc, kc + 1.
                        const int kc = 8 * g + 2 * sl;
                        const int8_t* qc = q8 + 32 * kc;
                        const int8x16_t b01 = vld1q_s8(qc), b23 = vld1q_s8(qc + 16);
                        const int8x16_t b01h = vld1q_s8(qc + 32), b23h = vld1q_s8(qc + 48);
                        const int8x16_t a01 = zip_lo(d0, d1), a01h = zip_hi(d0, d1);
                        const int8x16_t a23 = zip_lo(d2, d3), a23h = zip_hi(d2, d3);
                        m[0][0] = vmmlaq_s32(vmmlaq_s32(m[0][0], a01, b01), a01h, b01h);
                        m[0][1] = vmmlaq_s32(vmmlaq_s32(m[0][1], a01, b23), a01h, b23h);
                        m[1][0] = vmmlaq_s32(vmmlaq_s32(m[1][0], a23, b01), a23h, b01h);
                        m[1][1] = vmmlaq_s32(vmmlaq_s32(m[1][1], a23, b23), a23h, b23h);
                    }
                }

                const float32x4_t dw = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(xb[i].d)));
                const float32x4_t dy = vld1q_f32(yk->d);
                const int32x4_t ys = vld1q_s32(ysum_scratch + 4 * i);
                const float32x4_t rows[2] = {vzip1q_f32(dw, dw), vzip2q_f32(dw, dw)};
                const float32x4_t cols[2] = {vcombine_f32(vget_low_f32(dy), vget_low_f32(dy)),
                                             vcombine_f32(vget_high_f32(dy), vget_high_f32(dy))};
                const int32x4_t ysq[2] = {vcombine_s32(vget_low_s32(ys), vget_low_s32(ys)),
                                          vcombine_s32(vget_high_s32(ys), vget_high_s32(ys))};
                for (int p = 0; p < 2; ++p)
                    for (int q = 0; q < 2; ++q)
                        acc[p][q] = vfmaq_f32(acc[p][q], vcvtq_f32_s32(vsubq_s32(m[p][q], ysq[q])), vmulq_f32(rows[p], cols[q]));
            }

            for (int p = 0; p < 2; ++p)
                for (int q = 0; q < 2; ++q) {
                    float out[4];
                    vst1q_f32(out, acc[p][q]);
                    for (int l = 0; l < 4; ++l) {
                        const int row = 2 * p + (l >> 1), col = 2 * q + (l & 1);
                        s[(size_t) (4 * yg + col) * bs + 4 * xg + row] = out[l];
                    }
                }
        }
    }
}

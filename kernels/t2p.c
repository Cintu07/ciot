#include "t2p.h"

#include <arm_neon.h>
#include <string.h>

static inline uint16_t f32_to_fp16(float v) {
    __fp16 h = (__fp16) v;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

void ciot_t2p_pack(const int8_t* w[4], const float scale[4], ciot_block_t2p* out) {
    for (int r = 0; r < 4; ++r) out->d[r] = f32_to_fp16(scale[r]);
    for (int p = 0; p < 2; ++p) {
        for (int wd = 0; wd < 4; ++wd) {
            for (int j = 0; j < 16; ++j) {
                const int row = 2 * p + (j >= 8);
                uint8_t byte = 0;
                for (int s = 0; s < 4; ++s) {
                    const int k = 8 * (4 * wd + s) + (j & 7);
                    byte |= (uint8_t) ((w[row][k] + 1) << (2 * s));
                }
                out->qs[p][wd][j] = byte;
            }
        }
    }
}

// Slots 1 and 2 are masked in place (values scaled by 4 and 16) and accumulate separately;
// one recombine per block replaces a shift per vector.
#define T2P_SLOT0(v) vreinterpretq_s8_u8(vandq_u8((v), vdupq_n_u8(0x03)))
#define T2P_SLOT1(v) vreinterpretq_s8_u8(vandq_u8((v), vdupq_n_u8(0x0C)))
#define T2P_SLOT2(v) vreinterpretq_s8_u8(vandq_u8((v), vdupq_n_u8(0x30)))
#define T2P_SLOT3(v) vreinterpretq_s8_u8(vshrq_n_u8((v), 6))

static inline int8x16_t dup8(const int8_t* p) {
    return vreinterpretq_s8_u64(vld1q_dup_u64((const uint64_t*) p));
}

void ciot_gemv_t2p_q8_K(int n, float* s, const ciot_block_t2p* x, const ciot_block_q8_K* y) {
    const int nb = n / 128;
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; ++i) {
        const ciot_block_q8_K* yb = y + (i >> 1);
        const int8_t* q8 = yb->qs + 128 * (i & 1);
        const int32_t ysum = vaddlvq_s16(vld1q_s16(yb->bsums + 8 * (i & 1)));

        // per pair, one accumulator per slot (8 independent SDOT chains of length 4);
        // lanes 0-1 belong to row 2p, lanes 2-3 to row 2p+1
        int32x4_t a[2][4];
        for (int p = 0; p < 2; ++p)
            for (int sl = 0; sl < 4; ++sl) a[p][sl] = vdupq_n_s32(0);
        for (int wd = 0; wd < 4; ++wd) {
            const int8x16_t y0 = dup8(q8 + 8 * (4 * wd + 0));
            const int8x16_t y1 = dup8(q8 + 8 * (4 * wd + 1));
            const int8x16_t y2 = dup8(q8 + 8 * (4 * wd + 2));
            const int8x16_t y3 = dup8(q8 + 8 * (4 * wd + 3));
            for (int p = 0; p < 2; ++p) {
                const uint8x16_t v = vld1q_u8(x[i].qs[p][wd]);
                a[p][0] = vdotq_s32(a[p][0], T2P_SLOT0(v), y0);
                a[p][1] = vdotq_s32(a[p][1], T2P_SLOT1(v), y1);
                a[p][2] = vdotq_s32(a[p][2], T2P_SLOT2(v), y2);
                a[p][3] = vdotq_s32(a[p][3], T2P_SLOT3(v), y3);
            }
        }
        const int32x4_t t0 = vmlaq_n_s32(vmlaq_n_s32(a[0][2], a[0][1], 4), vaddq_s32(a[0][0], a[0][3]), 16);
        const int32x4_t t1 = vmlaq_n_s32(vmlaq_n_s32(a[1][2], a[1][1], 4), vaddq_s32(a[1][0], a[1][3]), 16);
        // [row0, row1, row2, row3], each 16 * (u . y); subtracting 16 * sum(y) turns u back into w
        const int32x4_t tot = vsubq_s32(vpaddq_s32(t0, t1), vdupq_n_s32(16 * ysum));

        const float32x4_t dw = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x[i].d)));
        acc = vfmaq_f32(acc, vcvtq_f32_s32(tot), vmulq_n_f32(dw, yb->d * (1.0f / 16.0f)));
    }
    vst1q_f32(s, acc);
}

void ciot_gemm_t2p_q8_Kx4(int n, float* s, size_t bs, const ciot_block_t2p* x, const ciot_block_q8_Kx4* y,
                          int nr, int nc, int32_t* ysum_scratch) {
    const int nb = n / 128;
    const int nbk = n / 256;
    const int8x16_t ones = vdupq_n_s8(1);

    for (int yg = 0; yg < nr / 4; ++yg) {
        const ciot_block_q8_Kx4* yb = y + (size_t) yg * nbk;

        for (int i = 0; i < nb; ++i) {
            const int8_t* q = yb[i >> 1].qs + 512 * (i & 1);
            int32x4_t s01 = vdupq_n_s32(0), s23 = vdupq_n_s32(0);
            for (int c = 0; c < 16; ++c) {
                s01 = vdotq_s32(s01, vld1q_s8(q + 32 * c), ones);
                s23 = vdotq_s32(s23, vld1q_s8(q + 32 * c + 16), ones);
            }
            vst1q_s32(ysum_scratch + 4 * i, vpaddq_s32(s01, s23));
        }

        for (int xg = 0; xg < nc / 4; ++xg) {
            const ciot_block_t2p* xb = x + (size_t) xg * nb;
            float32x4_t acc[2][2];
            for (int p = 0; p < 2; ++p)
                for (int qc = 0; qc < 2; ++qc) acc[p][qc] = vdupq_n_f32(0.0f);

            for (int i = 0; i < nb; ++i) {
                const ciot_block_q8_Kx4* yk = yb + (i >> 1);
                const int8_t* q8 = yk->qs + 512 * (i & 1);

                // [pair][col pair], lanes (row 2p, col 2q), (2p, 2q+1), (2p+1, 2q), (2p+1, 2q+1)
                int32x4_t m1[2][2], m4[2][2], m16[2][2];
                for (int p = 0; p < 2; ++p)
                    for (int qc = 0; qc < 2; ++qc) m1[p][qc] = m4[p][qc] = m16[p][qc] = vdupq_n_s32(0);

                for (int wd = 0; wd < 4; ++wd) {
                    const uint8x16_t v0 = vld1q_u8(xb[i].qs[0][wd]);
                    const uint8x16_t v1 = vld1q_u8(xb[i].qs[1][wd]);
                    const int8_t* qc0 = q8 + 32 * (4 * wd);
#define T2P_MM(acc_, slot_, c_)                                                    \
    {                                                                              \
        const int8x16_t b01 = vld1q_s8(qc0 + 32 * (c_)), b23 = vld1q_s8(qc0 + 32 * (c_) + 16); \
        const int8x16_t w0 = slot_(v0), w1 = slot_(v1);                            \
        acc_[0][0] = vmmlaq_s32(acc_[0][0], w0, b01);                              \
        acc_[0][1] = vmmlaq_s32(acc_[0][1], w0, b23);                              \
        acc_[1][0] = vmmlaq_s32(acc_[1][0], w1, b01);                              \
        acc_[1][1] = vmmlaq_s32(acc_[1][1], w1, b23);                              \
    }
                    T2P_MM(m1, T2P_SLOT0, 0)
                    T2P_MM(m4, T2P_SLOT1, 1)
                    T2P_MM(m16, T2P_SLOT2, 2)
                    T2P_MM(m1, T2P_SLOT3, 3)
#undef T2P_MM
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
                    for (int qc = 0; qc < 2; ++qc) {
                        const int32x4_t t = vmlaq_n_s32(vmlaq_n_s32(m16[p][qc], m4[p][qc], 4), m1[p][qc], 16);
                        const int32x4_t w = vsubq_s32(t, vshlq_n_s32(ysq[qc], 4));
                        acc[p][qc] = vfmaq_f32(acc[p][qc], vcvtq_f32_s32(w), vmulq_n_f32(vmulq_f32(rows[p], cols[qc]), 1.0f / 16.0f));
                    }
            }

            for (int p = 0; p < 2; ++p)
                for (int qc = 0; qc < 2; ++qc) {
                    float out[4];
                    vst1q_f32(out, acc[p][qc]);
                    for (int l = 0; l < 4; ++l)
                        s[(size_t) (4 * yg + 2 * qc + (l & 1)) * bs + 4 * xg + 2 * p + (l >> 1)] = out[l];
                }
        }
    }
}

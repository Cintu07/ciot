/*
 * ciot | ternary bonsai 27b | snapdragon x | 8 cpu threads
 * recorded research runs, stock pq2_0 vs ciot t2p
 *
 *                         stock           ciot
 * prefill                 8.4             31.7 tok/s
 * decode                  6.5              9.7 tok/s
 * code edit speculation   8.88            14.10 tok/s
 *
 * matvec: 107 gb/s | measured dram: 110 to 115 gb/s
 * code edit check: 306 tokens, 0 token mismatches
 *
 * source excerpts from kernels/t2p.c, formatted for display.
 * this file is a screenshot panel, not a standalone kernel.
 */

// one word holds matching eight-weight chunks from two rows.
const int row = 2 * p + (j >= 8);
const int k = 8 * (4 * wd + s) + (j & 7);
byte |= (uint8_t) ((w[row][k] + 1) << (2 * s));

// masks leave slots scaled by 1, 4 or 16; combine once per block.
const int32x4_t t0 = vmlaq_n_s32(
    vmlaq_n_s32(a[0][2], a[0][1], 4),
    vaddq_s32(a[0][0], a[0][3]), 16);
const int32x4_t t1 = vmlaq_n_s32(
    vmlaq_n_s32(a[1][2], a[1][1], 4),
    vaddq_s32(a[1][0], a[1][3]), 16);

// u = w + 1, so remove the activation sum before rescaling.
const int32x4_t tot = vsubq_s32(
    vpaddq_s32(t0, t1), vdupq_n_s32(16 * ysum));

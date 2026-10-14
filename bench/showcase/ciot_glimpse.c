/*
 * ciot, bonsai 27b, snapdragon x, 8 threads. recorded runs.
 * stock pq2_0 -> t2p: prefill 8.4 -> 31.7; decode 6.5 -> 9.7 tok/s
 * code edit: greedy 8.88 -> spec 14.10 tok/s; 306 tokens, 0 mismatches
 * matvec 107 gb/s; dram 110 to 115 gb/s
 * kernels/t2p.c excerpt. display cut only.
 */

const int row = 2 * p + (j >= 8);
const int k = 8 * (4 * wd + s) + (j & 7);
byte |= (uint8_t) ((w[row][k] + 1) << (2 * s));

const int32x4_t t0 = vmlaq_n_s32(
    vmlaq_n_s32(a[0][2], a[0][1], 4),
    vaddq_s32(a[0][0], a[0][3]), 16);
const int32x4_t t1 = vmlaq_n_s32(
    vmlaq_n_s32(a[1][2], a[1][1], 4),
    vaddq_s32(a[1][0], a[1][3]), 16);

// u = w + 1; ditch the extra 16 * sum(y).
const int32x4_t tot = vsubq_s32(
    vpaddq_s32(t0, t1), vdupq_n_s32(16 * ysum));

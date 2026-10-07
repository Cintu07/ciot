// T2: CIOT's compute layout for ternary weights with one fp16 scale per 128 weights.
//
// Same information as PTQ1_0 / PQ2_0 (ternary, group 128), laid out for cheap NEON decode:
//   - each weight is stored as u = w + 1 in 2 bits, so no per-weight subtract is needed;
//     the -1 comes back once per block as -sum(activations), which Q8_K already carries in bsums;
//   - byte j (0..15) of 64-weight group g holds weights 64g + j + 16s in bits 2s..2s+1 (s = 0..3),
//     so one 16-byte load feeds four 16-lane SDOTs with only an AND or a shift per slot;
//   - four rows are interleaved per 128-weight block so they share every activation load.
//
// Storage is 2.125 bits/weight, 21% more than PTQ1_0's 1.75. That trade is deliberate: on this
// CPU the base-3 decode of PTQ1_0 is compute bound well below memory bandwidth.
#pragma once

#include <stddef.h>
#include <stdint.h>

#define CIOT_QK_T2 128
#define CIOT_QK_K 256

// One 128-weight block for 4 consecutive rows.
typedef struct {
    uint16_t d[4];       // fp16 scale per row
    uint8_t qs[2][4][16]; // [group][row][byte]
} ciot_block_t2x4;

// Same as ggml's block_q8_K.
typedef struct {
    float d;
    int8_t qs[CIOT_QK_K];
    int16_t bsums[CIOT_QK_K / 16];
} ciot_block_q8_K;

_Static_assert(sizeof(ciot_block_t2x4) == 136, "t2x4 block must be 136 bytes");
_Static_assert(sizeof(ciot_block_q8_K) == 292, "q8_K block must be 292 bytes");

// Same as ggml's block_q8_Kx4: 4 activation rows, quants interleaved in 8-byte chunks
// (qs[32c + 8r + k] is row r, element 8c + k). bsums are not used by CIOT kernels.
typedef struct {
    float d[4];
    int8_t qs[CIOT_QK_K * 4];
    int16_t bsums[CIOT_QK_K / 4];
} ciot_block_q8_Kx4;

_Static_assert(sizeof(ciot_block_q8_Kx4) == 1168, "q8_Kx4 block must be 1168 bytes");

// Packs 4 rows of 128 ternary weights (values -1, 0, +1) and their scales into one block.
void ciot_t2x4_pack(const int8_t* w[4], const float scale[4], ciot_block_t2x4* out);

// s[0..3] = dot(row r, y) for the 4 interleaved rows. n is the row length (multiple of 256).
void ciot_gemv_t2x4_q8_K_neon(int n, float* s, const ciot_block_t2x4* x, const ciot_block_q8_K* y);

// s[(activation row) * bs + weight row] for nr activation rows (multiple of 4, as Q8_Kx4 groups)
// and nc weight rows (multiple of 4, as T2x4 groups). ysum_scratch holds n / 32 int32s.
// Requires i8mm (kernels/t2_i8mm.c).
void ciot_gemm_t2x4_q8_Kx4_i8mm(int n, float* s, size_t bs, const ciot_block_t2x4* x, const ciot_block_q8_Kx4* y,
                                int nr, int nc, int32_t* ysum_scratch);

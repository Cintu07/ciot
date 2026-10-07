// T2P: T2 rearranged in row pairs so the decoded vectors are SMMLA operands as-is.
//
// Same bits as T2 / PQ2_0: u = w + 1 in 2 bits, one fp16 scale per row per 128 weights,
// 4 rows per 136-byte block. Inside a block, rows are grouped in pairs (0,1) and (2,3), and
// each pair has four 16-byte words. Byte j of word w of pair p holds, in bits 2s..2s+1:
//
//     row 2p + (j >= 8), weight 8 * (4w + s) + (j & 7)
//
// so (word >> 2s) & 3 is the 16-lane vector [row 2p: 8 weights of chunk c | row 2p+1: same
// chunk], c = 4w + s. That is SMMLA's 2x8 operand directly, which removes the two zips per
// slot the T2 i8mm kernel needs. For the single-token kernel, SDOT against the chunk's
// 8 activations duplicated into both halves gives row 2p in lanes 0-1 and row 2p+1 in 2-3.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "t2_neon.h" // ciot_block_q8_K, ciot_block_q8_Kx4

typedef struct {
    uint16_t d[4];        // fp16 scale per row
    uint8_t qs[2][4][16]; // [pair][word][byte]
} ciot_block_t2p;

_Static_assert(sizeof(ciot_block_t2p) == 136, "t2p block must be 136 bytes");

// Packs 4 rows of 128 ternary weights (values -1, 0, +1) and their scales into one block.
void ciot_t2p_pack(const int8_t* w[4], const float scale[4], ciot_block_t2p* out);

// s[0..3] = dot(row r, y) for the 4 rows in x. n is a multiple of 256.
void ciot_gemv_t2p_q8_K(int n, float* s, const ciot_block_t2p* x, const ciot_block_q8_K* y);

// s[(activation row) * bs + weight row] for nr activation rows (multiple of 4, as Q8_Kx4)
// and nc weight rows (multiple of 4). ysum_scratch holds n / 32 int32s. Requires i8mm.
void ciot_gemm_t2p_q8_Kx4(int n, float* s, size_t bs, const ciot_block_t2p* x, const ciot_block_q8_Kx4* y,
                          int nr, int nc, int32_t* ysum_scratch);

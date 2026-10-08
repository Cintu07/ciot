// PTQ1_0 x Q8_0 dot product for AArch64 NEON + dotprod.
//
// PTQ1_0 (PrismML): 128 ternary weights in 28 bytes = 1.75 bits/weight.
//   qs[24]: 5 trits per byte. Bytes 0..15 hold values 0..79 (trit nn of byte m is value 16*nn + m),
//           bytes 16..23 hold values 80..119 (trit nn of byte m is value 80 + 8*nn + m).
//   qh[2]:  4 trits per byte, trit nn of byte h is value 120 + 2*nn + h.
//   d:      fp16 scale.
// A byte stores its trits as a base-3 fixed-point fraction, so trit nn is
// floor(3 * (byte * 3^nn mod 256) / 256), which is 0, 1 or 2 and maps to weight -1, 0, +1.
//
// Decoding trick: floor(3v/256) == (v >= 86) + (v >= 171) for any byte v, so the weight is
// -1 - mask(v >= 86) - mask(v >= 171) where NEON compare masks are 0 or -1. That is one
// multiply, two compares and two subtracts per 16 weights, all on full 16-lane vectors,
// followed by one SDOT against the int8 activations.
#pragma once

#include <stdint.h>

#define CIOT_QK_PTQ1_0 128
#define CIOT_QK8_0 32

typedef struct {
    uint8_t qs[24];
    uint8_t qh[2];
    uint16_t d; // fp16 bits
} ciot_block_ptq1_0;

typedef struct {
    uint16_t d; // fp16 bits
    int8_t qs[CIOT_QK8_0];
} ciot_block_q8_0;

_Static_assert(sizeof(ciot_block_ptq1_0) == 28, "ptq1_0 block must be 28 bytes");
_Static_assert(sizeof(ciot_block_q8_0) == 34, "q8_0 block must be 34 bytes");

void ciot_vec_dot_ptq1_0_q8_0_neon(int n, float* s, const void* vx, const void* vy);

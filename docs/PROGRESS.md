# CIOT-V2 progress

Goal: fastest ternary LLM inference on a laptop CPU, measured honestly against the official runtime.
The first target is Ternary Bonsai 2 27B (prism-ml) on a Snapdragon X X126100 (8 cores, 16 GB).

## Machine limits (measured 2026-10-07)

- Read bandwidth ~110 GB/s, saturating at 4 threads.
- ISA: NEON, dotprod, i8mm, bf16. No SVE.
- Bonsai 2 27B reads ~5.7 GB per token in PTQ1_0 and ~6.8 GB in 2-bit form, so the bandwidth ceiling is ~16–19 tok/s.

## What the official runtime does on this CPU (PrismML-Eng/llama.cpp prism@6bfcd79)

- **PTQ1_0 (5.95 GB file):** `arch-fallback.h` aliases ARM to the generic C dot product. Their NEON attempt was disabled after it measured slower.
- **PQ2_0 (7.21 GB file):** NEON `vec_dot_pq2_0_q8_K`. It costs about 6 vector ops per 16 weights (tbl + shl + and + sub + sdot + load).
- **Weight repack for PQ2_0:** only enabled for AVX-512 VNNI.

## Kernels (bench/ternary_kernels.cpp, 32768x8192 matvec streamed from DRAM, 8 threads)

| kernel | Gweights/s | GB/s | 27B compute-limited tok/s |
|---|---|---|---|
| prism PQ2_0 NEON | 204 | 54 | ~8 |
| ciot PTQ1_0 NEON (kernels/ptq1_0_neon.c) | 175 | 38 | ~7 |
| ciot T2x4 NEON (kernels/t2_neon.c) | 282 | 75 | ~11 |

All kernels match an integer reference exactly. T2 is bit-identical to prism PQ2_0 on scaled inputs.

Two caveats:
- The generic PTQ1_0 loop's speed depends heavily on how clang vectorizes it in a given translation unit. Use the real llama-bench number, not the harness, for "today".
- PTQ1_0's base-3 decode is compute bound here. Storing 21% more bytes (2-bit) for ~2x less decode work wins on this CPU.

T2 layout:
- Each weight is stored as u = w + 1 in 2 bits.
- Byte j of a 64-weight group holds weights j, j+16, j+32 and j+48.
- Each 128-weight block interleaves 4 rows.
- The -1 offset is removed once per block using Q8_K bsums.

## Integration into the official runtime (patches/prism-llama.cpp-t2-neon.patch)

- Adds a load-time repack of PQ2_0 to T2x4 on AArch64 with dotprod.
- The repack is in place: 136 bytes per 4 rows, the same as 4 PQ2_0 rows.
- `CIOT_T2=0` disables it for A/B runs in one binary.

Validated on Qwen3-0.6B requantized to PQ2_0:
- Perplexity is identical to every printed digit with T2 on and off (34558588.7263 over 8x512 tokens).
- 32-token greedy output is byte-identical.
- pp128: 400 -> 578 tok/s.
- tg128: 145-160 -> 171-176 tok/s. A model this small is overhead dominated, so the 27B is the real test.

## Next

1. Bonsai 2 27B end to end: llama-bench for PTQ1_0 stock, PQ2_0 stock and PQ2_0 + T2, plus a perplexity A/B.
2. A proper i8mm prompt-processing gemm instead of de-interleave + gemv.
3. Speculative decoding with recurrent-state rollback (upstream has a `llama-rs-rollback` example; check it before claiming anything new).

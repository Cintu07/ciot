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

The table above spawned threads per matvec. With a persistent spinning pool (bench/ternary_kernels_pool.txt, 8 threads):

| kernel | Gweights/s | GB/s | 27B tok/s ceiling |
|---|---|---|---|
| prism PQ2_0 NEON | 232 | 61.5 | ~9 |
| ciot T2x4 NEON | 391 | 103.7 | ~15 |

T2 reaches ~94% of the measured 110 GB/s, so the matvec itself is now memory bound.
From here, end-to-end speed depends on runtime overhead and the non-matmul ops, not the kernel.

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

## Prompt processing with i8mm (kernels/t2_i8mm.c, bench/t2_gemm.txt)

- SMMLA tile: 4 T2 rows x 4 tokens. Each weight is decoded once per 4 tokens.
- Q8_Kx4's 8-byte interleave is already SMMLA's operand layout.
- Bit-identical to the T2 decode kernel, including with scaled inputs.
- 5120x17408 x 64 tokens, 8 threads: 400 -> 771 GMAC/s (~16 -> ~30 prompt tok/s on the 27B).
- In the runtime on Qwen3-0.6B PQ2_0: pp512 331 -> 837 tok/s, perplexity unchanged to every digit.

## Bonsai 2 27B end to end, PTQ1_0 file (llama-bench tg32, 8 threads, interleaved A/B)

- The model runs on this CPU and answers correctly ("The capital of France is Paris.").
- Stock decode is 2.2–2.7 tok/s at default priority.
- CIOT's PTQ1_0 NEON kernel in arch/arm/quants.c (`CIOT_PTQ1=0` turns it off):
  - Perplexity on 3x128 tokens: 10.0305 vs stock 10.0231 (0.07%, statistical error ±1.96).
  - Prompt pass: 96 s -> 72 s.
- Per-op profiler (`CIOT_PROF=1`, ggml-cpu.c) at default priority:
  - Ternary matmul is only 65% of a token.
  - ~2,400 small ops cost 75–160 µs each (~250 ms per token).
  - Cause: Windows preempting one of 8 spinning workers, so the other 7 wait at the barrier.
- `--prio 2` removes most of that: small ops drop to ~28 ms per token and matmul becomes 89%.

| config (prio 2) | tg32 tok/s |
|---|---|
| stock PTQ1_0 (generic C) | 3.8–4.2 |
| CIOT PTQ1_0 NEON | 5.9–6.0 |

At this point PTQ1_0 matmul runs at ~112 Gweights/s, limited by base-3 decode. That is the case for T2.

## Next

1. Bonsai 2 27B end to end: llama-bench for PTQ1_0 stock, PQ2_0 stock and PQ2_0 + T2, plus a perplexity A/B.
2. A proper i8mm prompt-processing gemm instead of de-interleave + gemv.
3. Speculative decoding with recurrent-state rollback (upstream has a `llama-rs-rollback` example; check it before claiming anything new).

## Bonsai 2 27B end to end, all paths (bench/e2e_bonsai_20261007_202636.txt)

- Setup: 8 threads, `--prio 2`, 2 interleaved rounds, llama-bench pp64 / tg32.
- Results, averaged over the two rounds:

| config | prompt tok/s | decode tok/s |
|---|---|---|
| stock PTQ1_0 (generic C, default ARM path) | 4.6 | 4.35 |
| ciot PTQ1_0 NEON | 7.2 | 4.0 / 6.2 (inconclusive) |
| stock PQ2_0 (PrismML NEON) | 8.4 | 6.5 |
| ciot PQ2_0 + T2 (sdot decode, i8mm prompt) | 27.2 | 9.4 |

- T2 perplexity on the 27B, 3x128 tokens: 9.2430 / 7.3521 / 10.0138 per chunk with T2 on and off, bit-identical.
- Perplexity pass time: 51.4 s -> 17.4 s.

## Profile of T2 decode on the 27B (CIOT_PROF=1, prio 2, mmap off)

- Matmuls take ~92 ms per token, about 74 GB/s.
- bench/decode_replay.cpp replays one token's 401 real-shape matvecs:
  - ggml-style chunking: 70 ms.
  - Static slices: 65.5 ms.
  - Fusing gate+up and q+k+v: 61.5 ms.
- The rest is runtime overhead: about 19 ms of recurrent-state gather and copy ops, plus norms and similar.
- With mmap on, the token-embedding lookup cost ~21 ms per token (page faults next to the repacked copy). `-mmp 0` removes it.
- `--poll 100` made no difference.

## Speculative decoding on the hybrid 27B (examples/ciot-spec in the patch)

How it works:
- Prompt-lookup drafts.
- Verify [next, drafts] in one batch.
- Roll the recurrent state back with llama_memory_seq_rm through the n_rs_seq snapshot ring.

Findings about the official runtime:
- `common_params_speculative::need_n_rs_seq()` only requests the ring for draft-model types, not ngram types. That is probably why KNOWN_ISSUES says ngram has no effect on this model; not verified against their server.
- The ring is filled from the current ubatch only. Upstream's rs-rollback example (one token per decode, rewind 4) therefore fails on the 27B with stock kernels too: 54/64 mismatches, first at the first rollback.
- Speculation only rolls back within the last verify batch, which this design supports.

Correctness checks on the 27B (ciot-spec --check):
- Oracle drafts (the true greedy continuation): 90/90 accepted, no mismatch. Batched verify picks the same tokens as one-token decode.
- Junk drafts (always wrong): 0/354 accepted, no mismatch. Rollback is exact on every step.
- Lookup mode diverged once on raw text, at a near-tie: greedy top-2 margin 0.032 vs 0.0004; every earlier token had at least 0.17. That is float batch-history noise, not corruption.

Runtime fix: leftover verify rows (batch size not a multiple of 4) used to stream the weights once per row. They now go through one zero-padded SMMLA pass. Oracle and junk checks still pass at k=1 and k=2.

Code-edit prompt (bench/prompts/code_edit_chat.txt, thinking pre-closed, 306 tokens, greedy 9.2–9.5 tok/s):

| k | acceptance | speedup | mismatches |
|---|---|---|---|
| 1 | 84% | 1.14x | 0 |
| 2 | 79% | 1.37x | 0 |
| 3 | 75% | 1.38–1.43x | 0 |
| 5 | 65% | 1.23x | 0 |

Limit: a 4-token verify step costs ~1.4x a single-token step, because i8mm at 4 columns is compute bound (~133 ms of math vs a 68 ms memory floor). Making verify cheaper means a faster SMMLA kernel. One option is a row-pair-interleaved T2 layout that needs no zips.

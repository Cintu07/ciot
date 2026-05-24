# Ciot Benchmark Protocol

Ciot avoids heavyweight benchmark dependencies. Production-level benchmarking here means repeatable, documented, low-noise measurements with enough metadata to compare commits and machines.

## What to benchmark

1. **Backend smoke**
   - Command: `./bin/ciot --simd-test`
   - Purpose: prove the selected backend is real: AVX-512/AVX2 on x86/x64, NEON on ARM64, scalar fallback otherwise.

2. **Ternary matvec suite**
   - Command: `./bin/ciot --bench-suite`
   - Sizes: `256x256`, `512x512`, `1024x1024`, `2048x2048`.
   - Metrics: min, median, p95, max, median logical Gop/s, checksum.

3. **Hero number**
   - Command: `./bin/ciot --bench-linear-pro 1024 1024 200 9 20`
   - Purpose: the headline number for a `1024x1024` packed ternary matrix-vector multiply.

4. **Loaded weight benchmark**
   - Create synthetic weights if you do not have a model yet:
     - `python scripts/make_synthetic_weights.py 1024 1024 data/linear_1024.f32`
     - `python scripts/pack_ternary.py data/linear_1024.f32 1024 1024 data/model.bits`
   - Command: `./bin/ciot --bench-bits data/model.bits 200`
   - Purpose: validate the real `.bits` path, not only synthetic in-memory matrices.

5. **Batched matvec benchmark**
   - Command: `./bin/ciot --bench-batch 1024 1024 4 50 5`
   - Purpose: benchmark prompt-style multi-token matvec with row/batch tiling.

6. **Scalar comparison**
   - Command: `CIOT_BACKEND=scalar ./bin/ciot --bench-linear-pro 1024 1024 20 5 5`
   - Purpose: compare the native SIMD backend against the scalar reference path in the same binary.

7. **Transformer block benchmark**
   - Command: `./bin/ciot --bench-transformer 256 50 5 10`
   - Purpose: benchmark a minimal single-token transformer block scaffold with six ternary projections.

8. **RoPE table benchmark**
   - Command: `./bin/ciot --bench-rope 1024 2048 1000 5`
   - Purpose: prove position rotation uses precomputed sin/cos tables instead of hot-path `pow`, `sin`, and `cos`.

9. **Trained `.bits` smoke**
   - Train artifact:
     - `python scripts/train_tiny_bits.py data/tiny_train.txt data/tiny_trained.bits data/tiny_vocab.txt --vocab-limit 16`
   - Command: `./bin/ciot --generate-bits data/tiny_trained.bits data/tiny_vocab.txt 24`
   - Purpose: prove Ciot can load a data-trained `.bits` artifact and emit tokens. This is a transition model, not a trained transformer.

10. **Toy transformer smoke**
   - Command: `./bin/ciot --tiny-generate 16`
   - Purpose: prove the model loop can use ternary matvec + softmax to emit text. This is a toy smoke test, not a trained LLM.

## How to run a publishable benchmark

1. Close browsers, editors, and background downloads.
2. Plug the laptop in.
3. Use the same power mode every time.
4. Run once to warm caches/JIT/emulation layers if any.
5. Run the benchmark script:
   - `python scripts/bench_ciot.py --root .`
6. Commit or attach the generated CSV from `data/`.

## Why these metrics

- **Median**: stable headline latency.
- **p95**: catches thermal throttling and OS scheduling noise.
- **Min**: shows best-case kernel capability.
- **Max**: shows worst observed hiccup.
- **Checksum**: prevents benchmark loops from being optimized away and catches accidental behavior changes.
- **Logical Gop/s**: treats ternary matvec like dense multiply-add work so people can compare across matrix sizes.

## Benchmarks to compare against

There is no perfect standard benchmark for this exact project because Ciot is not BLAS and not a full LLM runtime yet. The closest fair comparisons are:

- Scalar Ciot reference vs Ciot SIMD backend.
- Ciot AVX-512 vs AVX2 vs NEON on similarly sized matrices.
- `llama.cpp` prompt/eval tokens-per-second once Ciot has a real tokenizer/model path.
- BitNet-style ternary/1.58-bit kernels when comparing only linear layers.

Do **not** compare Ciot's tiny matvec directly to full end-to-end LLM tokens/s yet. That would be misleading until tokenizer, KV cache, attention, and full transformer execution are implemented.

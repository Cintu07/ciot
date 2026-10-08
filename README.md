# CIOT-V2

Fast ternary LLM inference on a laptop CPU. The target is [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) (PrismML, Qwen3.8-27B hybrid, weights in {-1, 0, +1}) running on a Snapdragon X laptop: 8 Oryon cores, 16 GB, no GPU.

The kernels plug into [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) as an AArch64 weight repack, so every model and tool in that runtime gets them. Output is bit-identical to the official runtime.

## Results

Bonsai 2 27B, Snapdragon X X126100, 8 threads, `--prio 2`, llama-bench. Every number comes from a log in `bench/`, and `bash bench/showcase.sh` prints them.

| runtime | prefill tok/s | decode tok/s |
|---|---|---|
| official, 1.75-bit file (generic C on ARM) | 4.5 | 4.3 |
| official, 2-bit file (NEON) | 8.4 | 6.5 |
| CIOT T2P (sdot decode, smmla prefill) | 31.7 | 9.7 |

- Speculative decoding (prompt lookup, code-edit prompt): 14.0-14.1 tok/s, against greedy at 8.9-9.4 tok/s in the same runs, with 0 token mismatches.
- Perplexity is identical to the official runtime on every chunk (9.2430 / 7.3521 / 10.0138).
- The decode kernel streams 107 GB/s of weights in a matvec benchmark. This laptop's DRAM measures 110-115 GB/s (`bench/bw.txt`).

## What is in here

- `kernels/t2p.c`: the T2P layout and kernels.
  - Weights are the same 2-bit codes as PQ2_0 (u = w + 1), regrouped in row pairs.
  - A decoded slot is `[row A: 8 weights | row B: same 8]`, which is exactly the operand of `smmla` (2x8 x 8x2 int8 tile). For single-token decode it is one `sdot` covering two rows.
  - Slots are masked in place, so their dots come out scaled by 1, 4 and 16 and go into separate accumulators. One recombine per 128 weights replaces a shift per vector.
  - The +1 offset is removed once per block with the activation sums that Q8_K already carries.
- `kernels/t2_neon.c`, `kernels/t2_i8mm.c`, `kernels/ptq1_0_neon.c`: earlier kernels. `ptq1_0_neon.c` reads the 1.75-bit format directly, using the fact that `floor(3v/256) == (v >= 86) + (v >= 171)` for every byte `v`.
- `patches/pr-arm-pq2_0-repack.patch`: the clean runtime change (one file, `ggml-cpu/repack.cpp`) on top of PrismML `4fbda12`.
- `patches/prism-llama.cpp-t2-neon.patch`: the full research patch. It adds:
  - an A/B switch (`CIOT_T2`) and a per-op profiler (`CIOT_PROF=1`);
  - `examples/ciot-spec`, prompt-lookup speculative decoding for hybrid models, with oracle and junk draft checks that prove the recurrent-state rollback is exact.
- `bench/`:
  - exactness and speed harnesses for every kernel;
  - a replay of one token's 401 real-shape matvecs;
  - memory bandwidth;
  - end-to-end logs;
  - the equal-memory eval (`bench/eval`, GSM8K + HumanEval).
- `kaggle/ciot_equal_memory_eval.ipynb`: the same eval on a free Kaggle T4. Bonsai 2 27B vs Qwen3.8-27B at 2-bit vs Qwen3-8B at Q6_K, all near 7 GB.
- `docs/PROGRESS.md`: the full log of measurements, dead ends and findings.

## Findings along the way

- **Thread priority.** At default Windows priority, about 1/3 of each token was lost to preemption: one of 8 spinning workers gets descheduled and the other 7 wait at the barrier. `--prio 2` alone took decode from 3.8 to 6.0 tok/s.
- **Leftover batch rows.** Speculative verify batches that are not a multiple of 4 used to stream all weights once per leftover row. They now go through one padded `smmla` pass.
- **Why ngram speculation does nothing on Bonsai in the official server.** `need_n_rs_seq()` gives ngram drafting no rollback ring, so the server takes a full state checkpoint every verify step.

## Run it

```
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp && git checkout 4fbda12
git apply ../CIOT-V2/patches/pr-arm-pq2_0-repack.patch
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_NATIVE=ON
cmake --build build --target llama-server llama-bench
```

The repack turns on by itself on AArch64 CPUs with dotprod (i8mm for prefill). On Windows, `run_bonsai.ps1` starts a local chat with the model.

## Status

- **Correctness:** checked against the official runtime: perplexity, greedy output, and integer-exact kernel tests.
- **Hardware:** only tested on one machine, a Snapdragon X X126100.
- **Equal-memory accuracy comparison:** in progress. Partial laptop run of Bonsai 2 27B: GSM8K 58/60, HumanEval 19/19 on the first 19 problems.
- **Remaining speed gap:** end-to-end decode is ~9.7 tok/s against a ~15 tok/s kernel bound. The rest is runtime overhead.

Built with help from Claude Code.

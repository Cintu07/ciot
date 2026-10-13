# ARM PQ2_0 contribution preparation

Prepared on 2026-10-13 against PrismML prism revision e8fd209c975c80aa7fb36a8423571ff3b1b7e955. No upstream submission has been made.

## Scope and evidence

The candidate is patches/pr-arm-pq2_0-repack.patch. It changes only ggml/src/ggml-cpu/repack.cpp. It adds an internal row-pair layout for the existing PQ2_0 type, SDOT decode, SMMLA batches, a dotprod-only fallback, and handling for batches with two or three remaining activation rows. It changes no GGUF format and trains no new model.

The complete research patch contains other changes, including speculation and profiling. Do not include that patch in this contribution or attribute its entire performance result to this clean candidate before reproducing it.

Checks completed on this x64 laptop:

- The clean candidate applies to current prism revision e8fd209. Git whitespace checks pass.
- The full modified repack.cpp passes Clang 23.1.3 ARM64 Windows syntax checks for armv8.2-a+dotprod and armv8.6-a+dotprod+i8mm.
- A CPU-only MSVC x64 Release build succeeds for test-backend-ops, llama-bench, llama-perplexity and llama-completion.
- Existing PQ2_0 MUL_MAT tests: 219/219 executed CPU/reference cases pass. Unsupported cases are skipped by upstream.
- Existing Q4_0/Q8_0 tests covering the shared remainder-loop wrapper: 75/75 executed cases pass.
- Both standalone check tools compile on x64. The direct repacked tool correctly exits with SKIP on x64. The native token check passed a same-binary control on the small PQ2_0 fixture: three prompts, 32 IDs each. This validates the tool, not stock-vs-candidate ARM correctness.

These x64 tests do not execute ARM instructions or the new ARM repacked buffer. Cross-compilation checks syntax, not execution or speed. The standard backend operator tests allocate ordinary CPU buffers; passing them alone does not validate this repacker.

Recorded evidence: bench/contribution/preparation.json and the two backend status JSON files. Raw native test output is in the adjacent .log files.

Historical measurements, which still need reproduction on the clean candidate:

- Research integration, Snapdragon X, Bonsai 27B PQ2_0: stock pp64 8.4 tok/s and tg32 6.5; T2P pp64 about 31.7 and tg32 about 9.7. Sources: bench/e2e_bonsai_20261007_202636.txt and bench/e2e_t2p_27b.txt.
- The handoff records clean-candidate validation on Qwen3-0.6B with matching printed perplexity and about a 2.2x faster pass. This is a historical note, not a fresh reproduction.
- The 14 tok/s code-edit result includes custom speculation from the research patch. It is not a result of the clean repack contribution by itself.
- The GPU accuracy comparison uses PrismML's official CUDA runtime. It is independent of validation of this ARM patch.

## Running the remaining model checks

Use the Snapdragon laptop with its ARM64 clang/clang++ and Ninja on PATH. The existing PQ2_0 model files must be in models/. The runner creates isolated stock and candidate directories and preserves unexpected edits. It downloads source code only, not models.

Preview the commands on either machine:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/validate_arm_contribution.ps1 -PlanOnly
```

Run the smaller model first:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/validate_arm_contribution.ps1 -Stage Small -Python C:\Python314\python.exe
```

After inspecting the small run, reproduce the 27B results:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/validate_arm_contribution.ps1 -Stage Large -Python C:\Python314\python.exe
```

Both commands build stock and candidate at the same revision, use CPU only, record their commands and outputs, compare three printed perplexity chunks, compare greedy token IDs on three prompts through the native model API, and interleave two benchmark rounds. They also run a 78-case direct repacked-buffer check. They use two compile workers and four inference threads by default; historical headline measurements used eight inference threads. Matching the old settings requires -Threads 8 and an appropriate thermal/power setup. No GPU workload runs. Adjust -Python to the actual Python installation if needed.

The runner's ARM execution is not yet tested here. Native ARM execution of the operator cases, stock/candidate token-ID comparison, and full applicable upstream CI remain pending. A match at printed perplexity precision is not a proof of identical logits. The tiny model is a requantized fixture, not a trained ternary language model; its generated prose can be incoherent. A server-based attempt hit chat-parser errors, so the check tools use native token IDs without chat parsing.

The standalone operator tool covers activation widths 1, 2, 3, 4, 5, 7 and 8; row counts 4, 8 and 12; inner widths 256, 512 and 1024; selector rejection for non-tiled shapes; zero activations/weights; and constant or extreme inputs. Its integer activations make Q8_K conversion exact, allowing an independent scalar dot-product oracle. Run it on dotprod-only and i8mm builds. General floating-point inputs, runtime ISA selection and full applicable CI still need review. These tools remain outside upstream's tests directory; upstream requires approval before adding new files there.

## Explaining the design

PQ2_0 stores u = w + 1, where w is -1, 0 or +1. Thus u is 0, 1 or 2. Repacking rearranges these two-bit codes and keeps the original fp16 scale for each 128-weight block.

SMMLA multiplies a 2-by-8 matrix by an 8-by-2 matrix. Grouping eight weights of row A next to the same eight weights of row B gives its left operand directly. The old layout requires shuffling weights into that shape during each multiply. Loading a model pays the rearrangement cost once.

Three slots are masked without shifting. Their integer values carry factors 1, 4 and 16; the highest slot is shifted down. Let A0 and A3 be the unscaled slot sums, A1 the sum scaled by 4, and A2 the sum scaled by 16. Then:

```text
A2 + 4*A1 + 16*(A0 + A3) = 16*dot(u, y)
dot(w, y) = dot(u, y) - sum(y)
```

The kernel subtracts 16*sum(y) and includes 1/16 in the scale. Integer accumulation is exact within the bounded 128-weight block; this does not by itself guarantee identical floating-point results for every input. SDOT gets sum(y) from Q8_K's existing block sums. SMMLA calculates column sums from its interleaved activation chunks and reuses them across weight rows.

SDOT covers one activation row, with eight activation values duplicated to match both weight rows. SMMLA handles four activation rows at once. For two or three leftover activation rows, the patch pads a four-row activation tile with zeros, performs one batch multiply, and copies only the real outputs. One leftover row still uses SDOT. Without i8mm, the fallback reconstructs ordinary activation rows and runs SDOT.

The important review questions are:

- Why can these codes stay in place during the dot product without changing the result?
- Why is the sum of activations subtracted, and why does it carry a factor of 16?
- Why does the layout use row pairs, even for single-token decoding?
- What happens to padded rows and unsupported matrix shapes?
- Which CPU features are required, and how does the implementation avoid unsupported instructions?

## Submission responsibilities

The PTQ1_0 idea was already implemented by upstream PR #290. This candidate targets PQ2_0 repacking. Searches on 2026-10-13 found no obvious duplicate T2P proposal, but related repack fixes exist and should be checked again before submission.

PrismML's AGENTS.md and CONTRIBUTING.md require understanding and manual review of the code, explicit disclosure of meaningful AI assistance, and a feature discussion before a PR. Its AGENTS.md also restricts automated submissions and reviewer replies. The user writes and submits the issue/PR text, understands the design and owns maintenance. This document supplies facts and a walkthrough, not text to publish unchanged.

The user's existing instruction also requires reviewing exact public text before publication. No issue, PR, comment or social post has been created during this preparation.

References:

- https://github.com/PrismML-Eng/llama.cpp/blob/prism/AGENTS.md
- https://github.com/PrismML-Eng/llama.cpp/blob/prism/CONTRIBUTING.md
- https://github.com/PrismML-Eng/llama.cpp/pull/290

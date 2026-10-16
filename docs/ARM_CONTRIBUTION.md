# ARM PQ2_0 contribution preparation

Prepared on 2026-10-13 against PrismML prism revision e8fd209c975c80aa7fb36a8423571ff3b1b7e955. No upstream submission has been made.

## Read this first: status on 2026-10-16

The ARM validation of the clean patch is finished. The patch is ready for Pawan to propose to PrismML. Nothing has been posted upstream: no issue, PR or comment. The prompt at the end of this file tells the next session how to continue and when to delete this file.

### What was done on the Snapdragon laptop (2026-10-16)

1. Pulled the 10 commits made on the RTX 5060 laptop, including tools/validate_arm_contribution.ps1, tools/check_arm_tokens.py and bench/arm_repack_check/.
2. Ran tools/validate_arm_contribution.ps1. Stock and candidate were both cloned at e8fd209 and built CPU-only with ARM64 clang (llvm-mingw).
   - Small stage (Qwen3-0.6B PQ2_0 fixture) passed. Evidence: bench/contribution/arm-20261016-162025.
   - Large stage (Bonsai 2 27B PQ2_0) passed. Evidence: bench/contribution/arm-20261016-163951.
3. Fixed two bugs in the test tools. Both came from writing them on x64, and neither is in the patch:
   - bench/arm_repack_check/CMakeLists.txt: ciot-token-check now links ggml-cpu. Static MinGW libraries need it explicitly; on x64 the runtime was built as DLLs, which hid this.
   - bench/arm_repack_check/main.cpp: the exact-integer oracle now forces absmax 127 in every 256-value Q8_K block, not only the first. Without that, rows longer than 256 gave Q8_K a scale other than 1, so conversion was inexact. That caused 24 false failures (k = 512/1024, activation columns 3 and 4).
4. Checked the patch against the newest prism commit, 5ae2e16 (2026-10-15).
   - It applies cleanly.
   - repack.cpp, repack.h, arch-fallback.h and arch/arm/repack.cpp are byte-identical to e8fd209. Only ggml-cpu.c changed, and the patch does not touch it.
5. Cleaned up the Snapdragon, freeing 11.5 GB (C: went from 8.4 to 19.9 GB free).
   - Deleted: Bonsai PTQ1_0, the partial Qwen3.8-27B IQ2_XXS and Qwen3-8B Q6_K downloads, Qwen3-0.6B Q8_0 and PTQ1_0, and the old vendor/prism-pr worktree.
   - Kept: models/Ternary-Bonsai-2-27B-PQ2_0.gguf (chat model, Large stage), models/Qwen3-0.6B-PQ2_0.gguf (Small stage), vendor/prism-llama.cpp with its build (run_bonsai.ps1), vendor/prism-arm-stock and vendor/prism-arm-candidate, and build/ (validation builds, so a rerun takes minutes).
6. Pushed everything as Pawan <pawankalyan1892@gmail.com>, with no Co-Authored-By trailers: commits 31c8858, beafd08 and this update.

### Results (stock PrismML vs the patch, same revision, 4 threads, --prio 2)

| check | Qwen3-0.6B PQ2_0 | Bonsai 2 27B PQ2_0 |
|---|---|---|
| direct CPU_REPACK cases (runs the new ARM code) | 78/78 | 78/78 |
| test-backend-ops MUL_MAT pq2_0 | 219/219 on both | not run at this stage |
| perplexity, 3 chunks | identical | identical (10.5976, 17.2967, 16.5409) |
| greedy token IDs, 3 prompts | all match | all match |
| prefill pp64 | 240.1 -> 788.4 tok/s (3.28x) | 4.87 -> 18.27 tok/s (3.76x) |
| decode tg32 | 135.0 -> 146.2 tok/s (1.08x, overhead bound) | 4.17 -> 6.63 tok/s (1.59x) |

The 27B perplexity pass took 27.52 s stock and 7.44 s with the patch. The older headline numbers (8.4 -> 31.7 prefill, 6.5 -> 9.7 decode) used 8 threads and the research build. These runs reproduce the same ratios with the clean patch alone.

### What comes next

1. T2P walkthrough with Pawan, using the review questions in "Explaining the design" below. He must be able to answer them without help; PrismML's AGENTS.md requires it.
2. Pawan writes the PrismML issue himself, proposing the PQ2_0 ARM repack with the numbers above. AI help goes in their template disclosure.
3. Right before the PR, rerun the Small stage on the Snapdragon against the newest prism revision (update $revision in tools/validate_arm_contribution.ps1). That takes about 5 minutes with builds in place.
4. Pawan opens the PR from the clean patch, with the AI disclosure line in PrismML's template.
5. Optional:
   - The full RTX 5060 eval (-Gsm 250 -He 164, 1-2 hours of GPU).
   - A later PR for ngram speculation on hybrid models: cap the rollback ring, use short drafts. Check open PRs #208, #289 and #320 first.
   - Further ARM speed work.

The Snapdragon is still needed for step 3 and for any change reviewers request. ARM code can only be tested on ARM hardware, or on an ARM cloud VM with i8mm such as AWS Graviton3.

Standing rules:
- Pawan writes every public word: issues, PRs, comments, posts.
- No Co-Authored-By trailers or assistant credits in his commits or READMEs.
- Commit as pawankalyan1892@gmail.com.
- PrismML requires AI disclosure inside the PR template.

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

Recorded evidence: bench/contribution/preparation.json and the two backend status JSON files. Readable native test output is in the adjacent .txt files, with ANSI colors and trailing spaces removed. Raw .log files remain locally; their hashes are recorded in the status files.

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

## ARM Small-stage validation, 2026-10-16 (Snapdragon X X126100)

Run: bench/contribution/arm-20261016-162025. Command: tools/validate_arm_contribution.ps1 -Stage Small. Stock and candidate builds at e8fd209, CPU only, 4 inference threads, 2 build jobs.

- 78/78 direct CPU_REPACK cases passed. This runs the new ARM repacker: activation widths 1-8, rows 4/8/12, inner widths 256/512/1024, and the selector declining unsupported shapes.
- test-backend-ops MUL_MAT type_a=pq2_0: 219/219 on both stock and candidate.
- Perplexity, Qwen3-0.6B PQ2_0 fixture, 3 chunks: 273346581.6497, 135191845.7288, 113512154.8283 on both builds.
- Greedy token IDs: 3 prompts x 32 tokens, all match.
- llama-bench, two interleaved rounds: pp64 240.1 -> 788.4 tok/s (3.28x); tg32 135.0 -> 146.2 tok/s (1.08x, noisy; a model this small is overhead bound).

Two fixes to the test tools were needed on ARM. Neither is a change to the patch:

- bench/arm_repack_check/CMakeLists.txt: ciot-token-check did not link ggml-cpu. Static MinGW libraries need it; the x64 build used DLLs, which hid the problem.
- bench/arm_repack_check/main.cpp: the exact-integer oracle forced absmax 127 only in a column's first 256 values. Q8_K has one scale per 256, so for k = 512/1024 the second block could get a scale other than 1. Conversion was then inexact and the comparison failed (24 cases, only columns whose later blocks lacked +/-127). The tool now forces +/-127 in every 256-value block.

Still pending: the Large stage (Bonsai 27B), full applicable upstream CI, and testing on other ARM cores.
## ARM Large-stage validation, 2026-10-16 (Bonsai 2 27B PQ2_0, Snapdragon X)

Run: bench/contribution/arm-20261016-163951. Command: tools/validate_arm_contribution.ps1 -Stage Large. Same stock and candidate builds at e8fd209, CPU only, 4 inference threads, --prio 2, --load-mode none.

- 78/78 direct CPU_REPACK cases passed again.
- Perplexity, 3 chunks of 128 tokens: 10.5976, 17.2967, 16.5409 on both builds. Time per pass: 27.52 s stock, 7.44 s candidate.
- Greedy token IDs: 3 prompts, all match. 32, 28 and 25 tokens; two prompts stopped at end of generation.
- llama-bench, two interleaved rounds (stddev 0.21 or less):
  - pp64: 4.87 -> 18.27 tok/s (3.76x).
  - tg32: 4.17 -> 6.63 tok/s (1.59x).
- The earlier headline numbers (8.4 -> 31.7 and 6.5 -> 9.7) used 8 threads and the research build. This run reproduces the ratio on the clean patch alone, at the default validation settings.

A first attempt stalled for minutes on `git rev-parse` in the stock checkout and was killed and rerun unchanged. Its empty run folder was deleted.

Checked on 2026-10-16:
- The patch also applies cleanly to the newest prism commit, 5ae2e16 (2026-10-15).
- repack.cpp, repack.h, arch-fallback.h and arch/arm/repack.cpp are byte-identical between e8fd209 and 5ae2e16. Only ggml-cpu.c changed, and the patch does not touch it.
- Before submitting, rerun Small against the newest revision.
## Prompt for the next session (paste this as the first message)

```text
You are continuing CIOT-V2 (repo github.com/Cintu07/ciot, branch main). Run git pull first.

Read, in this order: docs/ARM_CONTRIBUTION.md (start with "Read this first: status on 2026-10-16"), docs/HANDOFF.md, docs/PROGRESS.md.

State: the clean ARM patch patches/pr-arm-pq2_0-repack.patch is fully validated on the Snapdragon X against PrismML prism e8fd209, and it still applies to 5ae2e16:
- Small and Large stages passed.
- 78/78 direct repack cases.
- Identical perplexity and greedy tokens.
- 27B: 3.76x prefill and 1.59x decode at 4 threads.
- Evidence is in bench/contribution/arm-20261016-162025 and arm-20261016-163951.
Nothing has been posted to PrismML.

Do this:
1. Move everything still useful from docs/ARM_CONTRIBUTION.md into docs/HANDOFF.md and docs/PROGRESS.md, so nothing is lost:
   - the validation results table and run folder paths;
   - the two test-tool fixes;
   - the design explanation and the five review questions;
   - the submission rules;
   - the "what comes next" list.
2. Delete docs/ARM_CONTRIBUTION.md. Update every reference to it (grep the repo: README.md, docs/*.md, tools/, bench/) so none point at a missing file.
3. Commit as Pawan <pawankalyan1892@gmail.com>. No Co-Authored-By trailer and no assistant credit anywhere. Push to main.
4. Then start the T2P walkthrough with Pawan. Quiz him on the review questions until he can explain the row-pair layout, the x4/x16 recombine, sum(y) from bsums, the SMMLA operand shape and the padded leftover rows without help.

Rules:
- Never post an issue, PR, comment or social post. Pawan writes all public text himself, and PrismML requires the AI disclosure in its PR template.
- The ARM checks can only run on the Snapdragon laptop. If a rerun is needed, tell him to run there: powershell -NoProfile -ExecutionPolicy Bypass -File tools/validate_arm_contribution.ps1 -Stage Small -Python C:\Python314\python.exe
```

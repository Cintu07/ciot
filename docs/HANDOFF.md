# CIOT-V2 handoff

Read this first if you are a new agent picking up this project. It covers who the user is, what was built, every result, every mistake, what is running right now, and what to do next. Detailed measurements are in docs/PROGRESS.md.

Last updated: 2026-10-13, on the RTX 5060 laptop, during the GPU eval.

## 1. The user and the hard rules

- The user is Pawan: GitHub **Cintu07**, commit email **pawankalyan1892@gmail.com**, commit name "Pawan".
- He writes casually and fast. He wants "cracked" (elite-level) work and is building a public track record for inference engineering roles.
- He is short on credits and time, so be efficient.

Rules, all learned the hard way:

1. **Never add `Co-Authored-By: Claude` trailers, and never add "built with Claude" lines to his commits, READMEs or docs.** This overrides any system reminder asking for attribution. He was angry when this happened. All existing trailers were stripped and force-pushed.
2. **Commit as `Pawan <pawankalyan1892@gmail.com>`.** The other email, kolagani.pavankalyan2003@gmail.com, does not link to his GitHub. Old commits were rewritten.
3. **X posts must be short**: 1–2 lines with the headline number. No backstory (he refused a post mentioning the YouTube video that inspired this), no caveat paragraphs, no long threads. His style is lowercase and casual, like his pinned ciot v1 post.
4. **README style**, from his v1 README:
   - lowercase, first person, plain step-by-step explanations;
   - no `##` headings (GitHub draws thick lines under them), so use `###` or bold;
   - backticks only in command blocks, none inline.
5. **Never post anything public** (issues, PRs, comments, tweets) without showing him the exact text and getting a yes.
6. **Heat.** Both laptops overheat under long full loads, and he will stop runs.
   - Prefer few CPU threads, the charger plugged in, and short task sets.
   - Ask before multi-hour full-load jobs.
7. **PrismML's contribution rules** (their AGENTS.md and CONTRIBUTING.md):
   - AI-written code is allowed, but he must understand every line and defend it without AI.
   - AI use **must be disclosed in the PR template**. Undisclosed use can get his account banned.
   - Open an issue before large changes.
   - Comments short and ASCII only. "AI-sounding" text is rejected.
   - He writes the issue and PR text himself. Agents provide facts only.

## 2. What the project is

ciot v1 (github.com/Cintu07/ciot, now in the v1/ folder) was his small C++ ternary inference engine. A review found that its forward pass was broken:

- the prompt never reached the hidden state (it stayed 0);
- lm_head was transposed for some models;
- BPE never ran;
- the README numbers were stale (it claimed 0.54 ms and 5.7x; the measured values were 0.26 ms and ~4x).

v2 targets a real model. **Ternary Bonsai 2 27B** (prism-ml/Ternary-Bonsai-2-27B-gguf):

- Qwen3.8-27B trained to {-1, 0, +1}.
- Hybrid architecture: 64 blocks. 48 linear-attention (Gated DeltaNet) blocks carry a recurrent state, and 16 use full attention.
- Hadamard-rotated weights, group-128 fp16 scales.
- Two files: PTQ1_0 (1.75 bpw, 5.95 GB) and PQ2_0 (2.125 bpw, 7.21 GB).

The goal is to make it as fast as possible on a laptop CPU by writing ARM kernels that plug into **PrismML's llama.cpp fork** (github.com/PrismML-Eng/llama.cpp, branch prism), with output bit-identical to the official runtime.

## 3. Machines

**Old laptop** (all kernel work was done here):

- Hardware: Snapdragon X X1-26-100, 8 Oryon cores, 15.6 GB RAM, Windows on ARM64.
- Path: C:\Users\Pawan\Desktop\CIOT-V2. The Windows user is "Pawan", capital P.
- ISA: dotprod, i8mm, bf16. No SVE.
- Measured DRAM read bandwidth: ~110–115 GB/s (bench/bw.txt).
- Toolchain: llvm-mingw aarch64 clang 22 (WinGet), cmake 4.3.2, ninja 1.13.2, Python 3.14 at C:\Python314.
- Git Bash on that machine is x86-emulated.

**New laptop** (where we are now):

- Hardware: Intel i7-13650HX, **RTX 5060 Laptop GPU 8 GB** (driver 581.95), 15.7 GB RAM, Windows x64.
- Path: C:\Users\pavan\Desktop\CIOT-V2. Windows user "pavan", lowercase. The whole folder was copied from the old laptop, including models/ and vendor/.
- Python 3.12.10 was installed per-user with winget at %LOCALAPPDATA%\Programs\Python\Python312\python.exe. It is **not on PATH** in older shells. Typing `python` hits the Microsoft Store alias.
- git and gh are present (gh is logged in as Cintu07). curl.exe is built in.
- Internet: ~3.5 MB/s in total, shared across downloads.
- **This laptop is x86, so the ARM NEON/i8mm kernels cannot be built or tested natively here.** ARM work needs the Snapdragon laptop or an ARM cloud VM with i8mm (for example AWS Graviton3; Oracle's free Ampere A1 / Neoverse N1 has dotprod but no i8mm).

## 4. Repository state

GitHub: **github.com/Cintu07/ciot**

- **main**: a merge commit that joins both histories. v1 was moved into v1/ and v2 sits at the top level.
- **v1-backup**: the old main, untouched.
- **v2**: the pre-merge v2 branch, now redundant (he can delete it).

Layout:

```
kernels/      ptq1_0_neon.c, t2_neon.c, t2_i8mm.c, t2p.c (+ .h). t2p.c is the current best.
patches/      pr-arm-pq2_0-repack.patch: the clean PR candidate (only ggml-cpu/repack.cpp, on prism 4fbda12)
              prism-llama.cpp-t2-neon.patch: the full research patch on prism 6bfcd79 (profiler, CIOT_* toggles, ciot-spec example, PTQ1_0 kernel)
bench/        harnesses + raw outputs: ternary_kernels, t2_gemm, t2p_bench, decode_replay, bw, e2e_bonsai*.txt, e2e_t2p_27b.txt, spec*.txt, ppl27*.log
bench/showcase.sh   prints the results box from the logs (used for the X screenshot)
bench/eval/   run_eval.py (llama-server based GSM8K + HumanEval harness), fetch_data.py, data/*.jsonl,
              run_rtx_windows.ps1 (Windows+NVIDIA runner), results/ (laptop CPU partial run), results-rtx/ (GPU runs)
kaggle/       ciot_equal_memory_eval.ipynb (generated by tools/make_kaggle_notebook.py), NEVER RUN, untested
tools/        make_kaggle_notebook.py
docs/         PROGRESS.md (full measurement log), HANDOFF.md (this file)
run_bonsai.ps1  one-click local chat with Bonsai (uses vendor/prism-llama.cpp build; ARM laptop only)
v1/           original ciot
```

Not in git (ignored): models/, vendor/, build/, gpu-runtime/, *.log, *.err.

vendor/ on this laptop was copied from the old one:

- **vendor/prism-llama.cpp** is the fork at 6bfcd79 with the full research patch applied, uncommitted. Its build/ folder holds **ARM64** binaries, so it is useless on this x86 laptop.
- **vendor/prism-pr** is a git worktree on branch arm-pq2_0-repack (origin/prism 4fbda12 plus the clean patch, uncommitted). The worktree metadata points at C:\Users\Pawan\..., so it is probably broken here. Recreate it with `git -C vendor/prism-llama.cpp worktree prune`, then `git worktree add`, then apply patches/pr-arm-pq2_0-repack.patch.

## 5. Results (all from logs in bench/)

**Bonsai 2 27B on the Snapdragon X CPU** (8 threads, `--prio 2`, llama-bench, interleaved rounds):

| runtime | prefill tok/s | decode tok/s |
|---|---|---|
| official, PTQ1_0 file (generic C on ARM) | 4.5 | 4.3 |
| official, PQ2_0 file (PrismML NEON) | 8.4 | 6.5 |
| ciot T2 (first layout) | 27.2 | 9.4 |
| **ciot T2P (current)** | **31.7** | **9.7** |

- **Perplexity** on a 3x128-token README text: identical on every chunk with stock (9.2430 / 7.3521 / 10.0138).
- **Speculative decoding** (examples/ciot-spec, code-edit prompt, chat format with thinking pre-closed):
  - 14.0–14.1 tok/s vs greedy 8.9–9.4 in the same runs (1.5–1.6x);
  - 76% of drafts accepted at k=2, 0 token mismatches;
  - on prose: 0.96x, caused by the cost of the snapshot ring.
- **Kernels** (8 threads, persistent pool):
  - T2P matvec 107 GB/s, about 94% of DRAM bandwidth;
  - 4-token SMMLA verify 1136 GMAC/s, vs 862 for T2;
  - 64-token prompt batches 1.27x faster than T2.
- **Thread priority**: at default priority, Windows preemption cost about 1/3 of each token. `--prio 2` took decode from 3.8 to 6.0 tok/s.

**Accuracy eval** (bench/eval/run_eval.py):

- Same prompts for every model: ChatML with thinking pre-closed (`<think>\n\n</think>`), greedy decoding.
- GSM8K is graded by exact answer. HumanEval is graded by its unit tests, run in a subprocess with a 30 s timeout.
- Both graders were validated: 1319/1319 GSM8K gold answers parse, 164/164 HumanEval reference solutions pass.

| run | model | GSM8K | HumanEval | speed |
|---|---|---|---|---|
| Snapdragon CPU, stopped for heat | Bonsai PQ2_0 | 58/60 | 19/19 (first 19) | ~30 s/task |
| RTX 5060, full set, stopped for heat | Bonsai PTQ1_0 | 69/70 | not reached | 14 tok/s on battery |
| **RTX 5060, quick set (60 + 40)** | **Bonsai PTQ1_0** | **59/60** | **38/40** | 37.6 tok/s, 5.3 s/task |
| RTX 5060, quick set, RUNNING | Qwen3.8-27B UD-IQ2_XXS (7.27 GB) | 7/8 so far | - | ~49 s/task (part runs on CPU) |
| RTX 5060, quick set, QUEUED | Qwen3-8B Q6_K (6.73 GB) | - | - | - |

Result files for the full stopped run: bench/eval/results-rtx/bonsai2-27b-ptq1_0.stopped-full-run.jsonl.

Observations so far:

- At equal memory, Bonsai answers about 7–9x faster on the 8 GB GPU, because it fits entirely in VRAM and the 2-bit Qwen does not.
- Qwen3.8 IQ2_XXS failed task 3 by emitting a 1-token answer.

## 6. Chronological log

1. **Reviewed v1.**
   - The forward pass was broken: the hidden state stayed 0 and lm_head was transposed.
   - The int8 SDOT kernel was 10–14x faster than v1's fp32 masked-add.
2. **Researched the field**: the YouTube video, the comments, Bonsai 2, and PrismML's KNOWN_ISSUES.
   - KNOWN_ISSUES says: CPU paths are slow, "MTP is refused", "ngram has no effect", and the fork targets CUDA/Metal.
   - In arch-fallback.h, PTQ1_0 on ARM is aliased to the generic C loop.
3. **Built the fork for ARM64** with llvm-mingw.
   - Workarounds: `-D_WIN32_WINNT=0x0A00` (httplib's CreateFile2), `-DGGML_OPENMP=OFF`, `LLAMA_CURL=OFF`.
   - llama-cli is now called llama-completion.
4. **ptq1_0_neon.c**: decodes trits with the identity floor(3v/256) = (v>=86)+(v>=171).
   - 1.5–1.7x over generic in the harness.
   - In the runtime: prompt 1.34x; decode inconclusive.
5. **T2 layout** (2-bit, slots masked in place with x4/x16 accumulators, offset removed through Q8_K bsums, 4 rows interleaved):
   - 1.4–1.7x over PrismML's PQ2_0 NEON in the harness;
   - integrated as an in-place repack of PQ2_0 in repack.cpp (tag INTER_SIZE 16), with `CIOT_T2=0` to turn it off.
6. **i8mm SMMLA gemm for prompts**: 2.5x prefill.
7. **Per-op profiler** (`CIOT_PROF=1` in ggml-cpu.c).
   - Found that ~2,400 small ops cost ~100 µs each because of Windows preemption.
   - `--prio 2` fixed it.
8. **27B end to end**: T2 gave 9.4 decode and 27.2 prefill, bit-identical.
9. **Speculative decoding** (examples/ciot-spec):
   - Found the root causes behind PrismML's speculation issues (sections 1 and 7 of PROGRESS.md).
   - Proved rollback correct with oracle and junk drafts.
   - Fixed odd-sized verify batches (they used to read the weights once per leftover row; now they get one padded SMMLA pass).
   - Added adaptive drafting with backoff.
10. **T2P row-pair layout**: decoded slots are SMMLA operands as-is, with no zips.
    - +16% prefill (31.7) and the best speculative speed (14.1).
11. **Eval harness and Kaggle notebook.**
    - The laptop CPU eval was stopped because of heat.
    - Recon then found that PrismML #290 (karusrus, "arm: NEON + i8mm vec_dot for PTQ1_0", merged 2026-10-07) did our planned PR 1, so **PR 1 is dropped**.
12. **Recon for contributing to PrismML**: very active, merges outside PRs daily.
    - The ARM PQ2_0 repack (our T2P) is free; no open PR covers it.
    - The clean PR candidate was built and validated with `--no-repack` as the A/B: perplexity identical, 2.2x faster pass on Qwen3-0.6B.
    - Not committed in the fork. He must write the commit and PR text.
13. **Published to GitHub**:
    - rewrote the commit email;
    - stripped the co-author trailers;
    - rewrote the README in his v1 voice;
    - merged v1 and v2 on main.
14. **New laptop (RTX 5060)**: run_rtx_windows.ps1 uses PrismML's official Windows CUDA 13.3 build (release prism-b10772-e8fd209). Bonsai quick-set results are in section 5.

## 7. Mistakes and lessons

- **Two different harness compiles** of the same generic kernel gave 15 vs 2.8 Gweights/s, because clang vectorized them differently. Use the real llama-bench number as the baseline, not a harness copy.
- **Spawning threads on every iteration** in a benchmark undercounted 8-thread throughput: 282 Gw/s, vs 391 with a persistent pool.
- **PowerShell 5.1 `>` and Out-File** write a BOM, UTF-16 or CRLF, which corrupts patches. Use `git diff --output=FILE`.
- **Git Bash / MSYS stalls**: processes started from Git Bash (timeout.exe, filter-branch's sh, git commit) can hang forever when the CPU is saturated by a high-priority llama run.
  - Launch long jobs with PowerShell Start-Process.
  - Prefer PowerShell for git.
  - Kill stuck git/sh/bash processes when needed (81 were killed once).
- **`git log` without `--no-pager`** hung a tool call.
- **Windows Terminal `wt`** treats `;` as "new tab". Use `&&` inside the command.
- **`--prio 2` llama runs starve everything else**, including the agent's own shell commands and the HumanEval test subprocess. Re-grade offline if timeouts look suspicious.
- **Eval grader bug**: models sometimes end the turn without closing the code fence. extract_code now accepts an unclosed fence. HumanEval/49 hit the old 15 s test timeout under load, so it was raised to 30 s.
- **curl `-C -` on an already-complete file** errors out. The script now checks Hugging Face's x-linked-size header and skips complete files.
- **Concurrent downloads starve each other.** The runtime zip ran at 50 KB/s until the model downloads were paused.
- **Misread "common_fit_params: failed to fit params ... n_gpu_layers already set by user to 99, abort" as a failure.** It is only a warning; the model loaded fully on the GPU.
  - Acting on that misreading killed the chain process that was meant to start the Qwen runs after the downloads, so they had to be restarted by hand.
  - Check the summary files before concluding anything failed.
- **The laptop was on battery during the first GPU run.** The GPU drew 24 W and ran at 14 tok/s; on the charger it ran at 37.6 tok/s. Check Win32_Battery BatteryStatus (1 = battery, 2 = AC).
- **PowerShell `-f` precedence**: `-f a, 100 * b, c` breaks the argument list. Wrap computed values in parentheses. (Fixed in the table at the end of run_rtx_windows.ps1.)
- **Co-author trailers were added to his commits** against his wishes. They were stripped with git fast-export, a Python rewrite that recomputes the data lengths, and fast-import, then force-pushed. **Don't do this again.**
- **The README had a "Built with help from Claude Code" line.** Removed. Don't add it back.
- **Upstream rs-rollback example**: it fails on the 27B on CPU with stock kernels too, because the snapshot ring only covers the last ubatch. Within-batch speculation is fine.
- **A k=7 speculative sweep run died at exit** and lost buffered stdout. ciot-spec now uses setvbuf for unbuffered output.
- **Over-claiming risk**: PTQ1_0 decode gains were inconclusive (4.0 / 6.2 vs ~4.35). Only claim the consistent 1.5x prompt gain.

## 8. Running right now (at the time of writing)

On this laptop, launched with:

```
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\pavan\Desktop\CIOT-V2; & .\bench\eval\run_rtx_windows.ps1 -Gsm 60 -He 40 *> bench\eval\results-rtx\run2_qwen.log"
```

- Processes: powershell (PID 20388), python run_eval.py (22452), llama-server (25184).
- It skips Bonsai (its summary exists) and runs qwen3.8-27b-iq2_xxs (about 75 more minutes from 8/100), then qwen3-8b-q6_k (about 10–20 minutes).
- Settings: 4 CPU threads (`-CpuThreads 4`, for heat), auto-fit layer placement for both Qwen models. Bonsai used `-ngl 99`.
- Check progress with:
  ```
  Get-Content bench\eval\results-rtx\run2_qwen.log -Tail 3
  ```
  Also look at the *.partial.jsonl files.
- If it died, rerun the same command. Finished models are skipped, but a half-done model restarts from zero.

## 9. Next steps, in order

1. **Wait for both Qwen runs to finish.** Then:
   - read the three *.summary.json files;
   - compare accuracy, tokens per answer, s/task and tok/s;
   - look at the failure texts in the jsonl files (for example Qwen's 1-token answer).
2. **Commit the results.**
   - The results-rtx folder: jsonl and summary.json. The server logs are ignored; force-add them only if needed.
   - The pending run_rtx_windows.ps1 changes, if they are not committed yet.
   - Update the README "status" section and PROGRESS.md with the head-to-head.
   - Push to main as Pawan, with no trailers.
3. **Optional**: the full set (`-Gsm 250 -He 164`) for statistical strength. That is about 1–2 hours of GPU time with heat, so ask first. Alternatively the Kaggle notebook (untested).
4. **X post with the head-to-head**: short, his style. Only if the numbers are clear.
5. **PrismML PR** (the big credibility item). Needs the Snapdragon laptop or an ARM VM to rebuild and test.
   1. Walk him through T2P until he can explain it: the row-pair layout, why the x4/x16 recombine is exact, bsums giving sum(y), and the SMMLA operand layout.
   2. Rebase the clean patch onto the latest prism branch (upstream moves daily).
   3. He opens an issue first ("ARM CPU: PQ2_0 repack, 3.8x prefill on Snapdragon X, bit-identical, interested in a PR?"), written by him.
   4. Then the PR, with the AI disclosure in the template.
   - The ngram speculation fix is a possible later PR (cap the ring, short drafts). Check open PRs #208, #289 and #320 first for overlap.
6. **Further speed work, on ARM**:
   - Close the runtime gap: decode is 9.7, against a ~15 tok/s kernel bound. Targets: recurrent-state copies, fusing gate+up and q+k+v, lighter scheduling.
   - Better drafters (MTP / draft heads).
   - Hexagon NPU moonshot.
   - a CUDA port of the T2P idea to cover GPU skills.
7. **Cleanup when everything is done**:
   - Delete the partial and duplicate models and the Qwen3-0.6B test files.
   - Keep the Bonsai file he wants to use: PQ2_0 on the ARM laptop; PTQ1_0 is enough on this one.
   - Delete gpu-runtime/ if no longer needed.

## 10. Command cheat sheet

```
# progress of the GPU eval (this laptop)
Get-Content bench\eval\results-rtx\run2_qwen.log -Tail 3
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader

# rerun / resume the GPU eval (skips finished models)
powershell -ExecutionPolicy Bypass -File bench\eval\run_rtx_windows.ps1 -Gsm 60 -He 40

# python on this laptop (not on PATH in old shells)
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" ...

# results box for screenshots (ARM numbers)
bash bench/showcase.sh

# commit as him (no trailers)
git -c user.name="Pawan" -c user.email="pawankalyan1892@gmail.com" commit -m "..."
```

### ciot v2

ternary llm inference kernels for arm cpus. the target is ternary bonsai 2 27b on the cpu of a snapdragon x laptop, no gpu. the kernels plug into prismml's llama.cpp fork as a load time repack of the pq2_0 weights. on the same laptop with the same settings, prompt processing is about 3.8x faster and token generation about 1.5x faster than the stock neon path, and the output is identical on every check that was run.

the repo holds the kernels in c, a clean one file patch against the runtime, the full research patch, every benchmark with its raw output, an accuracy eval harness, and the original v1 engine.

a longer walkthrough of the main kernel: https://cintu07.github.io/posts/ciot-t2p-kernel/

### the model

ternary bonsai 2 27b from prismml is a qwen3.8 27b trained down to weights of -1, 0 and +1. it is a hybrid: 64 blocks, 48 of them linear attention (gated deltanet) that carry a recurrent state, and 16 with full attention. weights are hadamard rotated and stored with one fp16 scale per group of 128. it ships in two files.

- ptq1_0: 1.75 bits per weight, 5.95 gb. five trits per byte, base 3.
- pq2_0: 2.125 bits per weight, 7.21 gb. two bits per weight stored as u = w + 1, so 0, 1 or 2.

activations are quantized on the fly to q8_k: 256 int8 values, a float scale and 16 partial sums per block. the two arm instructions that matter are sdot (16 int8 times 16 int8 into four int32 lanes) and smmla from i8mm (a 2x8 int8 matrix times an 8x2 int8 matrix into a 2x2 int32 tile, 32 multiply adds).

### results

all numbers come from one machine: snapdragon x x126100, 8 oryon cores, 16 gb ram, windows on arm64. the cpu has dotprod and i8mm and no sve. raw outputs are in bench.

the clean patch, validated on prism e8fd209 against a stock build of the same commit. 4 threads, high priority, two interleaved rounds, laptop on battery.

```
                    stock     patched   ratio
prefill pp64        4.87      18.27     3.76x    tok/s
decode tg32         4.17      6.63      1.59x    tok/s
perplexity pass     27.52     7.44      3.7x     seconds
```

the research build, prism 6bfcd79 with the full patch. 8 threads, priority 2, two rounds. the stock rounds and the patched rounds were separate sessions with the same settings, so read the ratio as approximate.

```
                    official pq2_0 neon    ciot t2p
prefill pp64        8.4                    31.7       tok/s
decode tg32         6.5                    9.7        tok/s
```

speculative decoding on a code edit prompt, 306 tokens, prompt lookup drafts. 8.88 tok/s greedy against 14.10 with 2 drafts per step, 149 of 196 drafts accepted, zero token mismatches against greedy. with 3 drafts it is 9.36 against 13.98. this lives in the research patch, not the clean one.

kernel throughput, a 32768 x 8192 matvec streamed from dram on a persistent pool of 8 threads.

```
prism ptq1_0, plain c loop     19.6 gweights/s     4.3 gb/s
ciot ptq1_0, neon              124.7               27.3
prism pq2_0, neon              231.6               61.5
ciot t2x4, neon                390.5               103.7
```

the plain c row is the stock path at prism 6bfcd79. prismml pr 290 later added a neon i8mm kernel for ptq1_0, so that row is historical. the t2p layout reads 107.0 gb/s against 100.6 for t2x4 in the same run, and the measured dram read ceiling of this machine is 110 to 115 gb/s (bench/bw.txt). 4 threads already saturate it.

prompt batches with smmla on 17408 x 5120 weights, 8 threads: the 4 token tile goes from 862 to 1136 gmac/s with t2p and the 64 token tile from 850 to 1076.

decode is below the kernel ceiling. replaying the matmuls of one token outside the runtime takes about 70 ms, which is about 14 tok/s. inside the runtime the same matmuls take about 92 ms and recurrent state copies add about 19 ms. the remaining gap is runtime overhead, not the kernel.

### the layout, t2p

pq2_0 stores each weight as u = w + 1 in 2 bits. stock neon has to look up, shift, mask and subtract before sdot can use them, which is 16 preparation instructions for 4 dots per 64 weights. t2p keeps the same bits and the same size and changes only the order, so that preparing a vector costs one and.

a block is 4 rows by 128 weights in 136 bytes, the size of four pq2_0 blocks.

```
d[4]            4 fp16 scales, one per row           8 bytes
qs[2][4][16]    two row pairs, four 16 byte words    128 bytes

byte j of word w of pair p holds, in bits 2s..2s+1 for s = 0..3:
    row 2p + (j >= 8), weight 8 * (4w + s) + (j & 7)

one word, 16 bytes:
    byte   0  1  2  3  4  5  6  7   8  9  10 11 12 13 14 15
    row    a  a  a  a  a  a  a  a   b  b  b  b  b  b  b  b

slot s of all 16 bytes, as one 16 lane vector:
    [ a: weights 8c .. 8c+7 | b: the same 8 weights ], c = 4w + s
```

that vector is the 2x8 left operand of smmla as is, so prompt processing needs no zips. for single token decode, the 8 activations of chunk c are duplicated into both halves of a vector, so one sdot covers both rows of the pair: lanes 0 and 1 collect row 2p, lanes 2 and 3 collect row 2p+1.

the repack runs once at model load. after that it costs nothing per token.

### decode kernel

a slot comes out of a byte with an and and nothing else. and with 0x03 leaves u, with 0x0c leaves 4u, with 0x30 leaves 16u. sdot multiplies whatever int8s it is given, so the dot products come out scaled by 1, 4 and 16 and each slot keeps its own accumulator. slot 3 uses a shift by 6, because u = 2 at bit 6 would be 128 and sdot reads that as -128. per 16 byte word that is three ands and one shift.

each block runs 8 independent accumulators, so sdot latency hides behind itself. once per 128 weights the scales are put right. with a0 to a3 the four slot accumulators:

```
a2 + 4*a1 + 16*(a0 + a3) = 16 * dot(u, y)
dot(w, y) = dot(u, y) - sum(y)
```

the sum over the 128 activations is one add over the 8 partial sums q8_k already carries, and the factor 1/16 folds into the float scale that is multiplied in anyway. the largest intermediate is 128 products of at most 2 * 127 times 16, which is 520,192, so everything stays exact in int32 until the final float conversion.

the source is kernels/t2p.c. the pack function, the slot macros and the decode loop are short enough to read in one sitting.

### prompt kernel

with several activation columns in flight a loaded weight is used many times, so instruction count matters more than the memory bus. the gemm works on tiles of 4 weight rows by 4 activation columns. q8_kx4 already interleaves 4 columns in 8 byte chunks, which is the right operand for smmla. one decoded slot per row pair meets two activation operands, giving four smmla per decoded vector with every operand used twice.

batches that are not a multiple of 4 used to fall back to one matvec per leftover row, and every one of those streams the whole 7 gb of weights again. the leftover rows are now padded with zeros to a full tile, run in one smmla pass, and only the real outputs are copied out. this matters for speculative verification, where a batch of 3 lands exactly in the gap.

### base 3 detour, ptq1_0

the 1.75 bit file is 18% smaller than the 2 bit one. trit k of a byte b is floor(3 * (b * 3^k mod 256) / 256), which equals (v >= 86) + (v >= 171) for v = b * 3^k mod 256, checked over all 256 values. neon compares give -1 for true, so a weight is -1 minus two compare masks. kernels/ptq1_0_neon.c does this and reaches 124.7 gweights/s on 8 threads against 19.6 for the stock c loop at that commit. it still reads only 27 gb/s on a bus that delivers 110, because the cpu spends its time decoding. the 2 bit layout wins on this chip because a smaller file only helps when the cpu has spare instructions to unpack it.

### correctness

- an integer oracle compares the repacked buffers against a scalar loop exactly, with no tolerance: 78 direct cases covering batch widths 1 to 5, 7 and 8, row counts 4, 8 and 12, inner sizes 256, 512 and 1024, shapes the selector must reject, all zeros, constants and extreme values. 78 of 78 pass. the standard backend tests for pq2_0 pass 219 of 219 on stock and patched.
- perplexity on the real 27b, 3 chunks of 128 tokens: 10.5976, 17.2967 and 16.5409 on stock and on patched, identical at printed precision. the input text is tools/ppl-input.txt.
- greedy token ids on three prompts through the native api match exactly (32, 28 and 25 tokens).
- speculative decoding was checked with an oracle drafter (229 of 229 drafts accepted) and a junk drafter (0 of 912 accepted), both with zero mismatches against plain greedy.

integer sums are exact. the float math around them can round differently, so "identical at printed precision" is weaker than "identical bits", and the token check covers three prompts, not a proof of logit equality.

### runtime integration

patches/pr-arm-pq2_0-repack.patch changes only ggml/src/ggml-cpu/repack.cpp. it adds a pq2_0 tensor trait with 16 byte interleave, 4 columns and q8_k activations, chosen at load when the cpu reports neon and dotprod. the gemv uses sdot. the gemm uses smmla and needs i8mm. a dotprod only gemm exists that loops the gemv, and it compiles for armv8.2-a plus dotprod, but it has never been executed. the patch also routes leftover rows through one padded pass. it has no environment toggles. it applies to prism e8fd209 and was validated there, and it has not been submitted to prismml.

patches/prism-llama.cpp-t2-neon.patch is the full research patch on prism 6bfcd79. it adds the ptq1_0 neon kernel, a per op profiler, environment toggles to switch the new kernels on and off for a/b runs, and the speculative decoding example ciot-spec with rollback of the recurrent state from a snapshot ring.

### things that went wrong

- windows paused a third of every token. the runtime spins 8 workers at a barrier, the os preempts one for a moment and the other seven wait. a profile showed matmuls at only 65% of a token. running at high priority took decode from 3.8 to 6.0 tok/s on the 1.75 bit file with the first kernel and no code change.
- a benchmark that spawned threads on every iteration reported 282 gweights/s for t2x4. with a persistent pool it is 391.
- with the default memory mapped load the embedding lookup alone cost about 21 ms per token, apparently page faults while the mapped file sat beside the repacked copy. loading without mmap removed it.
- the first run of the integer oracle failed 24 cases. the kernel was fine. the test forced an absolute max of 127 only in the first 256 values of each column, so later blocks quantized with a scale other than 1 and the exact comparison failed for the wrong reason.
- two harness compiles of the same generic kernel measured 15 and 2.8 gweights/s because clang vectorized them differently, so the baseline is the real llama-bench number, not a harness copy.

### accuracy against other models

this compares model choices on an rtx 5060 laptop gpu with 8 gb using prismml's cuda runtime. it does not measure the arm kernels. prompts are chatml with thinking pre closed, decoding is greedy. gsm8k is graded by exact answer and humaneval by its unit tests in a subprocess with a 30 second timeout. the graders were checked first: 1319 of 1319 gsm8k gold answers parse and 164 of 164 humaneval reference solutions pass. the first 60 gsm8k and the first 40 humaneval problems were run.

```
model                              file gb    gsm8k    humaneval
bonsai 2 27b ptq1_0                5.95        59/60    38/40
qwen3.8 27b ud-iq2_xxs             7.27        53/60    37/40
qwen3 8b q6_k                      6.73        56/60    38/40
```

paired exact sign tests on the same problems. gsm8k against the qwen 27b 2 bit: bonsai solved 6 that it missed and the reverse was 0, p = 0.031, the only significant difference. against qwen3 8b: 3 to 0, p = 0.25. humaneval is within noise, 1 to 0 against the 27b and 1 to 1 against the 8b.

this is a small subset with fixed output limits and not a benchmark score. the qwen 27b run spans resumed sessions including a battery to ac transition, so its timings are descriptive. a full run on the laptop gpu was stopped for heat. a kaggle notebook of the same eval is in kaggle and has never been run.

### build and run

the patched runtime, on an arm64 windows machine with clang and ninja:

```
git clone https://github.com/prismml-eng/llama.cpp && cd llama.cpp && git checkout e8fd209
git apply ../ciot/patches/pr-arm-pq2_0-repack.patch
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-server llama-bench llama-perplexity
```

the repack turns on by itself on arm cpus with dotprod. benchmark the way the validation did:

```
llama-bench -m models/ternary-bonsai-2-27b-pq2_0.gguf -p 64 -n 32 -r 2 -t 4 -ngl 0 --prio 2 --load-mode none
```

tools/validate_arm_contribution.ps1 builds stock and patched side by side from clean checkouts and runs every check above, in two stages: small on qwen3 0.6b pq2_0 and large on bonsai 27b. its outputs land in bench/contribution.

```
powershell -executionpolicy bypass -file tools/validate_arm_contribution.ps1 -stage small
powershell -executionpolicy bypass -file tools/validate_arm_contribution.ps1 -stage large -threads 4
```

run_bonsai.ps1 starts a local chat server on port 8080 with the model, expecting a build of the research patch at vendor/prism-llama.cpp.

the kernel microbenchmarks need an arm64 cpu with i8mm:

```
clang -O3 -march=armv8.6-a+dotprod+i8mm -c kernels/t2_neon.c kernels/t2_i8mm.c kernels/t2p.c
clang++ -O3 -march=armv8.6-a+dotprod+i8mm -std=c++17 bench/t2p_bench.cpp t2_neon.o t2_i8mm.o t2p.o -o t2p_bench
```

on windows arm64 keep the thread count and priority the same between runs, and compare interleaved rounds. the model files are not in the repo. get ternary-bonsai-2-27b-pq2_0.gguf from prism-ml on huggingface and put it in models.

### layout of the repo

```
kernels/     t2p.c is the current kernel. t2_neon.c and t2_i8mm.c are the first layout,
             ptq1_0_neon.c is the base 3 kernel
patches/     pr-arm-pq2_0-repack.patch is the clean runtime change,
             prism-llama.cpp-t2-neon.patch is the full research patch
bench/       kernel harnesses, e2e and speculative logs, bandwidth test, the eval
             harness in bench/eval, and the validation runs in bench/contribution
kaggle/      the equal memory eval as a notebook for a free gpu
tools/       validation driver, greedy token checker, perplexity input text
v1/          the original ciot engine
```

### limits

- one machine. that the layout carries over to other arm cores is a hypothesis.
- only the smmla path has been executed. the dotprod only gemm compiles and has never run.
- both validation runs were on battery, started at 38% and 32% charge. stock and patched were interleaved under the same conditions, so the ratios hold, but the absolute numbers are probably lower than on the charger.
- the runtime moves quickly. research numbers are at prism 6bfcd79 and validation is at e8fd209, and a neon kernel for the 1.75 bit file landed in between. re-measure before comparing against a current build.
- the idea has precedent. ggml already repacks q4_0 and q8_0 into interleaved tiles for sdot and smmla. what is specific here is the in place slot masking with the factor 16 recombination and the row pair layout for this 2 bit format.
- decode is 9.7 tok/s against about 15 for the kernels alone.
- the model and the file formats belong to their authors at prismml. this repo is the kernels and the repack.

### v1

v1 is the original small c++ ternary engine, kept as it was. v2 does not build on it.

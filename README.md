# ciot v2

a 27 billion parameter ternary model running on a laptop cpu. no gpu. built by [@Cintu07](https://github.com/Cintu07).

### what it does

ciot v1 was a tiny engine i wrote from scratch to see how fast ternary weights could go on a cpu. v2 points the same idea at a real model: ternary bonsai 2 27b from prismml, which is qwen3.8 27b trained down to weights of -1, 0 and +1. it is 7.2 gb on disk, so it fits in the 16 gb of ram on my snapdragon x laptop.

prismml ships a llama.cpp fork that runs it, but that fork is built for cuda and metal. on an arm laptop cpu the 1.75 bit file falls back to a plain c loop, and the 2 bit file gets a basic neon kernel. so i wrote new kernels for the matmuls, plugged them into their runtime, and checked that the output stays bit identical.

on the same laptop, same file, same settings:

| runtime | prefill tok/s | decode tok/s |
|---|---|---|
| official, 1.75 bit file (plain c) | 4.5 | 4.3 |
| official, 2 bit file (neon) | 8.4 | 6.5 |
| ciot v2 | 31.7 | 9.7 |

with speculative decoding on a code editing prompt it goes to 14 tok/s, and the tokens match normal decoding one for one.

perplexity is the same on every chunk as the official runtime (9.2430, 7.3521, 10.0138). not close, identical.

### how it works, step by step

**the weights.** every weight is one of three values, stored in 2 bits as w + 1, so 0, 1 or 2. prismml already stores them like that. the problem is the order. their kernel has to shuffle bytes, shift and subtract before the cpu can use them. so when the model loads, ciot rearranges the bits once into a layout the arm instructions can eat directly. same bits, same size, different order.

**row pairs.** the layout groups weight rows in pairs. pull one 2 bit slot out of a 16 byte word and you get 8 weights of row a next to the same 8 weights of row b. that is exactly the shape the arm smmla instruction wants (a 2x8 by 8x2 int8 multiply). so for prefill there is no shuffling at all, the decoded weights go straight in.

**no shifts either.** three of the four slots in a byte come out with just an and. their values end up multiplied by 1, 4 or 16, so each one gets its own accumulator and they are combined once every 128 weights. the +1 offset is removed once per block too, using sums of the activations that the int8 activation format already carries.

**decode.** for one token at a time the same layout works with the sdot instruction, which covers both rows of a pair in one go. in a benchmark the decode kernel streams weights at 107 gb/s. this laptop's memory tops out around 110 to 115 gb/s, so the kernel is basically at the hardware wall.

**speculative decoding.** when the model edits code it mostly copies text that is already in the prompt. so ciot guesses the next few tokens by finding the current phrase earlier in the context, checks all the guesses in one batch, keeps the ones that match and rolls back the rest. bonsai is a hybrid model, 3 out of 4 layers carry a running state instead of a kv cache, so rolling back means restoring that state from snapshots. i tested it two ways: feeding it the correct tokens as guesses (all accepted, same output) and feeding it wrong ones on purpose (all rolled back, same output).

### things i found on the way

windows was eating a third of every token. at normal priority it kept pausing one of the 8 worker threads, and the other 7 waited for it at every step. running at high priority alone took decode from 3.8 to 6 tok/s with no code change.

short verify batches were reading all 6.8 gb of weights once per leftover token. now they get padded into one pass.

prismml's server says n-gram speculation has no effect on this model. the reason is that n-gram drafting never asks for the state snapshots, so the server saves the whole state on every step, which costs more than the guesses save.

### build

```
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp && git checkout 4fbda12
git apply ../ciot/patches/pr-arm-pq2_0-repack.patch
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_NATIVE=ON
cmake --build build --target llama-server llama-bench
```

the new kernels turn on by themselves on arm cpus with dotprod. on windows, run_bonsai.ps1 starts a local chat with the model in your browser.

### layout

```
kernels/      the arm kernels (t2p.c is the current one)
patches/      the change to prismml's runtime, plus the full research patch
bench/        every benchmark, its raw output, and the eval harness
kaggle/       the same eval as a notebook for a free kaggle gpu
docs/         PROGRESS.md, the full log of what was measured and what failed
```

### status

correct against the official runtime: perplexity, greedy output and integer exact kernel tests. only tested on one machine so far, a snapdragon x x126100.

the comparison against qwen3.8 27b at 2 bit and qwen3-8b at the same memory is still running. a partial run of bonsai on the laptop got 58 of 60 on gsm8k and 19 of 19 on the first humaneval problems.

decode is 9.7 tok/s while the kernels alone would allow about 15. the rest is runtime overhead, and that is the next thing to go after.

### why

people say ternary models need special hardware to be worth it. i wanted to see how far a normal laptop cpu goes if the kernels are built for the instructions it already has. so far that is 31 tok/s prefill and 14 tok/s on code edits for a 27b model, with nothing changed in the output.

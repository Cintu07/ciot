a 1.58-bit checkpoint is not a fucking benchmark.

it tells you something about the representation of the weights. it does not tell you how many instructions your processor needs before those weights become useful operands. that missing part is where a surprising amount of inference performance lives.

the argument sounds convincing: smaller weights, less memory traffic, faster model. but compressed data does not arrive at the arithmetic unit politely unpacked into the exact shape it wants. somebody pays for the decoding, shuffling, scaling and reduction. that somebody is your kernel, on every pass through the weights.

ciot was an attempt to put numbers on that bill, using ternary bonsai 27b on a snapdragon x laptop cpu.

the recorded stock pq2_0 path averaged about 8.4 tok/s prefill and 6.5 decode. the t2p research integration reached about 31.7 and 9.7. that is roughly 3.8x prefill and 1.5x decode on the tested paths. these were short pp64/tg32 measurements on eight threads, not a promise about every context length or every version of the runtime.

the smaller weight file was not automatically the faster path either. in the recorded baseline runs, the compact ptq1_0 implementation decoded at about 4.35 tok/s, while pq2_0 did about 6.5. those paths used different kernels. claiming this proves base-3 packing is inherently bad would be bullshit. what it shows is that the format and the decoder have to be evaluated together. a storage advantage can be swallowed by the implementation needed to recover it.

that is the first assumption worth challenging: bits per weight are an input to the performance argument. they are not the conclusion.

the next problem was what happened after loading a packed byte.

arm smmla wants a 2x8 matrix operand. the old t2 layout required weight shuffles to construct it. t2p puts eight weights from one row beside the matching eight weights from the next row. extract a slot and the two rows are already arranged for the instruction.

the model loader does the rearrangement once. the multiply loop then reuses it. the ternary codes and their scales stay the same; their order changes.

this is a useful way to think about data layout: every time the kernel repairs an awkward representation, it is charging the same conversion cost again. if the weights are reused across thousands of tokens, fixing their arrangement before generation can remove work from every one of those tokens.

the row-pair change had its own measurable result. against our earlier t2 kernel, four-token verification throughput went from 862 to 1136 gmac/s. the 64-token prompt kernel went from 850 to 1076. those are 1.32x and 1.27x kernel gains. they are separate from the 3.8x whole-model prefill comparison against stock. mixing those baselines would make the story sound bigger while making it technically worse.

then there is the bit trick that looks questionable until you write down the math.

we store u = w + 1, so the ternary codes are 0, 1 and 2. masking a two-bit slot leaves some codes multiplied by 4 or 16. instead of shifting every slot back into the same range, the kernel keeps separate accumulators for those scaled values and combines them once per 128-weight block.

with a1 holding the low slot, a4 the slot scaled by four, a16 the slot scaled by sixteen, and atop the shifted top slot:

```text
combined = a16 + 4*a4 + 16*(a1 + atop)
corrected = combined - 16*sum(y)
```

the first line gives sixteen times the encoded dot product. the second removes the +1 weight offset, because dot(w + 1, y) includes an extra sum(y). the final scale carries 1/16.

for the sdot path, q8_k already stores block activation sums, so the correction reuses that metadata. the matrix path computes column sums from the activation tile and shares them across weight rows. we avoid subtracting one from every decoded weight.

the top slot still shifts. code 2 in the highest two bits becomes 128, which is negative as signed int8. leaving that one in place would fuck up the arithmetic. the optimization is selective because the signed range of the actual instruction matters more than a neat explanation about eliminating shifts.

this is also why the cuda argument around ternary weights needs more care.

yes, mathematically a ternary multiply can become an addition, a subtraction or nothing. that alone does not prove a custom add/subtract implementation beats the hardware's multiply-and-accumulate path. tensor cores have supported formats and tile shapes. getting packed codes into those operands costs work, but the resulting matrix throughput can still make the complete path worthwhile.

the calculation has to include the conversion, the accesses and the useful outputs produced from each loaded tile. otherwise you are comparing an idealized equation with an implementation that has to do more work than the equation admits.

prefill and decode expose different amounts of reuse. many activation columns let one weight tile contribute to many outputs. one request generating one token has far less of that opportunity. peak matrix throughput cannot settle the decode argument by itself. batching changes the arithmetic intensity, and can change which resource limits execution.

high gpu utilization does not settle it either. a busy device can be busy executing an inefficient path. occupancy measures resident warps relative to a limit; it does not measure whether those warps are doing the best possible work. larger tiles and more accumulators can improve reuse while increasing register pressure and limiting concurrency. you have to measure the trade, not worship one percentage.

our eight-thread sdot matvec reached 107 gb/s. measured dram bandwidth on the machine was around 110 to 115 gb/s. that puts the particular kernel close to the available memory bandwidth. it does not mean the whole model operates at that efficiency.

one profile showed about 19 ms in recurrent-state gather and copy operations. there were also norms, attention and scheduling costs. the model still has to execute all of them after you make its dot products faster.

windows scheduling was another expensive surprise. one worker could get preempted while the other seven waited at a barrier. thousands of small operations accumulated delay. changing priority improved the recorded execution without changing the model math. no amount of staring harder at the dot-product assembly would have identified that cause on its own.

then we found a particularly annoying batch-tail problem.

the matrix kernel handled groups of four activation rows. two or three leftover rows fell back to separate matrix-vector passes. a small remainder in the batch dimension could therefore cause another pass through the weight matrix for each row.

billions of weights, reread because the last few rows did not fit the preferred tile. that is expensive as hell for what looks like a tiny boundary case.

the fix pads those rows into one four-row activation tile, runs the matrix kernel once and copies only the real outputs. the padded rows are zero. this spends some extra arithmetic to avoid repeated weight traffic. optimizing fewer operations and optimizing less execution time are not always the same decision.

speculation makes that trade especially important. drafts only help when verification is cheaper than generating the same continuation serially. prompt lookup provides cheap guesses for code edits because the continuation often copies text already in the prompt. but the target still has to verify them, and rejected guesses must leave its state correct.

bonsai carries recurrent state as well as attention state. rollback therefore involves more than trimming a kv cache. get that wrong and the next token is generated from a history that should no longer exist. a fast wrong answer is a bug, not a performance result.

the recorded code-edit run produced 306 tokens. greedy decoding ran at 8.88 tok/s; speculation reached 14.10, with 76% draft acceptance and zero token mismatches in that check. oracle and deliberately wrong drafts also exercised acceptance and rollback behavior.

but speculation did not win everywhere. an earlier adaptive version was slightly slower on prose. even when drafting backed off, maintaining the recurrent-state snapshot ring still cost time. acceptance rate omits that expense, so it cannot tell you the speedup by itself.

an earlier raw-text check also diverged at a near-tie in the logits. batched and single-token execution can produce small floating-point differences. exact integer accumulation inside a kernel does not automatically prove identical logits throughout the model. the code-edit match is evidence for that run, not permission to claim universal exactness.

the gpu evaluation has its own limits. bonsai got 59/60 on the small math subset, compared with 53/60 for the tested 27b iq2_xxs model and 56/60 for the 8b q6 model. coding tied the 8b at 38/40. these are first-60 and first-40 subsets with output caps, not full benchmark scores.

those runs used the official cuda runtime, and the 27b comparison spans mixed power conditions. they do not establish a ciot cuda-kernel speedup. calling them a controlled gpu speed comparison would be a convenient way to overclaim our own data.

the pattern here is specific. compression can hide decoding work. matrix-throughput claims can hide poor reuse. kernel results can hide runtime costs. draft acceptance can hide verification and snapshots. even a positive accuracy result can hide the limits of the protocol that produced it.

ciot's recorded gains came from measuring those costs and changing the execution path around them. the clean contribution still needs fresh arm validation against current upstream. the next job is to reproduce the result in the exact patch another engineer would build, with checks that fail loudly when its answers change.

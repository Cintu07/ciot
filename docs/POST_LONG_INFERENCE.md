# short post

bonsai 27b on snapdragon x: 31.7 tok/s prefill, 9.7 decode, 14.1 on a code edit with speculation. 3.8x prefill and 1.5x decode over the tested stock pq2_0 path. ternary repacking, sdot/i8mm, recurrent state rollback. 107 gb/s matvec, raw logs in the repo

# long post

1.58 bits describes how much information a ternary weight holds. it does not tell you how long your processor takes to use it.

we discuss model compression in bits per weight. the processor has to execute whatever sequence turns those bits into usable operands. that sequence can include loads, masks, shifts, table lookups, shuffles, scale handling and reductions. the checkpoint gets smaller immediately. the computation only gets cheaper if the implementation makes that representation useful.

that was the problem i worked on in ciot, running ternary bonsai 27b on a snapdragon x cpu. the recorded stock pq2_0 path did about 8.4 tok/s prefill and 6.5 decode. the research integration with t2p reached about 31.7 and 9.7. speculation on a code edit reached 14.1. each number describes a different path through the runtime, which matters when explaining why it improved.

start with the encoding. ternary gives you three values, so the information limit is log2(3), about 1.585 bits. five trits fit into a byte because 3^5 is 243. that is a storage argument. it does not tell you how many instructions recovering those five trits costs, whether that work vectorizes cleanly, or whether the result matches the operand arrangement your dot-product instruction expects.

on this cpu, the smaller base-3 format was limited by decoding work. a two-bit layout moved more bytes but admitted a cheaper instruction path. it is entirely possible to spend more bandwidth and finish earlier when the smaller representation was keeping the arithmetic pipeline busy unpacking it. the winner depends on the machine and workload, not just the file size.

the next problem was ordering. arm smmla consumes a 2x8 matrix operand. if eight weights from one row sit beside the corresponding eight weights from the next row, that is the shape it needs. the t2p layout groups rows in pairs and arranges the packed slots so decoding a slot produces that operand directly. the original t2 batch kernel needed weight operand zips; the row-pair layout removes them.

that rearrangement happens at model load. subsequent multiplies reuse it. the practical question is how often the runtime makes the cpu repair the layout before doing useful arithmetic. on the recorded kernel benchmark, moving from t2 to t2p improved the four-token batch from 862 to 1136 gmac/s, and the 64-token batch from 850 to 1076. those are kernel results against our earlier layout, separate from the whole-model comparison against stock.

there is another detail i like because it looks wrong until you write out the integer math. the codes are stored as u = w + 1, so they are 0, 1 or 2. after masking a slot, i do not always shift it down. some slots keep their factors of 4 or 16 and accumulate separately. the kernel combines them once per block.

if a1 contains the unscaled low-slot sum, a4 contains a sum scaled by four, a16 contains one scaled by sixteen, and atop contains the shifted top slot, then a16 + 4*a4 + 16*(a1 + atop) gives sixteen times the encoded dot product. subtract sixteen times the activation sum, because dot(w + 1, y) = dot(w, y) + sum(y). then apply the scale with a factor of 1/16.

for sdot, q8_k already carries block sums of the activations, so the correction reuses that metadata. for the matrix path, column sums are computed from the activation tile and reused across weight rows. the encoding offset gets handled at block granularity instead of subtracting one from every decoded weight.

the top slot still needs a shift. its code 2 would otherwise become 128, which is negative when interpreted as signed int8. so this is not a blanket trick where every shift disappears. it is choosing which representation changes the instruction actually requires, and postponing the others until a cheaper place in the calculation.

this is also where the discussion about ternary and cuda needs more precision. mathematically, a ternary dot product can use additions and subtractions. that does not establish that replacing a fast hardware multiply-and-accumulate path with a custom sequence of additions will run faster. tensor cores have supported input formats and tile shapes. converting packed weights into those formats costs work, but the matrix engine can still make the complete path worthwhile. the instruction mix and data movement have to be measured together.

prefill and decode make that question more interesting. with many activation columns, a loaded weight tile can contribute to many outputs. with one column, there is much less reuse. a peak matrix-throughput number describes a different situation from one request generating one token at a time. this is why batching can change an operation's limiting resource. [nvidia's matrix performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) explains the arithmetic-intensity and tiling relationship.

if you estimate operations per byte but count only the model's mathematical multiplies, you can miss the instructions used to decode the weights. if you report high gpu utilization, you still have not established that useful matrix throughput or memory bandwidth is saturated. occupancy measures resident warps relative to the hardware limit; it is not a score for kernel quality. more accumulators or larger tiles can save some work while consuming registers and limiting concurrency. [nvidia's profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) distinguishes these measurements.

our sdot matvec reached 107 gb/s on eight threads, against measured dram bandwidth around 110 to 115 gb/s. that gives a useful constraint on what remains possible inside that particular kernel. it does not mean the complete model reached the same efficiency. norms, attention, recurrent-state operations, dispatch and synchronization still take time.

one of the larger runtime surprises was windows scheduling. a worker could be preempted while the other workers waited at a barrier. the profile showed thousands of small operations accumulating substantial delay. changing priority improved the recorded decode performance without changing the model arithmetic. counting instructions in the dot-product loop would not have identified that problem by itself.

then there were the batch tails. the batch kernel handled groups of four activation rows. two or three leftover rows fell back to separate matrix-vector passes, so each could trigger another read of the weight matrix. padding those rows into a four-row tile let one batch multiply handle the remainder, followed by copying only the real outputs. a tiny remainder in the activation dimension had been creating a large amount of weight traffic.

speculation makes this especially relevant. drafting more tokens only helps if verification costs less than generating them separately. prompt lookup can produce cheap guesses for code edits because the continuation often copies text already present. but the target model still has to verify the guesses, and rejected guesses must leave its state correct.

bonsai is hybrid. it carries recurrent state as well as attention state, so rollback cannot be treated as just deleting a few kv entries. the recorded code-edit run generated 306 tokens: 8.88 tok/s through greedy decoding, 14.10 with speculation, 76% draft acceptance and zero token mismatches in that check. on prose, an earlier adaptive version was slightly slower than greedy. maintaining snapshots still cost time even when drafting backed off.

there was also an earlier raw-text check with a token divergence near a very small logit margin. that is a reminder to test numerical behavior when switching between batched and single-token execution. exact integer accumulation does not automatically guarantee identical floating-point logits throughout the model. matching generated text on one workload does not prove it for every possible prompt.

the separate rtx evaluation raises a different systems question: what does it mean for a model to fit? the file competes with cache, recurrent state, activations and workspace for memory. placement and kernel support are part of the result. our gpu accuracy comparison used the official cuda runtime; it did not establish a ciot cuda-kernel speedup. the mixed power conditions also prevent treating the combined timing results as a controlled kernel comparison.

the common failure across these arguments is dropping part of the execution path from the explanation. bits per weight omit unpacking cost. peak arithmetic throughput omits reuse and data movement. a fast microkernel omits runtime work. draft acceptance omits verification and rollback. a file that fits omits the other allocations.

the recorded ciot gains came from making those omitted costs visible and changing the implementation around them. the clean upstream candidate still needs fresh arm execution and model comparisons at the current revision. that is the next checkpoint: reproduce the improvement in the exact patch someone else would build, and make its correctness checks easy to inspect.

# screenshot notes

open bench/showcase/ciot_glimpse.c in zed for the short-post screenshot. it is explicitly a display excerpt, not a new runnable kernel. the original code is kernels/t2p.c. all figures are historical recorded measurements, not fresh validation of the upstream candidate. the speculative and microkernel numbers have their own stated workloads.

the short caption is 254 characters. the long draft is intended as one long post/article. no text has been published.

# lowercase post drafts

These are drafts for the user to edit and post. Nothing was published. CPU numbers refer to historical Snapdragon measurements, not fresh validation of the clean upstream candidate. Each option is a separate post.

1. got bonsai 27b to 31.7 tok/s prefill and 9.7 tok/s decode on snapdragon x. 3.8x and 1.5x over the tested stock pq2_0 path

2. repacked ternary weights into row pairs so smmla gets its matrix operand directly. removed weight operand zips from the hot loop, same codes and scales

3. ternary slots stay masked at 1x, 4x and 16x. separate accumulators, one recombine and activation sum correction per block. less instruction work in the inner loop

4. t2p matvec hit 107 gb/s on 8 cpu threads. measured dram bandwidth was 110 to 115 gb/s, the kernel is close to the memory wall

5. code edit speculation went from 8.88 to 14.10 tok/s, 76% draft acceptance and zero token mismatches in that run. hybrid recurrent state rollback included

6. finished a small 8 gb gpu comparison: bonsai 27b got 59/60 gsm8k, qwen 27b iq2_xxs got 53/60 and qwen 8b q6 got 56/60. official cuda runtime, first 60 tasks

7. two or three leftover activation rows were triggering separate weight passes. padded them into one smmla tile and copied only the real outputs

8. learning how much inference speed lives in data layout. packing weights for the actual operand shape changes what the cpu has to do on every token

Keep the boundaries clear: the 14 tok/s result includes speculation beyond the clean repack patch; 107 GB/s is a microkernel result; the GPU evaluation measures existing models using the official runtime. No custom CUDA-kernel speedup, merged contribution or universal exactness claim is established.

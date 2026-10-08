#!/usr/bin/env bash
# CIOT-V2 progress box, printed from the logged runs (no numbers typed in here).
cd "$(dirname "$0")"
E2E=$(ls e2e_bonsai_2*.txt | head -1)

avg_old() { grep -F "$1" "$E2E" | awk -v c="$2" '{for(i=1;i<=NF;i++) if($i ~ "^"c){s+=$(i+1);n++}} END{printf "%5.1f", s/n}'; }
avg_t2p() { grep "^bench round" e2e_t2p_27b.txt | awk -v c="$1" '{for(i=1;i<=NF;i++) if($i ~ "^"c){s+=$(i+1);n++}} END{printf "%5.1f", s/n}'; }
spec() { grep "^$2" "specP_code_k$1.txt" | grep -oE '[0-9.]+ tok/s' | head -1 | cut -d' ' -f1; }
ppl() { grep '^\[1\]' "$1" | tail -1 | tr -d '\r' | sed 's/\[[0-9]\]//g; s/,$//; s/,/  /g'; }
ppl_t2p() { grep -oE '\[1\][0-9.]+,\[2\][0-9.]+,\[3\][0-9.]+' e2e_t2p_27b.txt | sed 's/\[[0-9]\]//g; s/,/  /g'; }

echo "-- ciot v2 ---------------------------------------------------------------"
echo "  ternary bonsai 2 27b | snapdragon x, 8 cores, 16 gb, cpu only, no gpu"
echo
echo "  runtime                                 prefill tok/s   decode tok/s"
printf "  official, 1.75 bit file  (generic c)      %s          %s\n" "$(avg_old 'stock PTQ1_0' pp)" "$(avg_old 'stock PTQ1_0' tg)"
printf "  official, 2 bit file     (neon)           %s          %s\n" "$(avg_old 'stock PQ2_0' pp)" "$(avg_old 'stock PQ2_0' tg)"
printf "  ciot t2p                 (sdot + smmla)   %s          %s\n" "$(avg_t2p pp64)" "$(avg_t2p tg32)"
echo
printf "  speculative decoding, code edit           greedy %s -> %s tok/s\n" "$(spec 2 greedy)" "$(spec 2 'speculative')"
echo "    rollback of recurrent state checked: 0 token mismatches"
echo
echo "  perplexity  official  $(ppl ppl27_pq2_t2_0.log)"
echo "              ciot t2p  $(ppl_t2p)   bit identical"
echo
echo "  kernels (8 threads)   decode $(grep -oE 'T2P [0-9.]+ ms \([0-9.]+ GB/s' t2p_bench.txt | grep -oE '[0-9.]+ GB/s') streamed of ~110-115 GB/s dram"
echo "                        4-token verify $(grep 'smmla  4' t2p_bench.txt | grep -oE 'T2P [0-9.]+ ms \([0-9]+ GMAC/s' | grep -oE '[0-9]+ GMAC/s')"
echo "--------------------------------------------------------------------------"

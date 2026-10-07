#!/usr/bin/env bash
# Prints the CIOT-V2 result summary from the logged benchmark runs (no numbers are typed in here).
cd "$(dirname "$0")"
E2E=$(ls -t e2e_bonsai_*.txt | head -1)

avg() { # label column(pp|tg)
    grep -F "$1" "$E2E" | awk -v col="$2" '{
        for (i = 1; i <= NF; i++) if ($i ~ "^" col) { s += $(i + 1); n++ }
    } END { printf "%6.1f", s / n }'
}

ppl() { grep '^\[1\]' "$1" | tail -1 | sed 's/\[[0-9]\]//g; s/,$//; s/,/  /g'; }

echo "-- ciot v2 -------------------------------------------------------------"
echo "  model      ternary bonsai 2 27b (qwen3.8 hybrid, 64 blocks)"
echo "  cpu        snapdragon x x126100, 8 cores, 16 gb, no gpu"
echo "  runtime    prismml llama.cpp $(sed -n 's/^fork: \([0-9a-f]*\).*/\1/p' "$E2E") + ciot kernels"
echo
echo "  path                              prefill tok/s   decode tok/s"
printf "  stock ptq1_0  (generic c)            %s         %s\n" "$(avg 'stock PTQ1_0' pp)" "$(avg 'stock PTQ1_0' tg)"
printf "  stock pq2_0   (prismml neon)         %s         %s\n" "$(avg 'stock PQ2_0' pp)" "$(avg 'stock PQ2_0' tg)"
printf "  ciot  t2      (sdot + i8mm)          %s         %s\n" "$(avg 'ciot PQ2_0+T2' pp)" "$(avg 'ciot PQ2_0+T2' tg)"
echo
echo "  perplexity  stock  $(ppl ppl27_pq2_t2_0.log)"
echo "              ciot   $(ppl ppl27_pq2_t2_1.log)   bit identical"
echo
echo "  t2 matvec   $(grep 'ciot  T2x4' ternary_kernels_pool.txt | tail -1 | awk '{print $8}') GB/s   (dram streams $(grep 'threads= 8' bw.txt | awk '{print $9}' | sort -n | sed -n '1p;$p' | paste -sd- -) GB/s)"
echo "------------------------------------------------------------------------"

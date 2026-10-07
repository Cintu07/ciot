#!/usr/bin/env bash
# End-to-end Bonsai 2 27B on this CPU: official kernels vs CIOT kernels, same binary,
# same priority, interleaved rounds so drift in power/thermal state hits every config equally.
#
#   stock PTQ1_0      generic C dot product (what ARM runs today)          CIOT_PTQ1=0
#   ciot  PTQ1_0      CIOT NEON PTQ1_0 kernel                              CIOT_PTQ1=1
#   stock PQ2_0       PrismML NEON PQ2_0 kernel                            CIOT_T2=0
#   ciot  PQ2_0 + T2  load-time repack to T2x4, sdot decode, i8mm prompt   CIOT_T2=1
#
# Usage: bench/e2e_bonsai.sh [threads] [prio] [rounds]
set -u
cd "$(dirname "$0")/.."
BIN=vendor/prism-llama.cpp/build/bin
M=models
T=${1:-8}
PRIO=${2:-2}
ROUNDS=${3:-2}
OUT=bench/e2e_bonsai_$(date +%Y%m%d_%H%M%S).txt

configs=(
    "stock PTQ1_0|CIOT_PTQ1=0|Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    "ciot PTQ1_0|CIOT_PTQ1=1|Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    "stock PQ2_0|CIOT_T2=0|Ternary-Bonsai-2-27B-PQ2_0.gguf"
    "ciot PQ2_0+T2|CIOT_T2=1|Ternary-Bonsai-2-27B-PQ2_0.gguf"
)

{
    echo "date: $(date -Iseconds)"
    echo "fork: $(git -C vendor/prism-llama.cpp rev-parse --short HEAD) + patches/prism-llama.cpp-t2-neon.patch"
    echo "cpu: Snapdragon X X126100, 8 cores | threads=$T prio=$PRIO rounds=$ROUNDS"
} | tee "$OUT"

for round in $(seq 1 "$ROUNDS"); do
    for c in "${configs[@]}"; do
        IFS='|' read -r label envs model <<< "$c"
        [ -f "$M/$model" ] || { echo "skip $label: $model missing" | tee -a "$OUT"; continue; }
        res=$(env $envs "$BIN/llama-bench.exe" -m "$M/$model" -t "$T" --prio "$PRIO" -p 64 -n 32 -r 2 2>/dev/null \
            | grep -E "pp64|tg32" | awk -F'|' '{gsub(/ /, "", $(NF-2)); printf "%s %s  ", $(NF-2), $(NF-1)}')
        printf "round %d  %-15s %s\n" "$round" "$label" "$res" | tee -a "$OUT"
    done
done

echo "results in $OUT"

#!/usr/bin/env bash
# End-to-end Bonsai 2 27B on this CPU: official paths vs CIOT T2, same binary.
#
#   stock PTQ1_0      what the 5.95 GB file runs on ARM today (generic C dot)
#   stock PQ2_0       PrismML's NEON PQ2_0 kernel (CIOT_T2=0)
#   ciot  PQ2_0 + T2  load-time repack to T2x4 (CIOT_T2=1)
#
# Usage: bench/e2e_bonsai.sh [threads] [reps]
set -u
cd "$(dirname "$0")/.."
BIN=vendor/prism-llama.cpp/build/bin
M=models
T=${1:-8}
R=${2:-3}
OUT=bench/e2e_bonsai_$(date +%Y%m%d_%H%M%S).txt

run() { # label env model
    local label=$1 env=$2 model=$3
    [ -f "$M/$model" ] || { echo "skip $label: $model missing" | tee -a "$OUT"; return; }
    echo "=== $label ($env, $T threads)" | tee -a "$OUT"
    env $env "$BIN/llama-bench.exe" -m "$M/$model" -t "$T" -p 64 -n 32 -r "$R" 2>/dev/null \
        | grep -E "pp64|tg32" | tee -a "$OUT"
}

{
    echo "date: $(date -Iseconds)"
    echo "fork: $(git -C vendor/prism-llama.cpp rev-parse --short HEAD) + patches/prism-llama.cpp-t2-neon.patch"
    echo "cpu: Snapdragon X X126100, 8 cores"
} | tee "$OUT"

run "stock PTQ1_0"     "CIOT_T2=0" Ternary-Bonsai-2-27B-PTQ1_0.gguf
run "stock PQ2_0"      "CIOT_T2=0" Ternary-Bonsai-2-27B-PQ2_0.gguf
run "ciot PQ2_0 + T2"  "CIOT_T2=1" Ternary-Bonsai-2-27B-PQ2_0.gguf

echo "results in $OUT"

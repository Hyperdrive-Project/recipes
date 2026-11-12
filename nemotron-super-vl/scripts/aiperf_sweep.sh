#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# usage: scripts/aiperf_sweep.sh <isl> <osl> "<concurrencies>" [out dir]
#   B200/B300 curve:  scripts/aiperf_sweep.sh 8192 1024 "1 8 16 32 64 128"
#   Spark finals:     scripts/aiperf_sweep.sh 4096 1024 "1 2 4 8"
# NVIDIA AIPerf 0.13.0 (pip install aiperf==0.13.0). Every concurrency point gets its OWN random seed, so no point re-sends
# prompts that an earlier point left in the prefix cache, and --num-dataset-entries N+C makes every prompt in a point
# unique: without it AIPerf 0.13 replays the C warm-up prompts in the measured phase (~50% prefix-cache hits), which is how
# the October B200/B300 curves (4,612, 7,851) were measured. UNIQUE=0 reproduces that old protocol.
set -euo pipefail
ISL=$1; OSL=$2; CS=$3; OUT=${4:-results/aiperf-$(date -u +%Y%m%dT%H%M%SZ)}; TOK=${TOKENIZER:-./models/super-vl-nvfp4}
mkdir -p "$OUT"; i=0
for C in $CS; do
  i=$((i+1)); SEED=$((1000 + i*7919 + ISL)); N=$(( C*2 < 16 ? 16 : C*2 ))
  UNIQ=(); [ "${UNIQUE:-1}" = 1 ] && UNIQ=(--num-dataset-entries $((N + C)))
  CMD=(aiperf profile --model nemotron-3.5-super-vl --url http://127.0.0.1:8000 --endpoint-type chat --streaming
    --tokenizer "$TOK" --tokenizer-trust-remote-code --isl "$ISL" --isl-stddev 0 --osl "$OSL" --osl-stddev 0
    --extra-inputs ignore_eos:true --extra-inputs temperature:1.0 --extra-inputs top_p:0.95
    --concurrency "$C" --request-count "$N" --warmup-request-count "$C" --num-profile-runs 1 --random-seed "$SEED"
    "${UNIQ[@]}"
    --artifact-dir "$OUT/isl$ISL-osl$OSL-c$C" --ui-type simple)
  echo "${CMD[*]}" > "$OUT/isl$ISL-osl$OSL-c$C.cmd.txt"
  "${CMD[@]}" 2>&1 | tee "$OUT/isl$ISL-osl$OSL-c$C.log" | grep -E 'Output Token Throughput|Time to First Token' || true
done

#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# Nemotron 3.5 Super VL (NVFP4) on stock vLLM, Hyperdrive settings. One GPU, no patches.
# usage: GPU=b200|b300 configs/vllm/serve-hyperdrive.sh [extra vllm args...]
#   docker: vllm/vllm-openai:v0.32.0 (sha256:befccf0e7e40...), model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 rev 59b87216
# Measured (AIPerf 8K in / 1K out, unique seed per point but ~50% warm-up repeats hitting the prefix cache, 2026-10-30): B200 5,567-5,650 output tok/s @128 users,
# B300 6,470 @160 / 7,007 @256. Quality reasoning on: math 49/50, code 29/30, tools 11/11 on both.
# Differences vs the stock Dynamo vLLM recipe (ai-dynamo/dynamo recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b200-swebenchpro):
#   MTP 3 instead of 5, explicit max-num-seqs, 32K batched tokens instead of 16K, async scheduling, stream interval 4,
#   no Mamba stochastic rounding. Everything else is the stock recipe.
set -euo pipefail
echo "Hyperdrive recipes: https://github.com/Hyperdrive-Project/recipes (Apache-2.0)" >&2
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4}
case "${GPU:-b200}" in
  b200) SEQS=128 ;;   # B200 peaks at 128 users (5,650); 160+ users lose throughput
  b300) SEQS=256 ;;   # B300 keeps rising to 256 users (7,007); use 160 for ~56 tok/s per user at 6,470
  *) echo "GPU must be b200 or b300" >&2; exit 2 ;;
esac
exec vllm serve "$MODEL" --served-model-name nemotron-3.5-super-vl --trust-remote-code --tensor-parallel-size 1 \
  --max-model-len 131072 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 \
  --enable-prefix-caching --mamba-cache-mode align --block-size 64 --enable-mamba-shared-prefix-checkpoint \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --kernel-config '{"linear_backend_per_quant":{"fp8_w8a8":"torch"}}' \
  --compilation-config '{"pass_config":{"fuse_attn_quant":true},"use_inductor_graph_partition":true,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --max-num-seqs $SEQS --max-num-batched-tokens 32768 --stream-interval 4 --async-scheduling \
  --speculative-config '{"method":"mtp","disable_eagle_block_drop":true,"num_speculative_tokens":3,"moe_backend":"auto"}' \
  --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder "$@"

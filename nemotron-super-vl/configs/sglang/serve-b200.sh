#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# Super VL NVFP4 on one B200 with SGLang, Hyperdrive settings (2026-11-07). Stock image, no patches.
#   configs/sglang/serve-b200.sh [./models]          then: scripts/wait_ready.sh
# Measured (8K in / 1K out, unique prompts, AIPerf 0.13): 477 / 1,521 / 2,777 / 3,449 / 3,705 / 3,622 output tok/s at
# 1 / 8 / 32 / 64 / 128 / 160 users (cookbook defaults: 2,347 @128u). Quality, reasoning on: math 48/50, code 29/30,
# tools 11/11 (vLLM reference on the same model: 49/50, 29/30, 11/11). Text only; vision and B300 not measured.
# What matters: SGLang caps max_running_requests by the Mamba state cache (27 at 128 users with MTP on the defaults).
# BF16 Mamba state halves each request's state and a 6x Mamba memory share makes room, so 98 requests run at once.
# BF16 Mamba state changes the numerics slightly; that is why the quality gate above was run.
# Env: IMAGE (pin it: docker image inspect --format '{{.RepoDigests}}' after the pull), DOCKER (default "sudo docker").
set -euo pipefail
echo "Hyperdrive recipes: https://github.com/Hyperdrive-Project/recipes (Apache-2.0)" >&2
MODELS=$(realpath "${1:-./models}"); DOCKER=${DOCKER:-sudo docker}
IMAGE=${IMAGE:-lmsysorg/sglang:nemotron3-5-super}
$DOCKER run -d --name nhd-sglang --gpus=all --ipc=host --network=host --shm-size=16g -v "$MODELS":/models:ro \
  -e HF_HUB_OFFLINE=1 "$IMAGE" \
  python3 -m sglang.launch_server --model-path /models/super-vl-nvfp4 --served-model-name nemotron-3.5-super-vl \
    --host 127.0.0.1 --port 8000 --trust-remote-code --tp 1 --context-length 131072 --mem-fraction-static 0.88 \
    --kv-cache-dtype fp8_e4m3 --reasoning-parser nemotron_3 --tool-call-parser qwen3_coder \
    --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 32768 --max-running-requests 128 --cuda-graph-max-bs-decode 128 --stream-interval 4 \
    --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 6
echo "started nhd-sglang (cold load 12-14 min, warm ~2.5 min); stop: $DOCKER rm -f nhd-sglang"

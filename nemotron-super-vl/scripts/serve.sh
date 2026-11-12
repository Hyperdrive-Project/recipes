#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# usage: scripts/serve.sh <config.yaml> [env file] [extra trtllm-serve args...]
#   scripts/serve.sh configs/spark/hyperdrive-spark.yaml configs/spark/hyperdrive-spark.env
#   IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc1 scripts/serve.sh configs/nvidia-card/model-card-spark-rc1.yaml   # stock baseline
# Starts trtllm-serve in a detached container "nhd-server" on 127.0.0.1:8000 (OpenAI-compatible). Logs: runs/<timestamp>/.
# Env: IMAGE (default nemotron-hyperdrive:1.4.0rc1, see Dockerfile), MODELS (default ./models), MODEL (checkpoint directory
#   under MODELS, default super-vl-nvfp4; the v3 recipes serve super-vl-nvfp4-m4 from scripts/make_fp4_mamba.sh),
#   CACHE (default ./cache, mounted at /cache for kernel JIT caches), DOCKER (default "sudo docker").
set -euo pipefail
echo "Hyperdrive recipes: https://github.com/Hyperdrive-Project/recipes (Apache-2.0)" >&2
cd "$(dirname "$0")/.."
CFG=$(realpath "$1"); ENVF=${2:-}; shift; [ $# -gt 0 ] && shift
IMAGE=${IMAGE:-nemotron-hyperdrive:1.4.0rc1}; MODELS=$(realpath "${MODELS:-./models}"); DOCKER=${DOCKER:-sudo docker}
MODEL=${MODEL:-super-vl-nvfp4}; [ -d "$MODELS/$MODEL" ] || { echo "no checkpoint at $MODELS/$MODEL" >&2; exit 1; }
mkdir -p "${CACHE:-./cache}"; CACHE=$(realpath "${CACHE:-./cache}")
$DOCKER ps --format '{{.Names}}' | grep -q '^nhd-server$' && { echo "nhd-server already running (scripts/stop.sh)" >&2; exit 1; }
RUN=runs/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$RUN"; cp "$CFG" "$RUN/config.yaml"; echo "$MODEL" > "$RUN/model.txt"
ENVARGS=(); if [ -n "$ENVF" ]; then cp "$ENVF" "$RUN/env"; ENVARGS=(--env-file "$(realpath "$ENVF")"); fi
$DOCKER image inspect "$IMAGE" --format '{{.Id}} {{.RepoDigests}}' > "$RUN/image.txt"
git rev-parse HEAD > "$RUN/repo_commit.txt" 2>/dev/null || true
$DOCKER run -d --name nhd-server --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all --network=host \
  -v "$MODELS":/models:ro -v "$CACHE":/cache -v "$CFG":/config.yaml:ro -e HF_HUB_OFFLINE=1 "${ENVARGS[@]}" \
  "$IMAGE" trtllm-serve "/models/$MODEL" --config /config.yaml --served_model_name nemotron-3.5-super-vl \
  --host 127.0.0.1 --port 8000 --reasoning_parser nemotron-v3 --tool_parser qwen3_coder "$@" > "$RUN/container_id.txt"
( $DOCKER logs -f nhd-server > "$RUN/server.log" 2>&1 & )
echo "started; logs in $RUN/server.log (cold start ~9 min on Spark)"

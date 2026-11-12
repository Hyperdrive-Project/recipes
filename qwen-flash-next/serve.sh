#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# Qwen Flash-Next Hyperdrive on one DGX Spark: the Oct 30 SparkBench recipe plus FancyCake 48 (on by default).
#   qwen-flash-next/serve.sh                 FANCYCAKE=0 for the exact SparkBench-submitted config
# Image: eugr's vLLM b12x build, pinned by digest (the only image that loads FP4-Local's compressed NVFP4 scales).
# Overlays (all speed-only; outputs keep the target model's distribution):
#   mods/                                   34K MTP draft vocabulary (mtp.py + ids_34k.txt), recurrent checkpoint
#                                           8 tokens before each prompt end (boundary_checkpoint.py)
#   fancycake-48/                           static NVFP4 expert scales, by FancyCake (decode 5.2% faster, smaller KV cache)
# Model: nvidia/Qwen3.8-Flash-Next-FP4-Local @ b9ce18a6 in the Hugging Face cache (hf download ... --revision b9ce18a6...).
# Env: PORT (8000), NAME (hd-qwen), UTIL (0.8), FANCYCAKE (1), HF_HOME (~/.cache/huggingface), DOCKER ("sudo docker").
# Keep resident memory well under ~100 GiB on the Spark (drop the page cache before loading; see README).
set -euo pipefail
echo "Hyperdrive recipes: https://github.com/Hyperdrive-Project/recipes (Apache-2.0)" >&2
Q=$(cd "$(dirname "$0")" && pwd); MOD=$Q/mods
NAME=${NAME:-hd-qwen}; PORT=${PORT:-8000}; DOCKER=${DOCKER:-sudo docker}; HFC=${HF_HOME:-$HOME/.cache/huggingface}
I=eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
PY=/usr/local/lib/python3.12/dist-packages
SNAP=/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300
OV=(-v "$MOD/mtp.py:$PY/vllm/models/qwen4_exp/nvidia/mtp.py:ro"
    -v "$MOD/boundary_checkpoint.py:$PY/vllm/v1/core/boundary_checkpoint.py:ro"
    -v "$MOD/ids_34k.txt:/opt/hyperdrive/ids_34k.txt:ro"
    -e HYPERDRIVE_DRAFT_VOCAB=/opt/hyperdrive/ids_34k.txt -e HYPERDRIVE_TAIL_OFFSET=8)
if [ "${FANCYCAKE:-1}" = 1 ]; then
  OV+=(-v "$Q/fancycake-48/nvfp4_csf.py:$PY/vllm/model_executor/layers/quantization/nvfp4_csf.py:ro"
       -e HYPERDRIVE_CSF_STATIC=1 -e HYPERDRIVE_CSF_STATIC_DEFER=1 -e HYPERDRIVE_CSF_STATIC_FREE=1)
fi
mkdir -p "$HOME/.cache/vllm" "$HOME/.cache/flashinfer" "$HOME/.triton" "$HOME/.cache/b12x"
$DOCKER ps -a --format '{{.Names}}' | grep -q "^$NAME$" && { echo "$NAME exists ($DOCKER rm -f $NAME)" >&2; exit 1; }
$DOCKER run -d --name "$NAME" --gpus all --ipc host --network host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -e HF_HUB_OFFLINE=1 -e VLLM_PLE_TABLE_MEMORY=disk -e B12X_DISK_BACKEND=io_uring -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 -e B12X_POLICY_MODE=auto \
  -e VLLM_MXFP8_LM_HEAD=1 -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e B12X_W4A16_A4_PREFILL=1 "${OV[@]}" \
  -v "$HFC":/root/.cache/huggingface -v "$HOME/.cache/vllm":/root/.cache/vllm \
  -v "$HOME/.cache/flashinfer":/root/.cache/flashinfer -v "$HOME/.triton":/root/.triton -v "$HOME/.cache/b12x":/root/.cache/b12x \
  --entrypoint vllm "$I" serve "$SNAP" \
    --served-model-name qwen3.8-flash-next --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --mamba-cache-mode align --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16 \
    --kv-cache-dtype fp8 --quantization modelopt_mixed --block-size 16 --load-format b12x \
    --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"mtp","draft_sample_method":"probabilistic","rejection_sample_method":"standard","num_speculative_tokens":3}' \
    --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x --no-enable-flashinfer-autotune \
    --mm-encoder-tp-mode data --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    --gpu-memory-utilization "${UTIL:-0.8}" > /dev/null
for i in $(seq 240); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null && { echo "ready after $((i*10))s (FANCYCAKE=${FANCYCAKE:-1})"; exit 0; }
  $DOCKER ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo "container died"; $DOCKER logs --tail 40 "$NAME"; exit 1; }
  sleep 10
done; echo timeout; exit 1

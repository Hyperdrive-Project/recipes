#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# Voice brain: NVIDIA Nemotron 3.5 Bolt on NVIDIA Flare (our patch series, patches/flare-bolt-robot/), tuned for short
# spoken turns on one DGX Spark. OpenAI-compatible server on http://${HOST:-0.0.0.0}:${PORT:-8000}/v1 (model name "bolt").
#   - package exported for 2 concurrent requests with LiLiCorr speculation (see README, "Build and export")
#   - throughput scheduler; short prompt tails (<=512 new tokens) prefill in one step (cached turn ~0.09 s to first token)
#   - WARM_PROMPT=path/to/system_prompt.txt (default serve/persona.txt) is sent once at boot so every session's first
#     turn is cached; it must hold exactly the system message your client sends. WARM_PROMPT= disables it.
# For chat turns send "chat_template_kwargs": {"enable_thinking": false}; reasoning turns keep their last message
# checkpoint automatically (~0.17 s).
# Env: FL (Flare checkout, ~/flare-bolt), FX (dir with flare-config.toml, development-publisher.json, trust.toml),
#      PKG (exported package), PORT, HOST, MEMORY_LIMIT (60G), DRAFTS (4), MPS (0), EXTRA (extra flare args).
# Usage: serve/serve-bolt.sh   (foreground; Ctrl-C to stop)
set -eu
echo "Hyperdrive recipes: https://github.com/Hyperdrive-Project/recipes (Apache-2.0)" >&2
HERE=$(cd "$(dirname "$0")" && pwd)
FX=${FX:-$HOME/flare-voice}
FL=${FL:-$HOME/flare-bolt}
PKG=${PKG:-$FX/packages/bolt-lilicorr-c2-speech.safetensors}
PORT=${PORT:-8000}; HOST=${HOST:-0.0.0.0}
MEM=${MEMORY_LIMIT:-60G}  # Bolt ~28 GiB + ~30 GiB cache; leaves ~60 GiB for voice models and the OS
# Verify 4 drafts per round (verify-width sweep, 2026-10-25): 60.8 vs 58.8 tok/s and the 8th token 217 vs 230 ms on held-out voice turns.
TAIL="--prefill-short-tail 512 --prefill-short-tail-keep-last"  # auto: one pass for reasoning-off turns, keep-last for reasoning turns
sudo -n prlimit --pid $$ --memlock=16332894208:16332894208 2>/dev/null || true
FP=$(cd "$FL" && CUDA_VISIBLE_DEVICES='' .venv/bin/python - "$FX/flare-config.toml" <<'PY'
import sys
from pathlib import Path
from ignition.cli.config import Configuration
from ignition.development import publisher
print(publisher(Configuration.load(Path(sys.argv[1]))).fingerprint.hex())
PY
)
WARM_PROMPT=${WARM_PROMPT-$HERE/persona.txt}
if [ -n "${WARM_PROMPT:-}" ]; then
  ( for _ in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
    python3 - "$WARM_PROMPT" $PORT <<'PY'
import json, sys, urllib.request
system = open(sys.argv[1]).read(); port = sys.argv[2]
body = {"model": "bolt", "messages": [{"role": "system", "content": system}, {"role": "user", "content": "Hello."}],
        "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                       {"Content-Type": "application/json"}), timeout=120).read()
print("[serve-bolt] system prompt warmed into the prefix cache")
PY
  ) &
fi
# MPS=1: join (or start) a per-user CUDA MPS daemon shared with your speech models, so Bolt and STT/TTS kernels
# run side by side instead of time-slicing; Bolt is capped at MPS_PCT% of SM threads (default 70).
if [ "${MPS:-0}" = 1 ]; then
  export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/hv-mps/pipe} CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/hv-mps/log}
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  echo get_server_list | nvidia-cuda-mps-control > /dev/null 2>&1 || nvidia-cuda-mps-control -d
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${MPS_PCT:-70}
fi
export CUDA_HOME=/usr/local/cuda-13.0
exec $FL/flare/target/release/flare run "$PKG" --gpus 0 --host $HOST --port $PORT --served-model-name bolt \
  --publisher-keys "$FX/development-publisher.json" --dangerously-approve-key "$FP" \
  --dangerously-accept-publisher local-development --trust-store "$FX/trust.toml" \
  --scheduler-policy throughput --memory-limit $MEM $TAIL \
  --no-mtp-adaptive-verification --mtp-draft-tokens ${DRAFTS:-4} ${EXTRA:-}  # fixed-width verify (adaptive is slower for voice); 4 drafts by default, DRAFTS=N to change

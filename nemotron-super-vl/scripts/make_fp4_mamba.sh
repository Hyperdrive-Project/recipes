#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# usage: scripts/make_fp4_mamba.sh            (once per checkpoint, ~1 min on a B200/B300, ~2.3 GB of disk)
# Makes the v3 checkpoint: the 80 Mamba-2 projections (in_proj + out_proj of 40 layers) re-quantized from the checkpoint's static FP8 to
# NVFP4 (W4A4, 16-element block scales). A model change, kept because the reasoning-on quality gate held (B300 v3:
# math 49/50, code 29/30, tools 11/11, same as the original model). Writes $MODELS/super-vl-nvfp4-m4: symlinks to the
# original shards (absolute /models/... paths, so serve it through scripts/serve.sh's /models mount) plus
# nhd-m4-overlay.st, which the v3 image's weight-overlay patch loads via NHD_WEIGHT_OVERLAY.
# Needs the v3 image (docker build --build-arg NHD_STACK=v3 ...) and a GPU (the codec is checked against TRT-LLM's own
# fp4_quantize). Env: IMAGE (default nemotron-hyperdrive:1.4.0rc1-v3), MODELS (default ./models), DOCKER.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-nemotron-hyperdrive:1.4.0rc1-v3}; MODELS=$(realpath "${MODELS:-./models}"); DOCKER=${DOCKER:-sudo docker}
[ -f "$MODELS/super-vl-nvfp4/config.json" ] || { echo "no checkpoint at $MODELS/super-vl-nvfp4 (scripts/download_models.sh)" >&2; exit 1; }
$DOCKER run --rm --gpus=all --ipc=host -v "$MODELS":/models -v "$PWD/tools":/nhd-jobs:ro "$IMAGE" \
  python3 /nhd-jobs/m4_convert.py /models/super-vl-nvfp4 /models/super-vl-nvfp4-m4 --targets in_proj,out_proj --mode overlay
ls -l "$MODELS/super-vl-nvfp4-m4/nhd-m4-overlay.st"

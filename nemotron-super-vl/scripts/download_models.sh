#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# usage: download_models.sh [models dir]   (needs `pip install -U huggingface_hub`; ~70 GB + drafter)
# Pinned revisions used for every result in this repo.
set -euo pipefail
M=${1:-./models}; mkdir -p "$M"
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --revision 59b8721646c38111a242da18218859fb2b454322 --local-dir "$M/super-vl-nvfp4"
# LiLiCorr DFlash drafter: Spark and card recipes only (B200/B300 use the model's built-in MTP heads).
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr --revision 292bf1efc06ba39567da7351f2ad7a522afa3959 --local-dir "$M/super-vl-lilicorr"

#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# usage: fetch_flashinfer_cake.sh [dest dir]   (default /opt/nhd/fimain-e74a167)
# FlashInfer main's CAKE Mamba-2 SSD sources (flashinfer/mamba/cake_ssd_combined.py + csrc/cake_mamba_ssd_combined),
# pinned, laid out like an installed package so the cake-ssd patch loads them with NHD_CAKE_FI_DIR=<dest dir>. Main has
# the tcgen05 fences (FlashInfer #6808) that the 0.7.2 wheels lack, the SM103 one-launch path (#6715) and the look-back
# cap (#6884). The same sources the measured runs used. Idempotent.
set -euo pipefail
SHA=${FIMAIN_SHA:-e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34}
DST=${1:-/opt/nhd/fimain-${SHA:0:7}}
[ -f "$DST/.ok" ] && { echo "already there: $DST"; exit 0; }
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }
TMP=$DST.tmp; rm -rf "$TMP"; mkdir -p "$TMP/flashinfer/mamba" "$TMP/flashinfer/data/csrc"
git init -q "$TMP/src"
git -C "$TMP/src" remote add origin https://github.com/flashinfer-ai/flashinfer
git -C "$TMP/src" fetch -q --depth 1 --filter=blob:none origin "$SHA"
git -C "$TMP/src" checkout -q "$SHA" -- flashinfer/mamba/cake_ssd_combined.py csrc/cake_mamba_ssd_combined
cp "$TMP/src/flashinfer/mamba/cake_ssd_combined.py" "$TMP/flashinfer/mamba/"
cp -r "$TMP/src/csrc/cake_mamba_ssd_combined" "$TMP/flashinfer/data/csrc/"
rm -rf "$TMP/src" "$DST"; mv "$TMP" "$DST"; touch "$DST/.ok"
echo "FlashInfer ${SHA:0:7} CAKE SSD: $(find "$DST" -name '*.cu' | wc -l) CUDA sources in $DST"

#!/bin/bash
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# usage: apply_patches.sh [patch dir] [site-packages dir]
# Applies the numbered diffs in [patch dir] ([0-9]*.diff: patches/hyperdrive/01-07, or the v3 set 11-15), in order, to TensorRT-LLM 1.4.0rc1 (commit 0372799d). Dry-runs each patch before
# applying it, and if any hunk fails it reverts the patches it already applied, so the tree is unchanged.
# Works inside the NGC image (Docker build or a cloud GPU instance running the image).
set -euo pipefail
P=${1:-$(cd "$(dirname "$0")/../patches/hyperdrive" && pwd)}
SP=${2:-$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')}
command -v patch >/dev/null || { apt-get update -qq && apt-get install -y -qq patch >/dev/null; }
python3 - <<'PY'
import tensorrt_llm, sys
v = tensorrt_llm.__version__
if v != "1.4.0rc1": sys.exit(f"expected tensorrt_llm 1.4.0rc1, found {v}")
PY
cd "$SP"
# Each patch is dry-run on top of the ones before it (they stack), so a failure can only show up after earlier patches
# went in: undo those, newest first, so the tree is left as it was.
DONE=()
undo() { local i; for (( i=${#DONE[@]}-1; i>=0; i-- )); do patch -p1 -R --batch -s < "${DONE[$i]}" || echo "could not undo ${DONE[$i]}" >&2; done; }
for d in "$P"/[0-9]*.diff; do
  patch -p1 --forward --batch --dry-run -s < "$d" || { echo "dry run failed: $d; undoing ${#DONE[@]} applied patch(es)" >&2; undo; exit 1; }
  patch -p1 --forward --batch -s < "$d" || { echo "apply failed: $d" >&2; undo; exit 1; }
  DONE+=("$d"); echo "applied $(basename "$d")"; done

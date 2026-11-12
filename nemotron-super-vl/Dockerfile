# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
# Hyperdrive image: stock NGC TensorRT-LLM 1.4.0rc1 + the Hyperdrive source patches (Python / Triton / CuTe-DSL only).
# Every patch is env-gated: with no NHD_* variables set, this image behaves exactly like the stock one.
#   docker build -t nemotron-hyperdrive:1.4.0rc1 .                                     # Spark, B200 Oct, B300 Oct
#   docker build --build-arg NHD_STACK=v3 -t nemotron-hyperdrive:1.4.0rc1-v3 .         # B200 v3, B300 v2 / v3
# NHD_STACK=v3 adds, on top of patches 01-07, the November kernels in the order the cloud job loop applied them
# and FlashInfer main e74a167's CAKE Mamba-2 SSD sources. The v2/v3 numbers were measured with the same patches and
# configs on rented cloud GPUs, applied by our job scripts rather than this image: see docs/RECIPES.md.
FROM nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc1@sha256:3bedbe0321e7eb5430f3177d88894a3cf2dc2f6a087d0e85c13fc882667c75c0
ARG NHD_STACK=base
COPY patches/hyperdrive /opt/nhd/patches
COPY scripts/apply_patches.sh /opt/nhd/apply_patches.sh
RUN bash /opt/nhd/apply_patches.sh /opt/nhd/patches
# F5's fused norm+quant kernel is imported from nhd_mk/ (NHD_MK_DIR = the directory that contains nhd_mk).
ENV NHD_MK_DIR=/opt/nhd/patches
# v3 stack, numbered after 07 so they stack on the base patches.
COPY patches/v3/cake-ssd.diff /opt/nhd/v3/11-cake-ssd.diff
COPY patches/v3/conv-triton.diff /opt/nhd/v3/12-conv-triton.diff
COPY patches/v3/mamba-norm-fp8.diff /opt/nhd/v3/13-mamba-norm-fp8.diff
COPY patches/v3/guided-skip.diff /opt/nhd/v3/14-guided-skip.diff
COPY patches/v3/weight-overlay.diff /opt/nhd/v3/15-weight-overlay.diff
COPY scripts/fetch_flashinfer_cake.sh /opt/nhd/fetch_flashinfer_cake.sh
RUN if [ "$NHD_STACK" = v3 ]; then \
      bash /opt/nhd/apply_patches.sh /opt/nhd/v3 && bash /opt/nhd/fetch_flashinfer_cake.sh /opt/nhd/fimain-e74a167; \
    elif [ "$NHD_STACK" != base ]; then echo "NHD_STACK must be base or v3, got $NHD_STACK" >&2; exit 1; fi \
    && echo "$NHD_STACK" > /opt/nhd/STACK

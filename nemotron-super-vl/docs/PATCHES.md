# Patches

All diffs are against the files shipped in `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc1`
(digest `sha256:3bedbe03…c75c0`, commit `0372799d`) under `/usr/local/lib/python3.12/dist-packages/`. Applied to stock rc1 they reproduce the
files the Spark finals ran with byte for byte. Apply in order with
`scripts/apply_patches.sh` (dry-runs everything first), or build the image with the `Dockerfile`. Python, Triton and
CuTe-DSL source only, no binaries. **Every feature is env-gated: with the env unset, each patched file behaves as stock rc1.**

## Stacked (patches/hyperdrive)

| Diff | Feature | Enable | Measured |
|---|---|---|---|
| 01 `speculative/dflash.py` + 02 `speculative/utils.py` | **Smart draft:** per-step adaptive verify length from drafter confidence, inside the CUDA graph. Rows past the verify length reuse the last verified row's router logits and their drafts are force-rejected; exact-match acceptance, so outputs are unchanged | `NHD_ADAPTK=1 NHD_ADAPTK_THETA=0.25` | Spark: model tok/s 49.7 → 55.9 (+12.5%) single user; 4K code decode +11-22% |
| 01 + 02 | **Drafter prefix-cache context reuse:** on a KV prefix hit, resume or copy a kept drafter-arena slot holding the same tokens (only the 8192+64-column SWA window) | `NHD_DFLASH_PREFIX_CTX=1` | Spark: cached session +19% tok/s; cached-turn tool acceptance 0.42 → 0.85 |
| 03 `models/modeling_nemotron_h.py` | **F5:** fused add + RMSNorm + FP8 quant for the Mamba in_proj input (CuTe-DSL, derived from FlashInfer); also carries the adaptk routing hook used by 01 | `NHD_F5=1`, `NHD_MK_DIR=<dir containing nhd_mk>` | Bit-exact (96/96); ~1% long-prompt TTFT |
| 04 `modules/mamba/ssd_combined.py` + 2 new files | **Fused Mamba-2 SSD prefill** (Triton): one-pass scan for single-sequence chunks, fused chunk_state + state_passing otherwise | `NHD_SSD_FUSED_SCAN=1 NHD_SSD_FUSED_STATE=1` | Kernel bit-identical to rc1; Mamba layer -20.9% per 8K chunk; Spark TTFT -3.2% at 32K, -5.9% at 100K |
| 05 `models/modeling_nemotron_h_multimodal.py` | **Vision-encoder attention backend** (3 lines): moves only the vision tower off FlashInfer/cuDNN, whose plan build hangs on new image shapes under CUDA graphs | `NHD_VISION_ATTN_BACKEND=TRTLLM` | Image test 3/8 + freeze → 8/8; 7 image sizes |
| 06 `pyexecutor/model_engine.py`, `engine/runners/decoder/runner.py`, multimodal | **Text-only breakable prefill CUDA graphs** (prompt chunks up to 3K tokens, text-only batches) | `NHD_PREFILL_GRAPH_TEXT_ONLY=2` + `prefill_cuda_graph_backend: breakable` | Short-turn first token 0.26 / 0.35 / 0.48 s → 0.09 / 0.16 / 0.16 s |
| 07 `serve/tool_parser/qwen3_coder_parser.py` | **Streaming tool-call JSON fix:** empty-argument calls, stray `}`, missing close tag, calls merging within one chunk | always on when applied | 1,283/1,283 CPU checks; 0 errors in 39 GPU agent tool calls (was 6) |

06's multimodal hunks were made on top of 05 (apply in numeric order); they also apply to the stock file, which is what the B200 runs did. 01 also contains earlier experiments (suffix-automaton
`sa_config` on DFlash, ctx-carry); they stay off unless their own config or env is set.

**Which platform uses what**

- Spark: 01-07 with `configs/spark/hyperdrive-spark.env`.
- B300 (final): 03 (F5) + 04, 06 and 07 with `configs/b300/hyperdrive-b300.env`.
- B200: the B200 runs applied 04, 06 and 07 with `configs/b200/hyperdrive-b200.env`. 01-03 are
  for the LiLiCorr drafter path, which B200 doesn't use (MTP instead). The B200 patch set did not include 05, so
  `NHD_VISION_ATTN_BACKEND` was a no-op there; applying 01-07 everywhere is harmless because each feature is env-gated.

## v3 stack (patches/v3, Dockerfile `NHD_STACK=v3`)

| Diff | Feature | Enable |
|---|---|---|
| `cake-ssd.diff` | CAKE Mamba-2 SSD prefill on FlashInfer main e74a167's kernel (cu_seqlens form); D added by a Triton pass on calls with initial states | `NHD_SSD_CAKE=1 NHD_CAKE_FORM=cu NHD_CAKE_D_MODE=split NHD_CAKE_FI_DIR=...` |
| `conv-triton.diff` | Triton varlen channel-last causal conv1d for the Mamba-2 prefill | `NHD_CONV_TRITON=1 NHD_CONV_TILE=...` |
| `mamba-norm-fp8.diff` | Mamba-2 gated RMSNorm writes FP8 straight into the FP8 out_proj | `NHD_MAMBA_NORM_FP8=1` |
| `guided-skip.diff` | Skips the no-op guided-decoding host functions and GPU drains in steps with no structured-output request (lossless) | `NHD_GUIDED_SKIP=1` |
| `weight-overlay.diff` | Loads a safetensors overlay after the checkpoint (the FP4 Mamba projections from `tools/m4_convert.py`) | `NHD_WEIGHT_OVERLAY=<file>` |

What each step bought on B300 (160 users, unique prompts) is in [RECIPES.md](../../RECIPES.md) and the README.

# Bugs found

Six bugs: two in TensorRT-LLM, one in the model card recipe, one likely in FlashInfer, and two in the serving path. All
are worked around in this repo. The TensorRT-LLM and model-card ones are reported upstream, and the workarounds go away as
releases include the fixes.

| # | Bug | Impact | What we did | Status |
|---|---|---|---|---|
| 1 | **[TensorRT-LLM]** Vision encoder's cuDNN attention never finishes planning a new image shape while the LLM runs under CUDA graphs | Any new image size freezes the whole server, nothing logged | Patch 05 moves only the vision encoder to TensorRT-LLM's own attention kernel | Fixed: 7 sizes, image test 8/8, recall 9/9 |
| 2 | **[TensorRT-LLM]** CUTEDSL NVFP4 MoE weight-scale conversion underflows into E4M3 subnormals (SM121) | 8.1-9.5% error per MoE layer on decode and verify | `moe_config.backend: CUTLASS` on Spark; repro in `repro/` | Worked around here; reported upstream. CUTEDSL is accurate on B200 |
| 3 | Follow-up turns can't reuse the cached prefix: the chat template forks it one token before the prompt end, where the hybrid model has no Mamba state snapshot | Long follow-ups re-read the whole prompt | Snapshots at 1 and 576 tokens before each prompt end (config) | Fixed |
| 4 | Streaming `qwen3_coder` tool-call parser emits broken JSON (empty-argument calls, stray `}`, missing close tag, merged calls) | Agent tool calls fail to parse | Patch 07 | Fixed: 1,283/1,283 CPU checks; 39 GPU tool calls, 0 errors (was 6) |
| 5 | [FlashInfer, likely] b12x static decode kernel zero-pads every expert layer (2688 → 2816) at first CUDA-graph warmup: 61.9 GiB of copies plus 18.4 GiB scratch | Out of memory at warmup; hung the Spark twice | A shared-scratch patch (not included here) + `FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=0` | Fixed, not adopted (speed within noise of CUTLASS) |
| 6 | **[model card]** The card recipe sets `attention_backend: TRTLLM` for the LiLiCorr (DFlash) drafter; rc1 refuses it on Spark (SM121 unsupported) and on B200 ("does not support non-causal sliding-window attention") | The card recipe won't start on Spark or datacenter Blackwell | Drafter `attention_backend: VANILLA` (the suggested FA4 is Hopper-only) | Worked around here; reported upstream. We also turn CUDA graphs on |

## CUTEDSL NVFP4 MoE scale underflow (SM121)

`NVFP4CuteDslB12xFusedMoEMethod.transform_weights` in `tensorrt_llm/_torch/moe/fused_moe/quantization.py` (around line 3873 in
rc1) multiplies the E4M3 block scales by `weight_scale_2` (about 2e-5) and recasts the product to E4M3. The products (about
0.004) land in E4M3's subnormal range and lose precision. On layer 1, 0.8% of FC1 block scales become exactly 0 and the mean
scale error is 14.1%; layers 45 and 87 show 8.1-8.4%. CUTEDSL serves batches under 64 tokens, which is every decode and LiLiCorr
verify step.

| Path | Relative error per layer |
|---|---|
| FlashInfer 0.7.x static kernel, ModelOpt's original scales | 0.2-0.3% |
| TRT-LLM 1.4.0rc1 CUTEDSL (TRT scale conversion + FlashInfer 0.6.18 b12x), real weights | **8.1-9.5%** |
| Same, synthetic weights | 20-24% |

Environment: rc1 (commit `0372799d`), FlashInfer 0.6.18 b12x kernels, DGX Spark GB10 (SM121), checkpoint rev `59b87216`.
Repro: `repro/cutedsl_nvfp4_moe_scale_repro.py` (synthetic experts at the routed-expert shape: hidden 1024, intermediate 2688,
512 experts, top-22, relu2, W4A4 NVFP4). End-to-end impact on our small suites is within noise, so we don't claim a measured
quality drop.

## Vision encoder hang under CUDA graphs

A live stack dump shows the vision encoder's attention stuck in cuDNN's execution-plan build for a new input shape; the build
never finishes while the LLM runs under CUDA graphs. Repro: fresh server, CUDA graphs on; 768x320 and 640x480 images answer in
1-2 s, a 720x260 image never returns and wedges the server. All work with graphs off. Patch 05 (`NHD_VISION_ATTN_BACKEND=TRTLLM`)
switches only the vision tower; the LLM's attention is unchanged. A shape-agnostic default, or pre-planning vision shapes at
startup, would avoid it upstream.

## LiLiCorr drafter drafts nearly blind after a prefix-cache hit (SM121)

On a cached turn the target model gets the cached prefix but the drafter's context buffer doesn't (`reuse_addressable=False`),
so acceptance roughly halves (tool turn 81% → 42%, 62 → 37 tok/s). Patch 01's `NHD_DFLASH_PREFIX_CTX=1` keeps the drafter's
context slot and resumes it when the next turn reuses exactly the same prefix tokens: no recompute, no extra memory, outputs
unchanged. Measured +19% on cached agent sessions; cached-turn tool acceptance 0.42 → 0.85.

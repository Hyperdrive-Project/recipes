# Nemotron 3.5 Super VL

NVIDIA Nemotron 3.5 Super VL (120B-A12B, NVFP4) on one B300, one B200 or one DGX Spark, with TensorRT-LLM, SGLang and
vLLM. Results, dates and status for every recipe: [RECIPES.md](../RECIPES.md).

## B300: what v2 and v3 add

Step by step, with unique prompts at 160 users (burst / sustained): October config
5,658 / 5,500; prefix-cache block reuse off 6,092 / 5,996 (fits benchmarks with no shared prefixes; keep it on for chat);
lossless rejection sampling for MTP 6,120 / 6,138; CAKE Mamba-2 SSD prefill (FlashInfer) and a Triton varlen conv1d
6,396 / 6,359; FP8 Mamba norm 6,461 / 6,353; 64K prefill chunks 6,561 / 6,501; skipping TensorRT-LLM's no-op
guided-decoding drains 6,752 / 6,723 (**v2**); the 80 Mamba-2 projections re-quantized from FP8 to NVFP4 7,264 / 7,053 (**v3**).
Measured 2026-11-02 PST on rented B300s; v3 against the October config was measured back to back in one session.

## DGX Spark vs the stock vLLM recipe

Same prompts, seeds and sampling; AIPerf total output tok/s, Hyperdrive TensorRT-LLM vs the published recipe:

| Input/output | 1 user | 2 users | 4 users | 8 users |
|---|---|---|---|---|
| 1K/1K | **39.0** vs 35.3 (+10%) † | **53.7** vs 36.9 (+46%) † | **77.3** vs 36.4 (+112%) | **77.8** vs 36.6 (+113%) |
| 4K/1K | **38.3** vs 36.1 (+6%) † | **54.1** vs 37.2 (+45%) † | **73.5** vs 35.1 (+109%) | **73.1** vs 35.0 (+109%) |
| 32K/1K | **31.1** vs 29.0 (+7%) | **60.1** vs 30.7 (+96%) † | **53.6** vs 29.4 (+82%) | – |
| 120K/1K | **15.8** vs 14.0 (+13%) | – | – | – |

First token at 1 user, no cache: 10% faster at 32K (10.1 vs 11.2 s) and 16% faster at 120K (42.5 vs 50.6 s). The recipe serves
one request at a time as published, so at 2-4 users its first token queues for 27-116 s; Hyperdrive answers in 0.3-2 s at 1K/4K.
Accuracy, reasoning on: GSM8K 98.8% vs 98.8%, HumanEval 98.8% vs 97.6%. Caveats: vLLM uses less GPU energy per token at 1 user
(1.15 vs 1.54 J at 4K); we use 21-65% less from 2 users up. † = our prefix cache helped that cell.

## Quick start (TensorRT-LLM)

```bash
cd nemotron-super-vl
scripts/download_models.sh ./models                                         # pinned model + drafter revisions
docker build --build-arg NHD_STACK=v3 -t nemotron-hyperdrive:1.4.0rc1-v3 .  # stock NGC rc1 + env-gated patches
scripts/make_fp4_mamba.sh                                                   # v3 only: FP4 Mamba weights, once (GPU, ~1 min)

IMAGE=nemotron-hyperdrive:1.4.0rc1-v3 MODEL=super-vl-nvfp4-m4 \
  scripts/serve.sh configs/b300/hyperdrive-b300-v3.yaml configs/b300/hyperdrive-b300-v3.env    # or configs/b200/...-v3

scripts/wait_ready.sh && scripts/aiperf_sweep.sh 8192 1024 "1 8 32 64 128 160"     # unique prompts
python3 ../correctness/eval.py --tasks math,code,tools --reasoning on              # the quality gate
```

SGLang: [`configs/sglang/serve-b200.sh`](configs/sglang/serve-b200.sh) ([notes](docs/SGLANG.md)).
vLLM: [`configs/vllm/serve-hyperdrive.sh`](configs/vllm/serve-hyperdrive.sh) ([notes](docs/VLLM.md)).

The v2/v3 numbers were measured with these patches and configs applied by our cloud job scripts. The Docker image
packages the same files, but the numbers haven't been re-measured through it yet. Every patch is env-gated: with
no `NHD_*` variables set, the image behaves like stock TensorRT-LLM 1.4.0rc1.

## Upstream

Along the way we found six issues, each with a repro and the workaround this repo uses: [docs/BUGS.md](docs/BUGS.md).
The TensorRT-LLM and model-card ones are reported upstream, and we expect the workarounds here to go away as releases pick them up. Patch
details: [docs/PATCHES.md](docs/PATCHES.md).

## Layout

```
Dockerfile   stock NGC TensorRT-LLM 1.4.0rc1 + env-gated patches (NHD_STACK=v3 adds the November kernels)
configs/     b200/, b300/, spark/ TensorRT-LLM yaml + env; sglang/, vllm/ launch scripts; nvidia-card/ baselines
patches/     hyperdrive/ 01-07 + nhd_mk/ kernels; v3/ the November kernels
scripts/     build helpers, serve/stop/wait, AIPerf sweep (unique prompts), model download, FP4 Mamba weights
tools/       m4_convert.py (FP8 -> NVFP4 Mamba projections for v3)
repro/       standalone CUTEDSL NVFP4 MoE scale-underflow repro
docs/        SGLANG, VLLM, BUGS, PATCHES
```

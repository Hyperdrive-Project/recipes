# Recipes

Every Hyperdrive recipe in one place: what to run, what it was measured at, and how far the files here have been checked.
All throughput is output tok/s per GPU at 8,192 tokens in / 1,024 out (AIPerf 0.13, ignore_eos), unless the row says
otherwise. Quality gates run with reasoning on. Times are PST.

**Prompt protocol.** "Unique" means every measured prompt is new (`--num-dataset-entries N+C`). Runs before 2026-11-02
replayed the warm-up prompts in the measured phase, so **about half of the measured prompts were prefix-cache hits**.
Those numbers are labeled *cache-assisted*. Don't compare them with unique-prompt numbers.

**Status column.** *Measured as is* = these exact files produced the number. *Same files, new launcher* = the measured
runs used the same config, env and patches, applied by our cloud job scripts; the Docker path below packages them,
but the number has not yet been re-measured end to end through it.

## Nemotron 3.5 Super VL (NVFP4, all experts, lossless speculation)

| Recipe | Hardware | Result | Quality (math / code / tools) | Measured | Status |
|---|---|---|---|---|---|
| **TRT-LLM v3** | 1x B300 SXM6, 160 users | **7,264 burst / 7,053 sustained**, unique | 49/50, 29/30, 11/11 | 2026-11-02 1:27 PM | Same files, new launcher |
| TRT-LLM v2 (no model change) | 1x B300, 160 users | 6,752 / 6,723, unique | 48/50, 29/30, 11/11 | 2026-11-02 | Same files, new launcher |
| TRT-LLM October | 1x B300, 160 users | 5,658 / 5,500 unique (7,851 cache-assisted) | 49/50, 29/30, 11/11 | 2026-10-23 / 11-02 | Measured as is |
| **TRT-LLM v3** | 1x B200, 128 users | **6,280 / 6,307**, unique | not gated alone (see note) | 2026-11-03 ~1 AM | Same files, new launcher |
| TRT-LLM v2 | 1x B200, 128 users | 5,660 / 5,679, unique | 48/50, 29/30, 11/11 (gated before the lossless 64K-chunk and guided-skip steps) | 2026-11-02 | no config file yet |
| TRT-LLM October | 1x B200 | 4,612 @64u, 4,218 @128u, cache-assisted | 48/50, 29/30, 11/11 | 2026-10-23 | Measured as is |
| **SGLang** (stock image) | 1x B200, 128 / 160 users | **3,705 / 3,622**, unique | 48/50, 29/30, 11/11 | 2026-11-07 | Measured as is (flags) |
| vLLM 0.32.0 (stock image) | 1x B200, 128 users | 5,567, cache-assisted | 49/50, 29/30, 11/11 | 2026-10-30 | Measured as is |
| vLLM 0.32.0 (stock image) | 1x B300, 160 users | 6,470, cache-assisted | 49/50, 29/30, 11/11 | 2026-10-30 | Measured as is |
| TRT-LLM Spark | 1x DGX Spark, 1-8 users | 1.06-2.13x the stock vLLM recipe (table in [README](nemotron-super-vl/README.md)) | GSM8K 98.8%, HumanEval 98.8% (lm-eval) | 2026-10 | Measured as is |

Notes:

- **v2** = the October config plus block reuse off, lossless rejection sampling for MTP, 64K prefill chunks, CAKE Mamba-2
  SSD prefill (FlashInfer main, D added in a separate pass), Triton varlen conv1d, FP8 Mamba norm and the guided-decoding
  skip. **v3** = v2 plus the 80 Mamba-2 projections re-quantized from FP8 to NVFP4: a change to the weights, kept because
  the quality gate held. If you want no weight changes at all, use v2.
- **Block reuse is off in v2/v3** because the benchmark has no shared prefixes. For chat with shared system prompts or
  multi-turn sessions, set `enable_block_reuse: true` (and measure; the Mamba snapshot offsets then force extra prefill chunks).
- **B200 v3 quality:** v3 itself was not gated on B200. The same weights and kernels passed on B300, and on B200 v3 plus
  FP8 MoE latent projections passed (49/50, 29/30, 11/11).
- **Stock model-card recipe, B200** (baseline, [`configs/nvidia-card/model-card-b200-baseline.yaml`](nemotron-super-vl/configs/nvidia-card/model-card-b200-baseline.yaml)):
  278 / 911 / 1,130 / 1,256 tok/s at 1 / 8 / 16 / 32 users, cache-assisted (2026-10-23); it can't run above 32 users. It needed the
  drafter backend fix to start, and we turned CUDA graphs on for it. There is no card-recipe run on B300.
- **vLLM** has not been rerun with unique prompts, so the TRT-LLM vs vLLM gap is approximate.
- **Not in a recipe yet:** the MoE FC1 kernel changes (+2.9% burst on B200, unpaired run, Nov 6) and FancyCake's fused Mamba output kernel (−2.5% in the model on B200, not kept).

### TensorRT-LLM: build and run

```bash
cd nemotron-super-vl
scripts/download_models.sh ./models                                         # pinned model + drafter revisions
docker build -t nemotron-hyperdrive:1.4.0rc1 .                              # October configs, Spark
docker build --build-arg NHD_STACK=v3 -t nemotron-hyperdrive:1.4.0rc1-v3 .  # v2 and v3 (adds the Nov kernels)
scripts/make_fp4_mamba.sh                                                   # v3 only: FP4 Mamba weights, once, GPU, ~1 min

# B300 v3 / B200 v3
IMAGE=nemotron-hyperdrive:1.4.0rc1-v3 MODEL=super-vl-nvfp4-m4 \
  scripts/serve.sh configs/b300/hyperdrive-b300-v3.yaml configs/b300/hyperdrive-b300-v3.env
IMAGE=nemotron-hyperdrive:1.4.0rc1-v3 MODEL=super-vl-nvfp4-m4 \
  scripts/serve.sh configs/b200/hyperdrive-b200-v3.yaml configs/b200/hyperdrive-b200-v3.env
# B300 v2 (original weights)
IMAGE=nemotron-hyperdrive:1.4.0rc1-v3 scripts/serve.sh configs/b300/hyperdrive-b300-v2.yaml configs/b300/hyperdrive-b300-v2.env
# October configs
scripts/serve.sh configs/b300/hyperdrive-b300-final.yaml configs/b300/hyperdrive-b300.env
scripts/serve.sh configs/b200/hyperdrive-b200.yaml configs/b200/hyperdrive-b200.env
scripts/serve.sh configs/spark/hyperdrive-spark.yaml configs/spark/hyperdrive-spark.env

scripts/wait_ready.sh && scripts/aiperf_sweep.sh 8192 1024 "1 8 16 32 64 128 160"   # unique prompts by default
python3 ../correctness/eval.py --tasks math,code,tools --reasoning on                 # the quality gate
```

The first v2/v3 start compiles the CAKE kernels into `./cache` (mounted at `/cache`); later starts reuse them. Every
patch is env-gated, so the v3 image with the October env behaves like the base image.

### SGLang and vLLM

- SGLang, B200: [`configs/sglang/serve-b200.sh`](nemotron-super-vl/configs/sglang/serve-b200.sh); details [SGLANG.md](nemotron-super-vl/docs/SGLANG.md).
- vLLM, B200/B300: [`configs/vllm/serve-hyperdrive.sh`](nemotron-super-vl/configs/vllm/serve-hyperdrive.sh); details [VLLM.md](nemotron-super-vl/docs/VLLM.md).

## Nemotron 3.5 Bolt on Flare (DGX Spark, real-time voice)

Cached voice turn, reasoning off, one DGX Spark; baseline = the stock Flare recipe for Bolt, same model and drafter.

| Metric | Stock recipe | Hyperdrive | Measured | Status |
|---|---|---|---|---|
| First token | 0.219 s | **0.084 s** | 2026-10-24 to 10-26 PDT | Measured as is (patch series `flare-bolt-robot`) |
| First spoken phrase | 0.59 s | **~0.30 s** | same | same |
| Decode (held-out voice turns) | 37.9 tok/s | **60.8 tok/s** (verify width 4) | 2026-10-25 | same |
| AIPerf 4K / 1K, 1 user | 71.6 | 73.6 | 2026-10-24 | same |

Quality, reasoning on: math 48/50, code 27-29/30, tools 11/11; lm-eval GSM8K 0.956, HumanEval 0.970, needle 15/15 to
120K, the same as the stock recipe. The rebased `flare-bolt-robot-next` series (history drafts, phrase seed) is built
but not measured. Run: [`bolt-flare-voice/`](bolt-flare-voice/README.md).

## Qwen3.8 Flash-Next FP4-Local (DGX Spark)

| Recipe | Result | Quality | Measured | Status |
|---|---|---|---|---|
| **eugr vLLM b12x + Hyperdrive overlays + FancyCake 48** (the default) | **SparkBench PBM 49.8 / 25.7 / 16.2 tok/s** at 4K / 50K / 100K, mean of 5 repeats | GSM8K 49/50, HumanEval 28/30, tools 11/11, needle 9/9 (reasoning on, without FancyCake 48); FancyCake 48 vs stock: GSM8K 98/100 vs 99/100, HumanEval 27/30 and tools 17/20 both equal | 2026-11-10 5:45-6:07 PM | Measured as is |
| Same without FancyCake 48 (`FANCYCAKE=0`, the 2026-10-30 SparkBench submission) | 4K 47.2 single run (43.3 mean of 3 runs without warm-up), 50K 24.6, 100K 15.5 (3-run means); not re-run with the 5-repeat method | as above | 2026-10-30 | Measured as is |

Run: [`qwen-flash-next/serve.sh`](qwen-flash-next/serve.sh) (FancyCake 48 on by default, `FANCYCAKE=0` for the
Oct 30 submitted config). FancyCake 48 alone: decode 53.95 vs 56.88 ms per step (5.2% faster, single user), and it trades
KV cache for that speed; details in
[`qwen-flash-next/fancycake-48/`](qwen-flash-next/fancycake-48/README.md).

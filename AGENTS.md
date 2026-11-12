# Guide for coding agents

This file is for coding agents asked to run, adapt or benchmark a Hyperdrive
recipe. Humans can read it too; the per-recipe READMEs have the full detail.

## Fastest path

```bash
./hyperdrive detect               # which GPU is here, which recipes fit (add --json)
./hyperdrive show <id>            # exact setup / serve / stop / check commands, endpoint, served model name
./hyperdrive run <id>             # prints the plan; add --yes only after the user agrees to the download and GPU use
./hyperdrive check                # quality gate against the running server
```

The same data is in [recipes.json](recipes.json) if you'd rather read it directly. Every server listens on
`http://127.0.0.1:8000/v1` (OpenAI-compatible).

## What this repo is

Serving recipes: configs, patches, launch scripts and a quality suite. There is no package to install and no build step
at the root. Each recipe is a folder with its own README:

| Folder | Model | Hardware | Engine | Entry point |
|---|---|---|---|---|
| `nemotron-super-vl/` | Nemotron 3.5 Super VL (NVFP4) | B300, B200, DGX Spark | TensorRT-LLM 1.4.0rc1 (patched image), SGLang, vLLM | `scripts/serve.sh <config.yaml> <env>`, `configs/sglang/serve-b200.sh`, `configs/vllm/serve-hyperdrive.sh` |
| `bolt-flare-voice/` | Nemotron 3.5 Bolt | DGX Spark | NVIDIA Flare (patched) | `serve/serve-bolt.sh` |
| `qwen-flash-next/` | Qwen3.8 Flash-Next FP4-Local | DGX Spark | vLLM (eugr b12x image) | `serve.sh` |
| `correctness/` | any | any | any OpenAI-compatible server | `eval.py` |

## Ground rules

1. **Ask before spending money or taking a GPU.** Recipes download 70-125 GB of weights and hold a whole GPU. Check that
   the GPU is free (`nvidia-smi`) and that the user wants the download.
2. **Never change the model to get speed.** The recipes keep the published checkpoint, all experts and lossless
   speculation. The one weight change (`nemotron-super-vl` v3: FP4 Mamba projections) is opt-in and documented; v2 is
   the no-weight-change option.
3. **Measure quality after any change:** `python3 correctness/eval.py --label <name> --tasks math,code,tools --reasoning on`.
   Compare with the stock baseline on the same items. A speedup that loses quality is not a result.
4. **Measure throughput with unique prompts.** `nemotron-super-vl/scripts/aiperf_sweep.sh` passes
   `--num-dataset-entries N+C`. Without it, AIPerf replays warm-up prompts and prefix caching inflates the numbers ~2x.
5. **Warm up before timing.** The first request after a server load is slow (graph capture, kernel compile, the CAKE
   kernels compile into `./cache` on first start). Discard it.
6. **Pin what the README pins:** image digests, model revisions, upstream commits. Newer versions may not apply the
   patches cleanly or may change the numbers.

## Recipe-specific notes

- **TensorRT-LLM (B300/B200):** build the image with `docker build --build-arg NHD_STACK=v3 -t nemotron-hyperdrive:1.4.0rc1-v3 .`
  from `nemotron-super-vl/`. Every patch is gated by an `NHD_*` env var from the config's `.env` file; with none set the
  image is stock TensorRT-LLM. v3 needs `scripts/make_fp4_mamba.sh` once (GPU, ~1 min).
- **Block reuse is off** in the B200/B300 v2/v3 configs because the benchmark has no shared prefixes. For chat with long
  shared system prompts, set `enable_block_reuse: true` and re-measure.
- **DGX Spark has 128 GB of unified memory** shared by the CPU and GPU. Keep other GPU jobs off it while a recipe loads,
  drop the page cache before loading large models, and leave headroom (the Qwen recipe uses `UTIL=0.8`).
- **SGLang:** the two Mamba flags in `serve-b200.sh` are what lift the concurrency cap from 27 to 98; keep them.
- **Bolt / Flare:** build Flare at the pinned upstream commit, `git am` the patch series, export the speech package,
  then serve. Use `"chat_template_kwargs": {"enable_thinking": false}` for voice turns.

## Reporting results

Report: model and revision, GPU, engine and image digest, config file, input/output lengths, users, whether prompts were
unique, the mean of at least 3 runs with a warm-up, and the quality scores next to the baseline's. Use PST/PDT dates as
the repo does. Open an issue with the results if they differ from [RECIPES.md](RECIPES.md).

# Nemotron 3.5 Super VL on vLLM (B200, B300)

Status: 2026-10-30 2:45 AM PDT. Final for B200 and B300. Every screen below is done; the launch command did not change.

**Launch command:** [`configs/vllm/serve-hyperdrive.sh`](../configs/vllm/serve-hyperdrive.sh) (`GPU=b200` or `GPU=b300`).
Stock `vllm/vllm-openai:v0.32.0`, stock checkpoint, no patches, no custom kernels.
Measured on rented single-GPU cloud machines.

> **Measurement note (2026-11-02).** The October AIPerf runs here used `--request-count 2C --warmup-request-count C`.
> AIPerf 0.13 replays the warm-up prompts in the measured phase unless `--num-dataset-entries` covers both, so about half of the
> measured prompts were repeats that hit the prefix cache. The seed was unique per point, which is why these runs were called
> "cache-free" or "unique prompts"; that label was wrong. The numbers are real for that workload but overstate uncached
> throughput: the October B300 final config (7,851) gives 5,658 burst / 5,500 sustained with every prompt unique, and the
> November v2 config gives 6,752 / 6,723 and v3 (v2 + FP4 Mamba
> projections) **7,264 / 7,053** (B200 with the same changes: 5,660 / 5,679 at 128 users). Every TRT-LLM and vLLM
> AIPerf curve here from before 2026-11-02 used the same protocol, so comparisons among them are like-for-like, but not against
> unique-prompt numbers. Lambda's published figures are their own measurements.

Method: AIPerf 0.13.0, 8,192 in / 1,024 out, unique random seed per point (but ~50% warm-up repeats, see note), ignore_eos, temperature 1.0,
top_p 0.95, one warm-up pass after each server load, single run per point (treat differences of ~2% as noise).
Output tokens only, summed over users on one GPU.

## Baseline: the stock vLLM recipe

NVIDIA publishes a tuned datacenter vLLM recipe in the Dynamo repo
(`ai-dynamo/dynamo` `recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b200-swebenchpro/deploy.yaml`, same args for B300):
MTP 5, 16K batched tokens, FULL_AND_PIECEWISE CUDA graphs, FP8 KV, FlashInfer Mamba, attention+quant fusion.
We ran its worker args on plain `vllm serve` (Dynamo router/KV-event flags dropped) as `nv`.

## B200 (one GPU)

| Users | 1 | 8 | 16 | 32 | 64 | 128 | 160 | 256 |
|---|---|---|---|---|---|---|---|---|
| **vLLM + Hyperdrive settings** | 445 | 1,636 | 2,406 | 3,453 | 4,586 | **5,567** (5,650 rerun) | 5,001 | 4,796 |
| Stock vLLM recipe | 439 | 1,608 | 2,447 | 3,365 | 4,263 | 4,437 | – | – |
| Hyperdrive TRT-LLM (patched) | 348 | 1,415 | 2,292 | 3,361 | 4,612 | 4,218 | – | – |

- vLLM with our settings beats the stock vLLM recipe at every load (+25% at 128) and, in this cache-assisted comparison, our
  October TRT-LLM peak by ~21% (5,567-5,650 vs 4,612). The November TRT-LLM v3 config gives 6,280 at 128 users with unique
  prompts; vLLM has not been measured that way yet. Peak is 128 users; above that throughput falls, so the B200 config caps `max-num-seqs` at 128.
- Quality, reasoning on (`correctness/eval.py`): math 49/50, code 29/30, tools 11/11, identical to the TRT-LLM final.

Screens at 64 / 128 users (B200):

| Change | 64 | 128 | Kept |
|---|---|---|---|
| Stock recipe (MTP 5) | 4,263 | 4,437 | baseline |
| MTP 3, max-num-seqs 128, no stochastic rounding, stream interval 4 | 4,486 | 5,392 | yes |
| + async scheduling | 4,424 | 5,500 | yes (noise-level, harmless) |
| + 32K batched tokens | 4,568 | 5,564 | yes |
| MoE `flashinfer_cutedsl` instead of auto (`flashinfer_trtllm`) | – | 5,510 | no (-2.5%) |
| MoE `flashinfer_cutlass` | – | 3,877 | no (-30%) |

## B300 (one GPU)

| Users | 1 | 8 | 32 | 64 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|
| **vLLM + Hyperdrive settings** (max-num-seqs 256) | 457 | 1,668 | 3,613 | 4,975 | 6,071 | 6,470 | 6,731 | **7,007** |
| Hyperdrive TRT-LLM (patched, batch 160) | 383 | 1,549 | 3,676 | 5,355 | – | **7,851** | – | – |
| Hyperdrive TRT-LLM, unique prompts (Oct / Nov v2 / v3) | – | – | – | – | – | 5,658 / 6,752 / 7,264 | – | – |

- vLLM wins at 1-8 users, ties at 32, and trails at high load: 18% behind at 160 users and still 11% behind TRT's 160-user
  point at 256 users (where each user gets 39 tok/s instead of 75). **On B300, TRT-LLM is faster at high concurrency** in this cache-assisted comparison; vLLM has not been rerun with unique
  prompts, so its honest gap to TRT-LLM's 7,264 (v3) is not measured.
- Quality, reasoning on: math 49/50, code 29/30, tools 11/11.
Screens on B300 (8K/1K, total output tok/s at 160 / 256 users; control = launch command, run on the same pod as the screen):

| Change | 160 | 256 | Kept |
|---|---|---|---|
| Control (launch command, max-num-seqs 256) | 6,310 | 6,803 | – |
| MTP 2 instead of 3 | 6,312 | 6,906 | no (noise) |
| MoE `flashinfer_cutedsl` (CuTe DSL) | 6,071 @128 | 6,979 | no (tie with default) |
| MoE `flashinfer_cutlass` | 4,144 @128 | 4,602 | no (-32%) |
| MoE `cutlass` (vLLM CUTLASS) | 3,705 @128 | – | no |
| Fusion passes `fuse_norm_quant` + `fuse_act_quant` | 6,380 | 6,918 | no, see stack |
| 64K batched tokens | 6,412 | 6,871 | no, see stack |
| 16K batched tokens | 6,175 | 6,800 | no |
| `--mamba-backend triton` | 5,626 | 5,987 | no (-11%) |
| Mamba ReplaySSM (needs `--mamba-cache-mode none`, one-line class flag) | 6,070 | 6,490 | no (-4%) |
| No Mamba prefix cache, no replay (replay's own control) | 5,541 | 5,906 | – |
| **Stack: fusion passes + 64K + launch command** (full curve below) | 6,361 | 6,882 | no |

Stacked run (fusion passes + 64K): 467 / 1,703 / 3,519 / 4,987 / 6,111 / 6,361 / 6,555 / 6,882 at 1 / 8 / 32 / 64 / 128 / 160
/ 192 / 256 users, quality the same (math 49/50, code 29/30, tools 11/11). That matches the launch command's curve within run-to-run
noise (~2%), so the two +1-2% single screens were noise and the launch command stays as it is. Stock vLLM 0.32 on B300 has no
further lever we could find that closes the gap to TRT-LLM; what remains is engine-level (TRT-LLM's MoE GEMMs and our F5 fused
kernel at high batch).

## Moving to TRT-LLM later

The TRT-LLM configs are in [`configs/b200`](../configs/b200) and [`configs/b300`](../configs/b300) (B300 final:
`hyperdrive-b300-final.yaml` + `hyperdrive-b300.env` + patches, see [PATCHES.md](PATCHES.md) and [RECIPES.md](../../RECIPES.md)).
Settings that carry over 1:1 between the two engines: MTP 3, FP8 KV, fp16 Mamba SSM cache without stochastic rounding,
32K prefill chunks, stream interval 4, overlap/async scheduling, prefix caching. Engine-specific: TRT needs `moe_config.backend:
CUTEDSL` (CUTLASS is -28%), CUDA graph batch sizes up to the batch cap, and our F5 kernel (+3.3%). On B300, TRT-LLM's batch
160 config is the throughput leader (7,851 cache-assisted; 7,264 v3 with unique prompts); on B200, vLLM's batch 128 config led the October cache-assisted comparison (5,567-5,650).

TRT-LLM follow-up on B300 (2026-10-30, 2 points at 160 users each;
cache-assisted protocol, so compare rows with each other only):

| TRT-LLM build | tok/s @160 | Note |
|---|---|---|
| rc1 + B300 final config (control, rerun) | 7,620 | our fused Mamba SSD kernel |
| rc1 + FlashInfer Mamba SSD (`TRTLLM_USE_MAMBA_FI_SSD=1`, FlashInfer 0.6.18) | 7,496 | -1.6%, not kept |
| rc1 + FlashInfer Mamba SSD, FlashInfer 0.7.0.post1 | 7,297 | -4%, not kept |
| rc2 (`release:1.4.0rc2`), final config minus the breakable prefill graph | 5,939 | rc2 rejects the breakable prefill CUDA graph for VL models and our patches 06/07 no longer apply |
| rc2, stock (no Hyperdrive env) | 5,768 | |

rc3 is not published on nvcr.io yet (checked 2026-10-30). Stay on **1.4.0rc1** with the B300 final config until a newer release
accepts the text-only prefill graph patch.

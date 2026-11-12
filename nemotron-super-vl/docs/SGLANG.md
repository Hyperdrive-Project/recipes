# SGLang on B200: Super VL NVFP4 (Nov 6-7, 2026)

One B200, image `lmsysorg/sglang:nemotron3-5-super` (sglang dev, torch 2.14.1+cu130, flashinfer 0.7.1), TP1, FP8 KV,
AIPerf 0.13 8K in / 1K out, ignore_eos, **unique prompts per point** (`--num-dataset-entries N+C`; 0 cached prefill
tokens confirmed in the server logs). Recipe: [`configs/sglang/serve-b200.sh`](../configs/sglang/serve-b200.sh).

## Recipe result (Nov 7, 2026)

| Config | 1u | 8u | 32u | 64u | 128u | 160u |
|---|---|---|---|---|---|---|
| Cookbook stock (no MTP) | 229 | 1,013 | 1,945 | 2,234 | 2,347 | |
| Hyperdrive flags (EAGLE MTP 3/1/4, 32K chunks, max-running 128, decode graphs 128, stream interval 4) | | | | 2,334 | 2,459 | |
| **+ BF16 Mamba state + Mamba memory ratio 6 (the recipe)** | **477** | **1,521** | **2,777** | **3,449** | **3,705** | **3,622** |

Output tok/s per GPU. The recipe is +51% over the Hyperdrive flags alone at 128 users and +58% over the cookbook; TTFT at
128 users drops from 32 s to 10 s. Quality with reasoning on: math 48/50, code 29/30, tools 11/11 (vLLM on the same model:
49/50, 29/30, 11/11). For reference, TensorRT-LLM Hyperdrive v3 on B200 gives 6,280 at 128 users with the same protocol.

- **Why it was slow: requests queued.** SGLang sized its Mamba state cache so that `max_running_requests` was capped at
  27 with MTP (52 without) and gave the rest of memory to a 10-12M-token KV pool, so in the "128-user" runs only 27
  requests ever ran at once. Prefill itself ran at 34-60K tokens/s.
- **The fix:** `--mamba-ssm-dtype bfloat16` halves each request's state and `--mamba-full-memory-ratio 6` gives the Mamba
  cache more of the memory, so 98 requests run at once. BF16 state changes the numerics slightly, which is why the
  quality gate above was run before this became the recipe.
- **Tried and not kept (Nov 7, 128 users, on the recipe):** mixed chunks −16%; lazy Mamba buffer (111 running) +0.8%;
  ratio 8 +0.5%; CuTe DSL MoE +9% prefill but the same total; CUTLASS MoE never came up (host RAM). Every change that sped
  up prefill or added running slots left 128 users at ~3,670, so the remaining limit is not GPU kernel speed; a timeline
  profile to find it has not been run yet.
- **Earlier screens (Nov 6, Hyperdrive flags, before the fix):** MoE `flashinfer_trtllm` 2,450, `flashinfer_cutedsl`
  2,357, 64K prefill chunks 2,469, 16K 2,398 at 128 users.
- **Single user is strong:** 2.0 ms ITL, ~500 tok/s per user with MTP.
- **Not measured:** B300, vision.
- **Runner gotchas in this build:** `--cuda-graph-max-bs` is ambiguous (use `-decode`). A cold server load takes
  12-14 min; a warm one takes 2.5 min.

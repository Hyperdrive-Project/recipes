# Qwen Flash-Next Hyperdrive (DGX Spark)

NVIDIA **Qwen3.8-Flash-Next FP4-Local** (`nvidia/Qwen3.8-Flash-Next-FP4-Local` @ `b9ce18a6`, 125B total / ~6B active MoE,
NVIDIA QAD FP4, all 10 routed experts) on **one DGX Spark**, served by eugr's vLLM b12x build
(`eugr/spark-vllm-b12x@sha256:03a95654…`, the image that loads FP4-Local's compressed NVFP4 scales).

```bash
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --revision b9ce18a6887e18c089fdea8509a33e63429a3300
qwen-flash-next/serve.sh                 # FancyCake 48 on (the measured config); FANCYCAKE=0 for the Oct 30 submission
```

## Results (one DGX Spark)

SparkBench PBM (unmodified bench v2 ladder, fills of 4,096 / 50,000 / 100,000 tokens, one session per rung), the prefix
cache reset before each repeat, each repeat with SparkBench's own warm-up. 2026-11-10 5:45-6:07 PM PST, `serve.sh` defaults
(FancyCake 48 on). Output tok/s:

| | 4K | 50K | 100K |
|---|---|---|---|
| **Hyperdrive, mean of 5 repeats** | **49.8** | **25.7** | **16.2** |
| Range over the 5 repeats | 49.7-49.9 | 25.5-25.8 | 15.6-16.6 |
| One repeat without SparkBench's warm-up (diagnostic) | 43.3 | 25.6 | 15.8 |
| TensorFold FP4-Local, as listed in SparkBench | 45.4 | 22.6 | 13.8 |
| INT4 build routing 5 of 10 experts, as listed in SparkBench | 50.2 | 26.6 | 15.4 |

The fastest full-quality (FP4-Local) Flash-Next on DGX Spark in SparkBench PBM: every repeat beats the TensorFold entry at
every fill (mean +9.7% / +13.7% / +17.4%). The reduced INT4 build still leads at 4K and 50K; with half the experts per
token and INT4 weights it should be expected to lose some accuracy, but no accuracy results are published for it. Tool round trips were OK in all
sessions. Draft acceptance 0.59-0.60 across repeats.

Our 2026-10-30 SparkBench submission (without FancyCake 48) listed 47.2 / 24.6 / 15.5; its 4K figure was a
single run while 50K and 100K were 3-run means, so use the table above instead.

Quality (reasoning on): GSM8K 49/50, HumanEval 28/30, tools 11/11, needle to 100K 9/9
([results-quality/q-D8-final.json](results-quality/q-D8-final.json)).

**FancyCake 48** (2026-11-02, single user): decode 53.95 vs 56.88 ms per step (5.2% faster), at the cost of a smaller KV
cache. It is on in the SparkBench numbers above. Details: [fancycake-48/](fancycake-48/README.md).

## What the recipe changes (each measured, kept only if it won)

- **MTP 3 with probabilistic drafting** (`draft_sample_method: probabilistic`, standard rejection sampling): draft acceptance
  41% to 57%. Output distribution unchanged.
- **34K-token MTP draft vocabulary** ([mods/](mods/README.md)): the drafter's LM head projects onto 34,048 common token ids,
  about 0.8 GB less read per verify step. The target model is untouched.
- **Recurrent checkpoint 8 tokens before each prompt end** ([mods/](mods/README.md)): multi-turn agent turns re-prefill ~8
  tokens instead of 4-8K. 50K: 19.6 to 23.7 (single cold run).
- **FancyCake 48** (from FancyCake, a forthcoming Hyperdrive project): expands the compressed NVFP4 expert scales once at load, so no per-call scale decode runs at decode time.

Tried and not kept: MTP 4, 2K/8K/16K prefill chunks, async scheduling, `cudagraph_mode: FULL_AND_PIECEWISE`, 4-bit expert
activations, and BF16 recurrent state (this build's GDN kernel needs FP32 state). A `--max-num-seqs 4` cap raises single-session
4K throughput but queues the 5th request, so the recipe runs uncapped (`--max-num-seqs 8`).

Safety note: loading ~100 GiB resident on the Spark triggered the kernel OOM killer once. Keep resident weights well under
~100 GiB and drop the page cache before loading.

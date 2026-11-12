# FancyCake 48: static NVFP4 expert scales (Qwen Flash-Next on DGX Spark)

Kernel work by **FancyCake**, a forthcoming Hyperdrive project. Only the finished change ships here; the FancyCake tool
is not part of this repo. Overlay of
`vllm/model_executor/layers/quantization/nvfp4_csf.py` in `eugr/spark-vllm-b12x@sha256:03a95654…`.

FP4-Local stores its NVFP4 expert scales compressed (CSF) and the stock path decodes them on every MoE call. With
`HYPERDRIVE_CSF_STATIC=1 HYPERDRIVE_CSF_STATIC_DEFER=1 HYPERDRIVE_CSF_STATIC_FREE=1` this file expands the scales of all
48 MoE layers once, after the model loads, and drops the per-call decode. The expanded scales are bit-identical to the
decoded ones (verify run: 0 bytes differ). Without those variables the file behaves like the stock one.

Measured on one Spark (2026-11-02, single user, MTP 3): decode **53.95 vs 56.88 ms per step (5.2% faster)**; prefill of a
26K prompt 11.3 -> 10.8 s. Cost: the expanded scales take memory from the KV cache (test server 20.5 -> 16.1 GiB; live service
18.31 -> 16.92 GiB, 918K tokens after). Quality vs stock, reasoning on, temperature 0: GSM8K 98/100 vs 99/100 (the one loss hit the
4,096-token cap), HumanEval 27/30 vs 27/30, tool calls 17/20 vs 17/20, needle at 31K and 50K tokens 3/3.
SparkBench PBM with it on (2026-11-10, mean of 5 repeats): 49.8 / 25.7 / 16.2 tok/s at 4K / 50K / 100K ([../README.md](../README.md)).

Source: vLLM (Apache-2.0, SPDX header kept); Hyperdrive changes are the `_hd_*` functions and `HYPERDRIVE_CSF_*` switches.

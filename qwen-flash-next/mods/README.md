# Hyperdrive overlays for Qwen3.8-Flash-Next FP4-Local (eugr b12x)

Host-side file overlays for `eugr/spark-vllm-b12x@sha256:03a95654…` (vLLM `0.1.dev21830+g8244d1c44`), used by
[`../serve.sh`](../serve.sh). Both change speed only: MTP verification is
unchanged (standard rejection sampling), so outputs keep the target model's distribution.

Mounted read-only into:

```
/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py
/usr/local/lib/python3.12/dist-packages/vllm/v1/core/boundary_checkpoint.py
```

## 1. Reduced MTP draft vocabulary (`mtp.py`, `ids_34k.txt`)

`HYPERDRIVE_DRAFT_VOCAB=<file of token ids>` makes the MTP drafter's (already NVFP4) LM head project onto 34,048
token ids instead of 248,320. The rows are gathered from the BF16 head before the runtime NVFP4 quantization, and the
draft logits are scattered back into the full vocabulary with `-inf` elsewhere, so probabilistic drafting and rejection
sampling see a full-vocabulary proposal distribution. The drafter can only propose those tokens; the target model is
untouched. Saves ~0.8 GB of reads per verify step (3 draft passes). `ids_34k.txt` = every token seen in ~165M tokens of
English and code (33.8K) plus all added/special and single-character tokens, padded with the lowest BPE ids to a multiple
of 128. Text in other languages still decodes correctly, with lower draft acceptance.

## 2. Near-end recurrent checkpoint (`boundary_checkpoint.py`)

`HYPERDRIVE_TAIL_OFFSET=8` stores the recurrent (Gated DeltaNet / Mamba-style) state 8 tokens before each prompt's end.
Multi-turn chat re-renders the previous generation prompt (`<think>\n`) a token differently, so the next turn diverges
1-2 tokens before the old prompt end and never matches the prompt-end checkpoint. Stock `request_boundaries` then
falls back to a checkpoint one to two prefill chunks back, so every agent turn re-prefilled 4-8K tokens. With the
offset, a turn re-prefills ~8 tokens plus its new text. PBM effect on this model: 50K 19.6 -> 23.7 (cold, 47K draft vocab).

Source: Hyperdrive (vLLM files are Apache-2.0, see SPDX headers; changes marked `hyperdrive`).

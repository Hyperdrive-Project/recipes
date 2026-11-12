# Quality suite

Every Hyperdrive recipe has to pass the same quality gate before its speed counts. This suite runs against any
OpenAI-compatible server (TensorRT-LLM, vLLM, SGLang, Flare) and writes one JSON per run with per-item details.

```bash
python3 correctness/eval.py --label my-run --tasks math,code,tools --reasoning on             # the gate used in RECIPES.md
python3 correctness/eval.py --label my-run --tasks math,code,tools,image,needle --base-url http://127.0.0.1:8000
python3 correctness/eval.py --dry-run                                                         # full pipeline against a mock server
```

| Task | What it checks | Pass rule |
|---|---|---|
| `math` | 50 GSM8K test problems ([`data/gsm8k_test_50.jsonl`](data/gsm8k_test_50.jsonl)) | exact numeric answer |
| `code` | 30 HumanEval problems ([`data/humaneval_30.jsonl`](data/humaneval_30.jsonl)) | generated code passes the tests in a sandbox (time and memory limits, no network) |
| `tools` | 11 hand-written function-calling cases | right tool name and JSON arguments |
| `image` | 8 procedurally drawn images (shapes, counting, digits, a word, a bar chart, position) | exact answer |
| `needle` | needle retrieval at 4K-32K prompt tokens, three depths | needle found |
| `determinism` | 8 fixed greedy prompts diffed token by token against a saved reference | identical tokens |

The datasets are pinned: exact revisions, indices and sha256s are in [`data/MANIFEST.json`](data/MANIFEST.json), and
[`data/regen.py`](data/regen.py) rebuilds them from the source datasets.

**How we gate.** Reasoning on, the same items for the stock baseline and the recipe. Each recipe's README and
[`../RECIPES.md`](../RECIPES.md) list its scores next to the baseline's.

Other checks:

- [`context_ladder.py`](context_ladder.py): needle retrieval one request at a time at growing prompt lengths (up to 120K).
- [`image_check.py`](image_check.py), [`mixed_image_text_check.py`](mixed_image_text_check.py): image understanding,
  and mixed image + text batches (catches vision regressions from text-only optimizations).
- [`lmeval_tasks/`](lmeval_tasks/): the lm-eval HumanEval task (all 164, reasoning on) used for the DGX Spark comparison.

Requirements: Python 3.10+, standard library only for most tasks; `transformers` for token-exact needle prompts and
the determinism check.

# Benchmark helpers

Shared code for the quality suite and the single-user and agent benchmarks. Standard library only.

| File | What it does |
|---|---|
| [`bench.py`](bench.py) | workload runner for any OpenAI-compatible server: preset or custom input/output lengths, concurrency sweeps, warm-ups excluded, streamed TTFT / decode split, one JSON per run |
| [`agent_suite.py`](agent_suite.py) | realistic multi-turn agent sessions (long document, tool calls, follow-ups), 1 to N concurrent agents |
| [`textgen.py`](textgen.py) | deterministic synthetic prompts with exact token lengths |
| [`common.py`](common.py) | HTTP / streaming client, GPU lock, run metadata |

```bash
python3 benchmarks/bench.py --label my-run --workloads standard --concurrency 1,2,4 --reasoning off
python3 benchmarks/agent_suite.py --label my-run --contexts 4096,32768 --sessions 1,2,4 --runs 3
```

**Throughput numbers in RECIPES.md come from AIPerf**, not from these scripts:
[`nemotron-super-vl/scripts/aiperf_sweep.sh`](../nemotron-super-vl/scripts/aiperf_sweep.sh) runs AIPerf 0.13 with
`--num-dataset-entries N+C`, so every measured prompt is new. Without that flag AIPerf replays its warm-up prompts in
the measured phase, and on a server with prefix caching about half of them become cache hits.

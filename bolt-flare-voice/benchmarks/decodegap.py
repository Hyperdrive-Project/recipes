#!/usr/bin/env python3
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Decode gap probe: one single-user streaming generation (ignore_eos, reasoning off), with Flare metrics sampled
before and after. Reports wall decode tok/s, speculative GPU time share, emitted tokens per round by width,
and the implied non-GPU (host/scheduling) time between rounds."""
import json, re, sys, time, urllib.request

BASE = "http://127.0.0.1:8000"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024


def metrics():
    d = {}
    for line in urllib.request.urlopen(BASE + "/metrics").read().decode().splitlines():
        m = re.match(r"(\w+)(\{[^}]*\})? (\S+)", line)
        if m and not line.startswith("#"):
            d[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return d


model = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": "Tell me a long, detailed story about a small robot "
        "who learns to talk with the people in a busy workshop. Keep going for a long time."}],
        "max_tokens": N, "ignore_eos": True, "temperature": 0.7, "top_p": 0.95, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}
a = metrics()
req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); first = None; n = 0
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if line.startswith("data:") and line != "data: [DONE]":
            d = json.loads(line[5:])
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content"):
                    n += 1; first = first or time.time()
t1 = time.time(); b = metrics()
delta = {k: b[k] - a.get(k, 0) for k in b if b[k] - a.get(k, 0)}
gpu = sum(v for k, v in delta.items() if k.startswith("flare_speculative_gpu_nanoseconds_total")) / 1e9
emitted = sum(v for k, v in delta.items() if k.startswith("flare_speculative_emitted_tokens_total"))
rounds = sum(v for k, v in delta.items() if k.startswith("flare_speculative_rounds_total"))
wall = t1 - first
print(f"chunks={n} emitted={emitted:.0f} rounds={rounds:.0f} wall_decode={wall:.2f}s -> {emitted / wall:.1f} tok/s wall; "
      f"spec GPU {gpu:.2f}s ({100 * gpu / wall:.0f}% of wall) -> {emitted / max(gpu, 1e-9):.1f} tok/s GPU; "
      f"host gap per round {(wall - gpu) / max(rounds, 1) * 1e3:.2f} ms; GPU per round {gpu / max(rounds, 1) * 1e3:.2f} ms; "
      f"tokens/round {emitted / max(rounds, 1):.2f}")
for k in sorted(delta):
    if k.startswith("flare_speculative_rounds_total") or k.startswith("flare_scheduler_execution"):
        print(" ", k, round(delta[k], 1))

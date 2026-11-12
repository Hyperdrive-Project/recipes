#!/usr/bin/env python3
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""LLM-side voice latency profile on cached robot turns (reasoning off, streaming):
time to the k-th streamed content chunk (k = 1, 3, 5, 8, 12, 15) after request send, with the conversation's
turns spaced by an idle gap (0 s back-to-back, 2 s, 6 s: a robot listens for seconds between replies, so the GPU
may idle down before the next turn). Uses one long system prompt warmed once."""
import os
import argparse, json, random, statistics, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robot_turns as R

ap = argparse.ArgumentParser(); ap.add_argument("--gaps", default="0,2,6"); ap.add_argument("--turns", type=int, default=10)
a = ap.parse_args()
model = json.load(urllib.request.urlopen(R.BASE + "/v1/models"))["data"][0]["id"]
KS = (1, 3, 5, 8, 12, 15)


def turn(messages):
    body = {"model": model, "messages": messages, "max_tokens": 48, "temperature": 0.7, "top_p": 0.95, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(R.BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); marks = {}; n = 0; text = ""
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                for c in json.loads(line[5:]).get("choices", []):
                    piece = c.get("delta", {}).get("content") or ""
                    if piece:
                        n += 1; text += piece
                        if n in KS:
                            marks[n] = time.time() - t0
    return marks, text


rng = random.Random(777)
system = R.system_prompt(rng, 7)
turn([{"role": "system", "content": system}, {"role": "user", "content": "Hello."}])  # warm the system prompt
for gap in [float(g) for g in a.gaps.split(",")]:
    messages = [{"role": "system", "content": system}]
    rows = []
    for u in rng.sample(R.UTTER, a.turns):
        time.sleep(gap)
        messages.append({"role": "user", "content": u})
        marks, text = turn(messages)
        messages.append({"role": "assistant", "content": text})
        rows.append(marks)
    parts = []
    for k in KS:
        v = sorted(m[k] for m in rows[1:] if k in m)
        if v:
            parts.append(f"k{k} p50 {statistics.median(v) * 1e3:.0f} ms (n={len(v)})")
    print(f"gap {gap:.0f}s: " + "; ".join(parts), flush=True)

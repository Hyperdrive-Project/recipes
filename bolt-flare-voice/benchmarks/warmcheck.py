#!/usr/bin/env python3
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Why does a boot warm-up not make the first robot turn cached? Sends variants of a warm-up request for fresh
system prompts, then the real first turn, and prints prompt/cached tokens and TTFT of the first turn."""
import os
import json, random, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robot_turns as R

model = json.load(urllib.request.urlopen(R.BASE + "/v1/models"))["data"][0]["id"]


def call(messages, max_tokens, stream=False):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(R.BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=300)); return r, time.time() - t0


args = type("A", (), {"max_tokens": 32, "reasoning": "off"})()
for name, warm in (("none", None), ("hello-1tok", ("Hello.", 1)), ("hello-8tok", ("Hello.", 8)),
                   ("system-only-8tok", (None, 8)), ("hello-1tok-wait2s", ("Hello.", 1, 2.0))):
    system = R.system_prompt(random.Random(hash(name) % 10000 + 50000), 99)
    if warm:
        msgs = [{"role": "system", "content": system}] + ([{"role": "user", "content": warm[0]}] if warm[0] else [])
        call(msgs, warm[1])
        if len(warm) > 2:
            time.sleep(warm[2])
    first = [{"role": "system", "content": system}, {"role": "user", "content": "Hey, what's your name?"}]
    r = R.turn(model, first, args)
    print(f"{name:20s} prompt={r['prompt']} cached={r['cached']} ttft={r['ttft']:.3f}", flush=True)

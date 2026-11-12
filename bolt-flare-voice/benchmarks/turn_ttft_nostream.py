#!/usr/bin/env python3
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Non-streaming TTFT proxy for direct-mode servers: max_tokens=1 requests on a growing conversation.
Turn 0 is uncached (long system prompt); later turns add one short user message each. Prints seconds per turn."""
import os
import json, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
import robot_turns as R

BASE = "http://127.0.0.1:8000"
model = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
rng = random.Random(4242)
messages = [{"role": "system", "content": R.system_prompt(rng, 77)}]
out = []
for i, u in enumerate(rng.sample(R.UTTER, 6)):
    messages.append({"role": "user", "content": u})
    body = {"model": model, "messages": messages, "max_tokens": 1, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t0
    out.append((i, r.get("usage", {}).get("prompt_tokens"), (r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens"), round(dt, 3)))
    messages.append({"role": "assistant", "content": "Okay."})
print("turn, prompt tokens, cached, seconds:", out)

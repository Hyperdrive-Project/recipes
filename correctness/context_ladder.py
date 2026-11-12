# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Phase 0/1 context ladder: needle retrieval at increasing prompt lengths, one request at a time.

usage: context_ladder.py <base_url> <target_tokens,...>
Reports prompt_tokens (from server usage), TTFT, decode tok/s, and whether the needle was found. Stdlib only.
"""
import json
import random
import sys
import time
import urllib.request

base = sys.argv[1]
targets = [int(x) for x in sys.argv[2].split(",")]
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
rng = random.Random(1234)
words = ("river stone harbor lantern orchard meadow signal copper ledger window garden valley engine "
         "bridge market candle forest letter mirror island season thunder pocket ladder").split()


def sentence() -> str:
    return " ".join(rng.choice(words) for _ in range(12)).capitalize() + "."


for target in targets:
    # ~1.3 tokens per filler word for this tokenizer is a guess; server usage reports the real count.
    n_sent = int(target / 16)
    sents = [sentence() for _ in range(n_sent)]
    code = str(rng.randint(10000, 99999))
    sents.insert(n_sent // 2, f"The secret access code for the vault is {code}.")
    prompt = " ".join(sents) + "\n\nWhat is the secret access code for the vault? Reply with only the number."
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 32,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    piece = ch.get("delta", {}).get("content") or ""
                    if piece and ttft is None:
                        ttft = time.time() - t0
                    text += piece
        err = None
    except Exception as e:  # report and continue so one failure does not hide the rest of the ladder
        err = repr(e)[:300]
    wall = time.time() - t0
    pt = usage["prompt_tokens"] if usage else None
    print(json.dumps({"target": target, "prompt_tokens": pt, "ttft_s": round(ttft, 2) if ttft else None,
                      "prefill_tok_s": round(pt / ttft, 1) if pt and ttft else None, "wall_s": round(wall, 2),
                      "found": code in text, "answer": text.strip()[:40], "error": err}), flush=True)

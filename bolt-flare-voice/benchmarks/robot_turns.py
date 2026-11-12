#!/usr/bin/env python3
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Robot/speech conversation benchmark: multi-turn sessions with a long cached system prompt and short user turns.

Per turn (streaming chat, reasoning off by default): TTFT, time to first phrase (the 15th streamed content chunk, about
15 tokens: enough to start text-to-speech), and decode tok/s after the first token. Sessions use different persona
details and utterances, so no session reuses another's prefix; within a session the history is reused, as on a robot.
usage: robot_turns.py --label L [--sessions 3] [--turns 8] [--reasoning off|on] [--max-tokens 96] [--out F]"""
import argparse, json, random, statistics, time, urllib.request

BASE = "http://127.0.0.1:8000"
UTTER = ["Hey, what's your name?", "Can you look at the table and tell me what's on it?", "Pick up the red block please.",
         "Wait, stop. Put it back.", "What time do you think it is?", "Tell me a short joke.", "How far is the door from you?",
         "Can you wave at my friend?", "What did I ask you to do earlier?", "Turn your head to the left a bit.",
         "Is the light too bright for your camera?", "Say goodbye to everyone.", "What's two plus seventeen?",
         "Do you remember where you put the block?", "Give me one tip for staying focused."]


def system_prompt(rng, session):
    rules = [f"Rule {i}: " + rng.choice(["Keep answers to one or two short sentences.", "Never describe your motors in detail.",
             "If unsure what the user wants, ask one short question.", "Use plain words; you are speaking out loud.",
             "Confirm physical actions before doing them.", "Mention safety when moving near people.",
             "Refer to the user by name when you know it.", "Do not read lists aloud; summarize them."]) +
             f" (policy ref {session}-{i}-{rng.randint(1000, 9999)})" for i in range(60)]
    return (f"You are Reachy Mini, a small friendly desk robot (unit {session}-{rng.randint(100, 999)}) with a camera, two antennas "
            "and a speaker. You talk with people in real time; your replies are spoken by a text-to-speech voice.\n"
            + "\n".join(rules))


def turn(model, messages, a):
    body = {"model": model, "messages": messages, "max_tokens": a.max_tokens, "temperature": 0.7, "top_p": 0.95,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": a.reasoning == "on"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, phrase, chunks, text, usage = time.time(), None, None, 0, "", {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if (delta.get("reasoning_content") or delta.get("reasoning")) and first is None:
                    first = time.time()  # first token of any kind (thinking counts for TTFT)
                piece = delta.get("content") or ""
                if piece:
                    chunks += 1; text += piece
                    now = time.time()
                    first = first or now
                    if chunks == 15:
                        phrase = now
    t1 = time.time()
    out = usage.get("completion_tokens") or chunks
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return dict(ttft=None if first is None else first - t0, phrase=None if phrase is None else phrase - t0,
                decode=(out - 1) / (t1 - first) if first and out > 1 and t1 > first else None,
                prompt=usage.get("prompt_tokens"), cached=cached, completion=out, reply=text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True); ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--turns", type=int, default=8); ap.add_argument("--reasoning", default="off")
    ap.add_argument("--max-tokens", type=int, default=96); ap.add_argument("--out")
    a = ap.parse_args()
    model = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]
    rows = []
    for s in range(a.sessions):
        rng = random.Random(9000 + s)
        messages = [{"role": "system", "content": system_prompt(rng, s)}]
        for t, u in enumerate(rng.sample(UTTER, a.turns)):
            messages.append({"role": "user", "content": u})
            r = turn(model, messages, a)
            messages.append({"role": "assistant", "content": r.pop("reply")})
            rows.append(dict(session=s, turn=t, **r))
    def stat(key, skip_first=True):
        v = [r[key] for r in rows if r[key] is not None and (not skip_first or r["turn"] > 0)]
        if not v:
            return "n/a"
        v.sort(); return f"p50 {statistics.median(v):.3f} p95 {v[min(len(v) - 1, int(.95 * len(v)))]:.3f} (n={len(v)})"
    summary = (f"[robot {a.label}] reasoning {a.reasoning}: cached turns: TTFT {stat('ttft')} s; first phrase {stat('phrase')} s; "
               f"decode {stat('decode')} tok/s | first turns TTFT {[None if r['ttft'] is None else round(r['ttft'], 3) for r in rows if r['turn'] == 0]} "
               f"| mean prompt {statistics.mean(r['prompt'] or 0 for r in rows):.0f} tok, cached {statistics.mean((r['cached'] or 0) for r in rows if r['turn'] > 0):.0f}")
    print(summary)
    if a.out:
        json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

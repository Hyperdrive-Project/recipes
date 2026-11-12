# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Standing performance suite: realistic multi-turn agent sessions, single user and N concurrent agents.

Each session is an agent conversation: a system prompt, a user message carrying a project document of ~K tokens and a task,
then turns where the model may call tools (tools are sent on every turn, as agent frameworks do), a tool result, and two
follow-up requests. Reasoning on (model default) unless --reasoning off. Sampling per the model card (T=1.0, top_p=0.95)
with a per-session seed. Every call is streamed so it splits into time-to-first-token and generation time.

N sessions start together (--sessions N) and each runs its turns sequentially, so the server sees N concurrent agents.
Documents are unique per session (no cross-session prefix sharing) unless --shared-doc.

Outputs one JSON per (context, sessions, run) under benchmarks/results/agent/ and prints a one-line summary.

usage: agent_suite.py --label L --contexts 4096,32768 --sessions 1,2,4 --runs 3 [--kind prose|code] [--reasoning on|off]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import textgen  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "search_docs", "description": "Search the project documents for a phrase.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run the project's test suite or one test file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
]
SYSTEM = ("You are a careful engineering agent working on a user's project. Use the tools when they help, "
          "explain your reasoning briefly, and keep answers concrete.")
TASKS = {
    "prose": ("Find the three most important open problems described in this material and decide which one to tackle "
              "first. Search the documents if you need to."),
    "code": ("Review this code for bugs and missing error handling. Run the tests if you need to, then propose the most "
             "important fix."),
}
FOLLOWUPS = ["Summarize the key risks of your plan in five bullet points.",
             "Write the first concrete step: the exact change or the text of the report section, ready to use."]


def p(vals, q):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


class MemSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event(); self.min_avail = None; self.max_swap = 0; self.swap0 = None; self.power = []

    def run(self):
        while not self.stop.is_set():
            m = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1); m[k] = int(v.split()[0])
            avail = m["MemAvailable"] / 1048576; swap = (m["SwapTotal"] - m["SwapFree"]) / 1048576
            self.min_avail = avail if self.min_avail is None else min(self.min_avail, avail)
            self.swap0 = swap if self.swap0 is None else self.swap0
            self.max_swap = max(self.max_swap, swap)
            try:
                w = float(subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0])
                self.power.append((time.time(), w))
            except Exception:
                pass
            self.stop.wait(1.0)


def call(base, model, messages, seed, reasoning, max_tokens):
    body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "seed": seed, "stream": True, "stream_options": {"include_usage": True}}
    if reasoning == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = None; content = ""; usage = None; tcalls = {}; err = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:]); usage = d.get("usage") or usage
                for ch in d.get("choices", []):
                    dl = ch.get("delta", {})
                    if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or dl.get("tool_calls"):
                        tf = tf or time.perf_counter()
                    content += dl.get("content") or ""
                    for tc in dl.get("tool_calls") or []:
                        s = tcalls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                        s["id"] = tc.get("id") or s["id"]; fn = tc.get("function") or {}
                        s["function"]["name"] += fn.get("name") or ""; s["function"]["arguments"] += fn.get("arguments") or ""
    except Exception as e:
        body_txt = ""
        if hasattr(e, "read"):
            try:
                body_txt = e.read().decode(errors="replace")[:300]
            except Exception:
                pass
        err = (repr(e) + " " + body_txt)[:400]
    t1 = time.perf_counter(); u = usage or {}
    ct = u.get("completion_tokens") or 0
    rec = {"ttft_s": round((tf or t1) - t0, 3), "e2e_s": round(t1 - t0, 3), "prompt_tokens": u.get("prompt_tokens"),
           "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "completion_tokens": ct,
           "decode_tok_s": round(ct / (t1 - tf), 2) if tf and t1 > tf and ct > 1 else None, "tool_calls": len(tcalls), "error": err}
    msg = {"role": "assistant", "content": content}
    if tcalls:
        msg["tool_calls"] = [tcalls[k] for k in sorted(tcalls)]
    return rec, msg


def session(base, model, ctx_tokens, kind, seed, reasoning, max_tokens, shared_doc):
    doc = textgen.corpus(kind, 4242 if shared_doc else seed, int(ctx_tokens * 3.6))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Project material:\n\n{doc}\n\nTask: {TASKS[kind]}"}]
    calls = []; t0 = time.perf_counter()
    rec, m = call(base, model, msgs, seed, reasoning, max_tokens); calls.append(("task", rec)); msgs.append(m)
    if m.get("tool_calls"):
        for tc in m["tool_calls"]:
            msgs.append({"role": "tool", "tool_call_id": tc.get("id") or "call_0",
                         "content": json.dumps({"ok": True, "result": "3 matches found; tests: 41 passed, 2 failed (test_io.py)"})})
        rec, m = call(base, model, msgs, seed + 1, reasoning, max_tokens); calls.append(("after_tool", rec)); msgs.append(m)
    for i, f in enumerate(FOLLOWUPS):
        msgs.append({"role": "user", "content": f})
        rec, m = call(base, model, msgs, seed + 2 + i, reasoning, max_tokens); calls.append((f"followup{i + 1}", rec)); msgs.append(m)
    return {"seed": seed, "e2e_s": round(time.perf_counter() - t0, 2), "calls": calls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--contexts", default="4096,32768,65536,100000")
    ap.add_argument("--sessions", default="1")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--kind", default="prose", choices=["prose", "code"])
    ap.add_argument("--reasoning", default="on", choices=["on", "off"])
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--shared-doc", action="store_true")
    ap.add_argument("--seed-base", type=int, default=1000)
    a = ap.parse_args()
    model = json.load(urllib.request.urlopen(a.base_url + "/v1/models"))["data"][0]["id"]
    out_dir = ROOT / "benchmarks" / "results" / "agent"; out_dir.mkdir(parents=True, exist_ok=True)
    for ctx in [int(x) for x in a.contexts.split(",")]:
        for n in [int(x) for x in a.sessions.split(",")]:
            for run in range(a.runs):
                mem = MemSampler(); mem.start(); t0 = time.time()
                seeds = [a.seed_base + 10000 * run + 100 * i + ctx % 97 for i in range(n)]
                with ThreadPoolExecutor(max_workers=n) as ex:
                    sess = list(ex.map(lambda s: session(a.base_url, model, ctx, a.kind, s, a.reasoning, a.max_tokens, a.shared_doc), seeds))
                wall = time.time() - t0; mem.stop.set(); mem.join(timeout=3)
                calls = [c for s in sess for _, c in s["calls"]]
                errs = [c["error"] for c in calls if c["error"]]
                tot = sum(c["completion_tokens"] for c in calls)
                pw = mem.power
                energy = sum((pw[i + 1][0] - pw[i][0]) * (pw[i + 1][1] + pw[i][1]) / 2 for i in range(len(pw) - 1)) if len(pw) > 1 else None
                summ = {"label": a.label, "context": ctx, "sessions": n, "run": run, "kind": a.kind, "reasoning": a.reasoning,
                        "wall_s": round(wall, 2), "completion_tokens": tot, "total_tok_s": round(tot / wall, 2),
                        "per_agent_tok_s_median": p([sum(c["completion_tokens"] for _, c in s["calls"]) / s["e2e_s"] for s in sess], 0.5),
                        "ttft_p50": p([c["ttft_s"] for c in calls], 0.5), "ttft_p95": p([c["ttft_s"] for c in calls], 0.95),
                        "ttft_first_turn_p50": p([s["calls"][0][1]["ttft_s"] for s in sess], 0.5),
                        "decode_tok_s_p50": p([c["decode_tok_s"] for c in calls], 0.5),
                        "session_e2e_p50": p([s["e2e_s"] for s in sess], 0.5), "session_e2e_p95": p([s["e2e_s"] for s in sess], 0.95),
                        "tool_call_sessions": sum(1 for s in sess if s["calls"][0][1]["tool_calls"]),
                        "min_mem_available_gib": round(mem.min_avail, 1) if mem.min_avail else None,
                        "swap_growth_gib": round(mem.max_swap - (mem.swap0 or 0), 2),
                        "energy_j": round(energy, 1) if energy else None,
                        "j_per_token": round(energy / tot, 3) if energy and tot else None, "errors": len(errs)}
                rec = {"summary": summ, "sessions_detail": sess, "errors": errs[:5],
                       "measured_at": datetime.now(timezone.utc).isoformat()}
                f = out_dir / f"{a.label}-k{ctx}-c{n}-r{run}.json"
                f.write_text(json.dumps(rec, indent=1))
                print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()

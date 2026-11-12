#!/usr/bin/env python3
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Held-out quality suite for any OpenAI-compatible server (trtllm-serve, vLLM, SGLang).

Tasks (temperature 0 unless noted; reasoning mode recorded):
  math         50 GSM8K test items (pinned revision, correctness/data/), exact numeric match
  code         30 HumanEval items, generated code executed in a sandboxed subprocess (timeout, rlimits,
               network blocked), pass@1
  tools        11 hand-written function-calling cases: tool name + JSON args (server tool parser, with a
               client-side qwen3_coder XML fallback when the server has none)
  image        8 procedurally generated images (shapes, counting, digits, word, bar chart, position)
  determinism  8 fixed greedy prompts (reasoning off, 128 tokens) diffed token-by-token against a stored
               reference (correctness/reference/<name>.json); logprob drift when both sides have logprobs
  needle       needle retrieval at 4K/8K/16K/32K prompt tokens x depths 10/50/90%

Output: correctness/results/<run_id>.json with per-item details and a `quality` dict (the
no-regression gate).

  python3 correctness/eval.py --label trtllm-official --reasoning off
  python3 correctness/eval.py --label baseline --save-reference baseline
  python3 correctness/eval.py --self-test     # graders only, no server
  python3 correctness/eval.py --dry-run       # full pipeline against the mock server
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.common import (REASONING_MODES, HTTPError, git_info, gpu_lock, guess_framework,  # noqa: E402
                               http_post_json, make_run_id, server_identity, sha256_file, utc_now_iso)
from correctness import graders as G  # noqa: E402
from correctness import tasks as T  # noqa: E402

SCHEMA = "nemotron-hyperdrive/correctness/v1"
RESULTS_DIR = ROOT / "correctness" / "results"
REFERENCE_DIR = ROOT / "correctness" / "reference"
DRYRUN_DIR = Path("/tmp/nemotron-hyperdrive-dryrun")
ALL_TASKS = ["math", "code", "tools", "image", "determinism", "needle"]
GATED_TASKS = ["math", "code", "tools", "image", "needle"]


def strip_think(text: str) -> str:
    """If the server has no reasoning parser, drop a leading <think>...</think> block from content."""
    if text and "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text or ""


class Evaluator:
    def __init__(self, args: argparse.Namespace, base_url: str):
        self.args = args
        self.base_url = base_url.rstrip("/")
        self.ident = server_identity(self.base_url)
        if "model_id" not in self.ident:
            raise SystemExit(f"server at {self.base_url} not reachable: {self.ident.get('error')}")
        self.model = self.ident["model_id"]
        self.tkw = dict(REASONING_MODES[args.reasoning])
        self._tok = None
        self._tok_lock = threading.Lock()
        self.notes: list[str] = []

    @property
    def tok(self):
        """Lazily loaded model tokenizer (None if unavailable)."""
        with self._tok_lock:
            if self._tok is None:
                try:
                    from benchmarks.textgen import ChatTokenizer
                    self._tok = ChatTokenizer(self.args.tokenizer)
                except Exception as e:  # noqa: BLE001
                    self.notes.append(f"tokenizer unavailable: {e}")
                    self._tok = False
            return self._tok or None

    # ------------------------------------------------------------------ request
    def chat(self, messages: list[dict[str, Any]], max_tokens: int, tkw: dict[str, Any] | None = None,
             **extra: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": 0.0, "chat_template_kwargs": tkw if tkw is not None else self.tkw}
        if self.args.seed is not None:
            body["seed"] = self.args.seed
        body.update({k: v for k, v in extra.items() if v is not None})
        t0 = time.monotonic()
        try:
            r = http_post_json(self.base_url + "/v1/chat/completions", body, timeout=self.args.timeout)
        except HTTPError as e:
            return {"error": str(e), "status": e.status, "latency_s": time.monotonic() - t0}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}", "latency_s": time.monotonic() - t0}
        ch = (r.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "message": msg, "finish_reason": ch.get("finish_reason"), "usage": r.get("usage") or {},
                "logprobs": ((ch.get("logprobs") or {}).get("content")) if isinstance(ch.get("logprobs"), dict) else None,
                "latency_s": time.monotonic() - t0}

    def pmap(self, fn: Callable[[Any], dict[str, Any]], items: list[Any], workers: int | None = None) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max(1, workers or self.args.parallel)) as ex:
            return list(ex.map(fn, items))

    def _limit(self, items: list[Any]) -> list[Any]:
        return items[: self.args.limit] if self.args.limit else items

    def _mt(self, task: str) -> int:
        return int(T.MAX_TOKENS[task][self.args.reasoning] * self.args.max_tokens_scale)

    @staticmethod
    def _base(r: dict[str, Any]) -> dict[str, Any]:
        return {"error": r.get("error"), "finish_reason": r.get("finish_reason"),
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"), "latency_s": r.get("latency_s")}

    @staticmethod
    def _score(items: list[dict[str, Any]], key: str = "correct") -> dict[str, Any]:
        n = len(items)
        c = sum(1 for i in items if i.get(key))
        return {"n": n, "correct": c, "accuracy": (c / n) if n else None,
                "errors": sum(1 for i in items if i.get("error")),
                "truncated": sum(1 for i in items if i.get("finish_reason") == "length"), "items": items}

    # ------------------------------------------------------------------ tasks
    def task_math(self) -> dict[str, Any]:
        data = self._limit(T.load_jsonl("gsm8k_test_50.jsonl"))

        def one(it: dict[str, Any]) -> dict[str, Any]:
            r = self.chat([{"role": "user", "content": T.MATH_PROMPT.format(question=it["question"])}], self._mt("math"))
            gold = G.gsm8k_gold(it["answer"])
            pred, how = G.extract_number(strip_think(r.get("content", "")))
            return {"id": f"gsm8k-{it['_index']}", **self._base(r), "gold": str(gold), "pred": str(pred),
                    "extraction": how, "correct": G.numbers_equal(pred, gold),
                    "response": strip_think(r.get("content", ""))[-600:]}
        return self._score(self.pmap(one, data))

    def task_code(self) -> dict[str, Any]:
        data = self._limit(T.load_jsonl("humaneval_30.jsonl"))

        def one(it: dict[str, Any]) -> dict[str, Any]:
            r = self.chat([{"role": "user", "content": T.CODE_PROMPT.format(prompt=it["prompt"])}], self._mt("code"))
            code = G.extract_code(strip_think(r.get("content", "")))
            res = G.run_sandboxed(G.build_program(it["prompt"], code, it["test"], it["entry_point"]),
                                  timeout_s=self.args.code_timeout) if not r.get("error") else {"status": "no-response"}
            return {"id": it["task_id"], **self._base(r), "status": res["status"], "correct": res["status"] == "passed",
                    "stderr_tail": res.get("stderr_tail", "")[-300:], "code": code[:2000]}
        out = self._score(self.pmap(one, data))
        out["pass_at_1"] = out["accuracy"]
        out["sandbox"] = G.sandbox_method()
        return out

    def task_tools(self) -> dict[str, Any]:
        cases = self._limit(T.TOOL_CASES)

        def one(case: dict[str, Any]) -> dict[str, Any]:
            msgs = [{"role": "user", "content": case["prompt"]}]
            r = self.chat(msgs, self._mt("tools"), tools=case["tools"], tool_choice="auto")
            mode = "auto"
            if r.get("error") and r.get("status") == 400:  # e.g. vLLM without --enable-auto-tool-choice
                r = self.chat(msgs, self._mt("tools"), tools=case["tools"], tool_choice="none")
                mode = "none(retry)"
            calls = G.server_tool_calls(r.get("message") or {})
            parsed_by = "server"
            if not calls:
                calls = G.parse_xml_tool_calls(strip_think(r.get("content", "")), case["tools"])
                parsed_by = "client-xml-fallback" if calls else "none"
            g = G.grade_tool_case(calls, case["expect"])
            return {"id": case["id"], **self._base(r), **g, "tool_choice": mode, "parsed_by": parsed_by,
                    "calls": calls, "expected": case["expect"], "content": strip_think(r.get("content", ""))[:300]}
        out = self._score(self.pmap(one, cases))
        out["name_accuracy"] = sum(1 for i in out["items"] if i.get("name_ok")) / max(1, out["n"])
        return out

    def task_image(self) -> dict[str, Any]:
        imgs = T.make_images()
        cases = self._limit(T.IMAGE_CASES)

        def one(case: dict[str, Any]) -> dict[str, Any]:
            msgs = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": T.image_data_url(imgs[case["image"]])}},
                {"type": "text", "text": case["prompt"]}]}]
            r = self.chat(msgs, self._mt("image"))
            ans = strip_think(r.get("content", ""))
            ok = G.short_answer_ok(ans, case["expected"], case["kind"], case.get("alternatives"))
            return {"id": case["id"], **self._base(r), "expected": case["expected"], "answer": ans[:200], "correct": ok}
        out = self._score(self.pmap(one, cases, workers=1))  # concurrent image requests hang TRT-LLM (RESEARCH_LOG 2026-10-21)
        out["image_sha256"] = {k: hashlib.sha256(v).hexdigest() for k, v in imgs.items()}
        return out

    def task_needle(self) -> dict[str, Any]:
        from benchmarks.textgen import build_exact_prompt
        lengths = [int(x.lower().rstrip("k")) * 1024 if x.lower().endswith("k") else int(x)
                   for x in self.args.needle_lengths.split(",") if x.strip()]
        cases = [T.needle_case(L, d, i) for i, (L, d) in enumerate((L, d) for L in lengths for d in T.NEEDLE_DEPTHS)]
        cases = self._limit(cases)
        tok = self.tok
        if tok is None:
            return {"n": 0, "accuracy": None, "skipped": "tokenizer unavailable (needed for exact lengths)", "items": []}

        def one(case: dict[str, Any]) -> dict[str, Any]:
            p = build_exact_prompt(tok, case["length"], "text", case["seed"], self.tkw,
                                   make_content=lambda body: T.insert_needle(body, case["needle"], case["depth"])
                                   + "\n\n" + case["question"])
            r = self.chat(p["messages"], self._mt("needle"))
            ans = strip_think(r.get("content", ""))
            ok = G.short_answer_ok(ans, case["expected"], "code")
            return {"id": case["id"], **self._base(r), "length": case["length"], "depth": case["depth"],
                    "prompt_tokens_local": p["prompt_tokens"], "exact": p["exact"],
                    "expected": case["expected"], "answer": ans[:200], "correct": ok}
        out = self._score(self.pmap(one, cases))
        by_len: dict[str, list[bool]] = {}
        for it in out["items"]:
            by_len.setdefault(str(it["length"]), []).append(bool(it["correct"]))
        out["accuracy_by_length"] = {k: sum(v) / len(v) for k, v in by_len.items()}
        return out

    def task_determinism(self) -> dict[str, Any]:
        prompts = self._limit(T.DETERMINISM_PROMPTS)
        tkw = dict(REASONING_MODES["off"])
        mt = T.MAX_TOKENS["determinism"]["off"]
        lp_ok = [not self.args.no_logprobs]

        def one(pr: dict[str, str]) -> dict[str, Any]:
            msgs = [{"role": "user", "content": pr["prompt"]}]
            r = self.chat(msgs, mt, tkw=tkw, logprobs=True if lp_ok[0] else None)
            if r.get("error") and lp_ok[0] and r.get("status") == 400:
                lp_ok[0] = False
                r = self.chat(msgs, mt, tkw=tkw)
            text = r.get("content", "")
            tok = self.tok
            ids = tok.encode(text) if tok else text.split()
            lps = r.get("logprobs") or None
            return {"id": pr["id"], **self._base(r), "text": text, "token_ids": ids,
                    "token_source": "local-tokenizer" if tok else "whitespace",
                    "server_tokens": [e.get("token") for e in lps] if lps else None,
                    "server_logprobs": [e.get("logprob") for e in lps] if lps else None}
        items = self.pmap(one, prompts)
        out: dict[str, Any] = {"n": len(items), "items": items, "reference": self.args.reference,
                               "logprobs_available": any(i["server_logprobs"] for i in items)}
        ref_path = Path(self.args.reference_dir) / f"{self.args.reference}.json"
        if ref_path.exists():
            ref = json.loads(ref_path.read_text())
            ref_items = {i["id"]: i for i in ref["items"]}
            rates, exact, drifts = [], [], []
            for it in items:
                ri = ref_items.get(it["id"])
                if not ri or it.get("error"):
                    continue
                if it["server_tokens"] and ri.get("server_tokens"):
                    d = G.token_diff(ri["server_tokens"], it["server_tokens"])
                    d["token_source"] = "server-logprob-tokens"
                    if it["server_logprobs"] and ri.get("server_logprobs"):
                        k = d["prefix_match"]
                        diffs = [abs(a - b) for a, b in zip(ri["server_logprobs"][:k], it["server_logprobs"][:k])
                                 if a is not None and b is not None]
                        d["logprob_abs_diff_mean"] = sum(diffs) / len(diffs) if diffs else None
                        d["logprob_abs_diff_max"] = max(diffs) if diffs else None
                        if diffs:
                            drifts.append(d["logprob_abs_diff_mean"])
                else:
                    d = G.token_diff(ri["token_ids"], it["token_ids"])
                    d["token_source"] = it["token_source"]
                it["vs_reference"] = d
                rates.append(d["positionwise_match_rate"])
                exact.append(d["exact"])
            out.update({"reference_path": str(ref_path), "reference_label": ref.get("label"),
                        "token_match_rate_mean": sum(rates) / len(rates) if rates else None,
                        "exact_match_frac": sum(exact) / len(exact) if exact else None,
                        "logprob_abs_diff_mean": sum(drifts) / len(drifts) if drifts else None,
                        "compared": len(rates)})
        else:
            out.update({"reference_path": str(ref_path), "token_match_rate_mean": None, "exact_match_frac": None,
                        "note": "no stored reference; run with --save-reference NAME to create one"})
        return out

    # ------------------------------------------------------------------ run
    def run(self, task_names: list[str], run_id: str, out_dir: Path, dry_run: bool = False) -> dict[str, Any]:
        a = self.args
        started = utc_now_iso()
        t0 = time.monotonic()
        res: dict[str, Any] = {}
        for name in task_names:
            ts = time.monotonic()
            print(f"[eval] {name} ...", flush=True)
            try:
                res[name] = getattr(self, f"task_{name}")()
            except FileNotFoundError as e:
                res[name] = {"n": 0, "accuracy": None, "skipped": str(e), "items": []}
            res[name]["duration_s"] = time.monotonic() - ts
            acc = res[name].get("accuracy", res[name].get("token_match_rate_mean"))
            print(f"[eval]   {name}: {acc if acc is None else round(acc, 4)} "
                  f"({res[name].get('correct', '-')}/{res[name].get('n', 0)}) in {res[name]['duration_s']:.1f}s", flush=True)
        manifest = ROOT / "correctness" / "data" / "MANIFEST.json"
        summary = {
            "schema": SCHEMA, "run_id": run_id, "label": a.label, "framework": a.framework or guess_framework(self.ident),
            "image_tag": a.image_tag, "config": {"path": a.config, "sha256": sha256_file(a.config) if a.config else None},
            "notes": a.notes, "git": git_info(), "server": self.ident, "dry_run": dry_run,
            "reasoning_mode": a.reasoning, "chat_template_kwargs": self.tkw,
            "sampling": {"temperature": 0.0, "seed": a.seed, "max_tokens": {k: self._mt(k) for k in task_names},
                         "determinism": "reasoning off, 128 tokens, temperature 0"},
            "limit": a.limit, "started_utc": started, "finished_utc": utc_now_iso(),
            "duration_s": time.monotonic() - t0,
            "data_manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
            "data_manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
            "tasks": res,
            "quality": {k: res[k].get("accuracy") for k in GATED_TASKS if k in res and res[k].get("n")},
            "determinism": {k: res["determinism"].get(k) for k in ("token_match_rate_mean", "exact_match_frac",
                                                                   "logprob_abs_diff_mean", "reference")}
            if "determinism" in res else None,
            "notes_runtime": self.notes,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_id}.json"
        path.write_text(json.dumps(summary, indent=1) + "\n")
        summary["_path"] = str(path)
        if a.save_reference and "determinism" in res:
            ref_dir = Path(a.reference_dir)
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_path = ref_dir / f"{a.save_reference}.json"
            if ref_path.exists() and not a.overwrite_reference:
                print(f"[eval] reference {ref_path} exists; not overwriting (use --overwrite-reference)")
            else:
                ref_path.write_text(json.dumps({"label": a.label, "run_id": run_id, "created_utc": utc_now_iso(),
                                                "server": self.ident, "items": res["determinism"]["items"]},
                                               indent=1) + "\n")
                print(f"[eval] saved determinism reference -> {ref_path}")
        print(f"[eval] quality={json.dumps({k: (round(v, 4) if v is not None else None) for k, v in summary['quality'].items()})} -> {path}")
        return summary


# ============================================================================ self-test (no server)
def self_test(args: argparse.Namespace) -> int:
    fails: list[str] = []

    def check(c: bool, m: str) -> None:
        if not c:
            fails.append(m)
            print("  FAIL:", m)

    print("[self-test] math extraction")
    data = T.load_jsonl("gsm8k_test_50.jsonl")
    check(len(data) == 50, "gsm8k has 50 items")
    for it in data:
        g = G.gsm8k_gold(it["answer"])
        check(g is not None, f"gold parse {it['_index']}")
        check(G.numbers_equal(G.extract_number(f"blah 12 then\nAnswer: {g}")[0], g), f"extract {it['_index']}")
    check(G.numbers_equal(G.extract_number("so the total is **Answer: $1,234.00**")[0], G.gsm8k_gold("#### 1234")), "money format")
    check(G.extract_number("The answer is 17.")[0] == G.gsm8k_gold("#### 17"), "answer is N")
    check(not G.numbers_equal(G.extract_number("Answer: 18")[0], G.gsm8k_gold("#### 17")), "mismatch detected")

    print(f"[self-test] code sandbox ({G.sandbox_method()})")
    he = T.load_jsonl("humaneval_30.jsonl")
    check(len(he) == 30, "humaneval has 30 items")

    def canon(it: dict[str, Any]) -> bool:
        return G.run_sandboxed(G.build_program(it["prompt"], it["canonical_solution"], it["test"], it["entry_point"]))["status"] == "passed"
    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = list(ex.map(canon, he))
    check(all(ok), f"canonical solutions pass ({sum(ok)}/{len(ok)}): failing {[he[i]['task_id'] for i, o in enumerate(ok) if not o]}")
    it = he[0]
    full = f"Here you go:\n```python\n{it['prompt']}{it['canonical_solution']}```\nDone."
    check(G.run_sandboxed(G.build_program(it["prompt"], G.extract_code(full), it["test"], it["entry_point"]))["status"] == "passed",
          "full-function completion passes")
    wrong = f"```python\ndef {it['entry_point']}(*a, **k):\n    return None\n```"
    check(G.run_sandboxed(G.build_program(it["prompt"], G.extract_code(wrong), it["test"], it["entry_point"]))["status"] == "failed",
          "wrong solution fails")
    net = G.run_sandboxed("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)\n")
    check(net["status"] == "failed" and ("PermissionError" in net["stderr_tail"] or "Network" in net["stderr_tail"]
                                         or "unreachable" in net["stderr_tail"].lower()), f"network blocked: {net}")
    to = G.run_sandboxed("while True:\n    pass\n", timeout_s=2)
    check(to["status"] == "timeout", f"timeout enforced: {to['status']}")
    sp = G.run_sandboxed("import subprocess\nsubprocess.run(['ls'])\n")
    check(sp["status"] == "failed", "subprocess blocked")

    print("[self-test] tool grading")
    for case in T.TOOL_CASES:
        exp = case["expect"]
        if exp["name"] is None:
            check(G.grade_tool_case([], exp)["correct"], f"{case['id']}: no-call correct")
            check(not G.grade_tool_case([{"name": "get_weather", "arguments": {}}], exp)["correct"], f"{case['id']}: call wrong")
            continue
        args_ = {k: ("Hello" if isinstance(v, dict) else v) for k, v in exp["args"].items()}
        msg = {"tool_calls": [{"id": "x", "type": "function", "function": {"name": exp["name"], "arguments": json.dumps(args_)}}]}
        check(G.grade_tool_case(G.server_tool_calls(msg), exp)["correct"], f"{case['id']}: server-format correct")
        xml = "<tool_call>\n<function=" + exp["name"] + ">\n" + "".join(
            f"<parameter={k}>\n{json.dumps(v) if isinstance(v, (dict, list)) else (str(v).lower() if isinstance(v, bool) else v)}\n</parameter>\n"
            for k, v in args_.items()) + "</function>\n</tool_call>"
        check(G.grade_tool_case(G.parse_xml_tool_calls(xml, case["tools"]), exp)["correct"], f"{case['id']}: xml-format correct")
        bad = dict(args_)
        k0 = next(k for k, v in exp["args"].items() if not isinstance(v, dict))
        bad[k0] = 999999 if isinstance(bad[k0], (int, float)) and not isinstance(bad[k0], bool) else ("zzz" if isinstance(bad[k0], str) else (not bad[k0]))
        check(not G.grade_tool_case([{"name": exp["name"], "arguments": bad}], exp)["correct"], f"{case['id']}: wrong arg detected")

    print("[self-test] images")
    a1, a2 = T.make_images(), T.make_images()
    check(all(a1[k] == a2[k] for k in a1), "image generation deterministic")
    check(all(c["image"] in a1 for c in T.IMAGE_CASES), "image cases reference existing images")
    check(G.short_answer_ok("Red.", "red", "word", ["blue"]) and not G.short_answer_ok("red or blue", "red", "word", ["blue"]), "word grader")
    check(G.short_answer_ok("D", "D", "letter") and not G.short_answer_ok("Bar C, not D", "D", "letter"), "letter grader")
    check(G.short_answer_ok("There are 7 circles.", "7", "number") and not G.short_answer_ok("8", "7", "number"), "number grader")

    print("[self-test] determinism diff")
    d = G.token_diff([1, 2, 3, 4], [1, 2, 9, 4, 5])
    check(d["prefix_match"] == 2 and abs(d["positionwise_match_rate"] - 3 / 5) < 1e-9 and not d["exact"], "token_diff")
    check(G.token_diff([1, 2], [1, 2])["exact"], "token_diff exact")

    print("[self-test] needle prompt")
    try:
        from benchmarks.textgen import ChatTokenizer, build_exact_prompt
        tok = ChatTokenizer(args.tokenizer)
        case = T.needle_case(4096, 0.5, 0)
        p = build_exact_prompt(tok, 4096, "text", case["seed"], REASONING_MODES["off"],
                               make_content=lambda b: T.insert_needle(b, case["needle"], 0.5) + "\n\n" + case["question"])
        content = p["messages"][0]["content"]
        check(p["exact"] and p["prompt_tokens"] == 4096, f"needle exact length ({p['prompt_tokens']})")
        check(case["needle"] in content and 0.3 < content.index(case["needle"]) / len(content) < 0.7, "needle placed mid-document")
        check(G.short_answer_ok(f"The code is {case['expected']}.", case["expected"], "code"), "code grader")
    except FileNotFoundError as e:
        print(f"  (skipped: {e})")
    print("[self-test]", "PASS" if not fails else f"FAILED ({len(fails)})")
    return 1 if fails else 0


# ============================================================================ dry run (mock server)
def dry_run(args: argparse.Namespace) -> int:
    from benchmarks import mock_server
    from benchmarks.textgen import resolve_tokenizer_dir
    tok_dir = resolve_tokenizer_dir(args.tokenizer)
    ref_dir = DRYRUN_DIR / "reference"
    fails: list[str] = []

    def check(c: bool, m: str) -> None:
        if not c:
            fails.append(m)
            print("  FAIL:", m)

    def run_once(noise: int, save: str | None) -> dict[str, Any]:
        margs = mock_server.build_parser().parse_args(["--port", "0", "--tokenizer", tok_dir, "--time-scale", "0.02",
                                                       "--noise-seed", str(noise)])
        srv = mock_server.make_server(margs)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            a = argparse.Namespace(**vars(args))
            a.label, a.limit, a.needle_lengths, a.tokenizer = f"dryrun-eval-n{noise}", 3, "4k,8k", tok_dir
            a.reference_dir, a.reference, a.save_reference, a.overwrite_reference = str(ref_dir), "dryrun", save, True
            ev = Evaluator(a, f"http://127.0.0.1:{srv.server_address[1]}")
            return ev.run(ALL_TASKS, make_run_id(a.label), DRYRUN_DIR, dry_run=True)
        finally:
            srv.shutdown()

    s1 = run_once(0, "dryrun")
    for t in ALL_TASKS:
        check(t in s1["tasks"] and s1["tasks"][t].get("n", 0) > 0, f"task {t} ran")
        check(all(not i.get("error") for i in s1["tasks"][t]["items"]), f"task {t} no request errors")
    check(set(s1["quality"]) == set(GATED_TASKS), f"quality keys {sorted(s1['quality'])}")
    check(s1["tasks"]["tools"]["items"][0]["parsed_by"] == "server", "mock tool call parsed from server tool_calls")
    check(all(i["exact"] for i in s1["tasks"]["needle"]["items"]), "needle prompts exact length")
    s2 = run_once(0, None)
    check(s2["determinism"]["token_match_rate_mean"] == 1.0 and s2["determinism"]["exact_match_frac"] == 1.0,
          f"same server -> identical greedy outputs ({s2['determinism']})")
    s3 = run_once(5, None)
    check(s3["determinism"]["token_match_rate_mean"] is not None and s3["determinism"]["token_match_rate_mean"] < 1.0,
          f"perturbed server -> divergence detected ({s3['determinism']})")
    check(json.loads(Path(s1["_path"]).read_text())["schema"] == SCHEMA, "summary file written")
    print("[dry-run]", "PASS" if not fails else f"FAILED ({len(fails)})")
    return 1 if fails else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("HD_BASE_URL", "http://127.0.0.1:8000"))
    p.add_argument("--label", default=os.environ.get("HD_LABEL"))
    p.add_argument("--framework", default=None)
    p.add_argument("--image-tag", default=os.environ.get("HD_IMAGE_TAG"))
    p.add_argument("--config", default=os.environ.get("HD_CONFIG"))
    p.add_argument("--notes", default=None)
    p.add_argument("--run-id", default=os.environ.get("HD_RUN_ID"))
    p.add_argument("--out-dir", default=None, help=f"default {RESULTS_DIR}")
    p.add_argument("--tasks", default=",".join(ALL_TASKS))
    p.add_argument("--reasoning", choices=list(REASONING_MODES), default="off")
    p.add_argument("--limit", type=int, default=None, help="max items per task (quick checks)")
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--code-timeout", type=float, default=15.0)
    p.add_argument("--max-tokens-scale", type=float, default=1.0)
    p.add_argument("--needle-lengths", default="4k,8k,16k,32k")
    p.add_argument("--reference", default="baseline", help="determinism reference name to diff against")
    p.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    p.add_argument("--save-reference", default=None, help="store this run's greedy outputs as reference NAME")
    p.add_argument("--overwrite-reference", action="store_true")
    p.add_argument("--no-logprobs", action="store_true")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--no-lock", action="store_true")
    p.add_argument("--lock-timeout", type=float, default=None)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.self_test:
        sys.exit(self_test(args))
    if args.dry_run:
        sys.exit(dry_run(args))
    if not args.label:
        p.error("--label is required (or set HD_LABEL)")
    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in names if t not in ALL_TASKS]
    if bad:
        p.error(f"unknown task(s) {bad}; choose from {ALL_TASKS}")
    run_id = args.run_id or make_run_id(args.label)
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    with gpu_lock(enabled=not args.no_lock, timeout_s=args.lock_timeout):
        Evaluator(args, args.base_url).run(names, run_id, out_dir)


if __name__ == "__main__":
    main()

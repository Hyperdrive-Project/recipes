#!/usr/bin/env python3
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Nemotron Hyperdrive workload runner for OpenAI-compatible servers (trtllm-serve, vLLM, SGLang).

For every (workload, concurrency) cell it writes ONE JSON record to benchmarks/results/<run_id>.jsonl:
  * prompts of EXACT input token length (model tokenizer + chat template, unique per request so prefix
    caching cannot short-circuit prefill), forced output length (ignore_eos + min_tokens + max_tokens;
    any request whose usage.completion_tokens != target is flagged and the cell marked invalid)
  * streamed timing: TTFT, TPOT = (e2e - ttft) / (out_tokens - 1), e2e, per-request decode tok/s,
    aggregate output tok/s, prefill tok/s = input_tokens / TTFT, P50/P95 over measured requests
  * accepted output tokens = usage.completion_tokens (drafted tokens are never counted)
  * spec-decode acceptance from /metrics deltas (vLLM Prometheus names, TRT-LLM JSON iteration stats,
    SGLang gauges; raw deltas always recorded)
  * GPU power (nvidia-smi ~10 Hz), MemAvailable / swap / pswpin / pswpout (1 Hz), energy (trapezoid),
    J per accepted output token, GPU temperature/clocks at start and end

Usage:
  tools-venv/bin/python benchmarks/bench.py --label trtllm-official --framework trtllm \
      --config configs/trtllm-spark-official.yaml --image-tag <tag> \
      --workloads standard --concurrency 1,2,4 --repeats 3 --warmups 1
  tools-venv/bin/python benchmarks/bench.py --dry-run        # CPU-only self-test against the mock server

Workload spec: [kind:]<input>-<output>, kind in {text, code, image}, input like 4k (=4096) or 3000.
Presets: standard (text 4k-512 8k-512 16k-1024 32k-1024 64k-2048), code (same sizes, code corpus),
         image (image:4k-512), all (standard + code + image).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import (REASONING_MODES, REPO_ROOT, MetricsScraper, append_jsonl, git_info,  # noqa: E402
                               gpu_lock, guess_framework, host_info, http_post_json, make_run_id,
                               server_identity, sha256_file, sha256_text, stream_completion, summarize,
                               utc_now_iso)

SCHEMA = "nemotron-hyperdrive/bench/v1"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DRYRUN_DIR = Path("/tmp/nemotron-hyperdrive-dryrun")
STANDARD_SIZES = ["4k-512", "8k-512", "16k-1024", "32k-1024", "64k-2048"]
PRESETS = {
    "standard": [f"text:{s}" for s in STANDARD_SIZES],
    "code": [f"code:{s}" for s in STANDARD_SIZES],
    "image": ["image:4k-512"],
}
PRESETS["all"] = PRESETS["standard"] + PRESETS["code"] + PRESETS["image"]


# ============================================================================ workloads
@dataclass(frozen=True)
class Workload:
    kind: str           # text | code | image
    input_tokens: int
    output_tokens: int
    in_label: str

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.in_label}-{self.output_tokens}"


def parse_size(s: str) -> tuple[int, str]:
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024), s
    return int(s), s


def parse_workloads(spec: str) -> list[Workload]:
    """Comma-separated workload specs and/or preset names."""
    out: list[Workload] = []
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        if item in PRESETS:
            out.extend(parse_workloads(",".join(PRESETS[item])))
            continue
        kind = "text"
        if ":" in item:
            kind, item = item.split(":", 1)
        if kind not in ("text", "code", "image"):
            raise ValueError(f"unknown workload kind {kind!r}")
        sep = "-" if "-" in item else "/"
        a, b = item.split(sep, 1)
        n_in, lbl = parse_size(a)
        out.append(Workload(kind, n_in, parse_size(b)[0], lbl))
    seen, uniq = set(), []
    for w in out:
        if w.name not in seen:
            seen.add(w.name)
            uniq.append(w)
    return uniq


def bench_image_png(seed: int, w: int = 1024, h: int = 768) -> bytes:
    """Deterministic bar-chart PNG used by image workloads."""
    from PIL import Image, ImageDraw, ImageFont  # lazy: only image workloads need pillow
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=28)
    n = 8
    vals = [rng.randint(5, 100) for _ in range(n)]
    bw = (w - 160) // n
    for i, v in enumerate(vals):
        x0 = 100 + i * bw
        top = h - 80 - int(v / 100 * (h - 200))
        d.rectangle([x0 + 8, top, x0 + bw - 8, h - 80],
                    fill=(rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220)))
        d.text((x0 + bw // 2 - 10, h - 70), chr(65 + i), fill="black", font=font)
        d.text((x0 + bw // 2 - 18, top - 34), str(v), fill="black", font=font)
    d.line([90, h - 80, w - 40, h - 80], fill="black", width=3)
    d.line([90, 60, 90, h - 80], fill="black", width=3)
    d.text((100, 15), f"Quarterly volume, series {seed % 997}", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_token_estimate(w: int, h: int) -> int:
    """Rough visual-token estimate for the RADIO encoder (16px patches, 2x2 downsample, 1024..13312
    patches per preprocessor_config.json). Only used to size the text part; the server's usage is recorded."""
    patches = min(max((w // 16) * (h // 16), 1024), 13312)
    return patches // 4 + 8


# ============================================================================ system sampling
def gpu_query() -> dict[str, Any]:
    """GPU temperature / clocks / power snapshot via nvidia-smi (tolerant of unsupported fields)."""
    out: dict[str, Any] = {"t_utc": utc_now_iso()}
    groups = [("temperature.gpu", "clocks.gr", "clocks.sm", "power.draw"),
              ("clocks.max.gr",), ("clocks_event_reasons.active",), ("pstate",)]
    for fields in groups:
        try:
            r = subprocess.run(["nvidia-smi", "-i", "0", f"--query-gpu={','.join(fields)}",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                continue
            vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
            for f, v in zip(fields, vals):
                try:
                    out[f] = float(v)
                except ValueError:
                    out[f] = None if v.startswith("[") else v
        except Exception:
            break
    return out


def read_meminfo() -> dict[str, int]:
    """Selected /proc/meminfo (kB) and /proc/vmstat counters."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "MemFree", "Cached"):
                    out[k] = int(v.split()[0])
        with open("/proc/vmstat") as f:
            for line in f:
                k, v = line.split()
                if k in ("pswpin", "pswpout"):
                    out[k] = int(v)
    except OSError:
        pass
    return out


class SystemSampler:
    """Background sampling: GPU power via `nvidia-smi -lms` (~10 Hz), memory/swap at 1 Hz, optional
    callback (e.g. draining TRT-LLM iteration stats) at 1 Hz. Times are time.monotonic()."""

    def __init__(self, power: bool = True, power_interval_ms: int = 100, mem_interval_s: float = 1.0):
        self.power_enabled = power
        self.power_interval_ms = power_interval_ms
        self.mem_interval_s = mem_interval_s
        self.power: list[tuple[float, float]] = []
        self.mem: list[tuple[float, dict[str, int]]] = []
        self.callback: Callable[[], None] | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._proc: subprocess.Popen[str] | None = None
        self.notes: list[str] = []

    def start(self) -> None:
        if self.power_enabled:
            try:
                self._proc = subprocess.Popen(
                    ["nvidia-smi", "-i", "0", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                     f"-lms={self.power_interval_ms}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                t = threading.Thread(target=self._power_reader, daemon=True)
                t.start()
                self._threads.append(t)
            except (OSError, FileNotFoundError) as e:
                self.notes.append(f"power sampling unavailable: {e}")
        t = threading.Thread(target=self._mem_loop, daemon=True)
        t.start()
        self._threads.append(t)

    def _power_reader(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            try:
                self.power.append((time.monotonic(), float(line.strip().split(",")[0])))
            except ValueError:
                continue

    def _mem_loop(self) -> None:
        while not self._stop.is_set():
            self.mem.append((time.monotonic(), read_meminfo()))
            cb = self.callback
            if cb:
                try:
                    cb()
                except Exception:
                    pass
            self._stop.wait(self.mem_interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._proc:  # our own sampling child only
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        for t in self._threads:
            t.join(timeout=3)
        self.mem.append((time.monotonic(), read_meminfo()))


def integrate_energy(samples: list[tuple[float, float]], t0: float, t1: float) -> float | None:
    """Trapezoid integral of power (W) over [t0, t1] in joules, linearly interpolating the endpoints."""
    pts = sorted(samples)
    if len(pts) < 2 or t1 <= t0:
        return None

    def interp(t: float) -> float:
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                return pa + (pb - pa) * (t - ta) / (tb - ta) if tb > ta else pa
        return pts[-1][1]

    inner = [(t, p) for t, p in pts if t0 < t < t1]
    seq = [(t0, interp(t0)), *inner, (t1, interp(t1))]
    return sum((tb - ta) * (pa + pb) / 2.0 for (ta, pa), (tb, pb) in zip(seq, seq[1:]))


def power_summary(s: SystemSampler, t_idle0: float, t0: float, t1: float, out_tokens: int) -> dict[str, Any]:
    win = [p for t, p in s.power if t0 <= t <= t1]
    idle = [p for t, p in s.power if t_idle0 <= t < t0]
    energy = integrate_energy(s.power, t0, t1)
    idle_w = sum(idle) / len(idle) if idle else None
    dur = t1 - t0
    out: dict[str, Any] = {
        "source": "nvidia-smi power.draw (GB10 GPU rail as reported by NVML)", "samples": len(win),
        "sample_hz_effective": len(win) / dur if dur > 0 else None, "idle_power_w": idle_w,
        "idle_samples": len(idle), "mean_power_w": sum(win) / len(win) if win else None,
        "peak_power_w": max(win) if win else None, "energy_j": energy,
        "energy_above_idle_j": (energy - idle_w * dur) if (energy is not None and idle_w is not None) else None,
        "j_per_output_token": (energy / out_tokens) if (energy is not None and out_tokens) else None,
        "notes": s.notes,
    }
    if out["energy_above_idle_j"] is not None and out_tokens:
        out["j_per_output_token_above_idle"] = out["energy_above_idle_j"] / out_tokens
    return out


def memory_summary(s: SystemSampler, t_idle0: float, t0: float, t1: float) -> dict[str, Any]:
    win = [m for t, m in s.mem if t0 - 1.0 <= t <= t1 + 1.0 and "MemAvailable" in m]
    pre = [m for t, m in s.mem if t_idle0 <= t < t0 and "MemAvailable" in m]
    if not win:
        return {"samples": 0}
    total = win[0]["MemTotal"]
    used = [m["MemTotal"] - m["MemAvailable"] for m in win]
    swap = [m.get("SwapTotal", 0) - m.get("SwapFree", 0) for m in win]
    kb = 1024.0
    first, last = win[0], win[-1]
    return {
        "source": "/proc/meminfo + /proc/vmstat (unified memory: MemTotal - MemAvailable)", "samples": len(win),
        "mem_total_mb": total / kb, "peak_used_mb": max(used) / kb, "min_available_mb": min(m["MemAvailable"] for m in win) / kb,
        "pre_run_used_mb": ((pre[-1]["MemTotal"] - pre[-1]["MemAvailable"]) / kb) if pre else None,
        "swap_used_start_mb": swap[0] / kb, "swap_used_end_mb": swap[-1] / kb, "swap_peak_mb": max(swap) / kb,
        "swap_delta_mb": (swap[-1] - swap[0]) / kb,
        "pswpin_delta": last.get("pswpin", 0) - first.get("pswpin", 0),
        "pswpout_delta": last.get("pswpout", 0) - first.get("pswpout", 0),
    }


# ============================================================================ requests
class Runner:
    """Holds server/tokenizer context and runs cells."""

    def __init__(self, args: argparse.Namespace, base_url: str, out_path: Path, dry_run: bool = False):
        self.args = args
        self.base_url = base_url.rstrip("/")
        self.out_path = out_path
        self.dry_run = dry_run
        from benchmarks.textgen import TEXTGEN_VERSION, ChatTokenizer
        self.textgen_version = TEXTGEN_VERSION
        self.tok = ChatTokenizer(args.tokenizer)
        self.ident = server_identity(self.base_url)
        if "model_id" not in self.ident:
            raise SystemExit(f"server at {self.base_url} not reachable: {self.ident.get('error')}")
        self.model = self.ident["model_id"]
        self.framework = args.framework or guess_framework(self.ident)
        self.tkw = dict(REASONING_MODES[args.reasoning])
        self.fields = {"ignore_eos": not args.no_ignore_eos, "min_tokens": not args.no_min_tokens}
        self.dropped_fields: list[str] = []
        self.extra_body = json.loads(args.extra_body) if args.extra_body else {}
        self.scraper = MetricsScraper(self.base_url, enabled=not args.no_metrics)
        self.git = git_info()
        self.host = host_info()
        cfg: dict[str, Any] = {"path": args.config}
        if args.config:
            cfg["sha256"] = sha256_file(args.config)
            try:
                txt = Path(args.config).read_text()
                cfg["text"] = txt if len(txt) < 100_000 else None
            except OSError:
                cfg["text"] = None
        self.config_meta = cfg

    # ------------------------------------------------------------------ prompt + body
    def make_prompt(self, wl: Workload, seed: int, nonce: str) -> dict[str, Any]:
        from benchmarks.textgen import build_exact_prompt
        extra, reserve = None, 0
        if wl.kind == "image":
            png = bench_image_png(seed)
            extra = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                         base64.b64encode(png).decode()}}]
            reserve = self.args.image_token_estimate or image_token_estimate(1024, 768)
        corpus_kind = "code" if wl.kind == "code" else "text"
        question = None
        if wl.kind == "image":
            question = ("First describe the chart image in detail (every bar, label and value), then write a "
                        "detailed report on the archive documents above, covering each document in order. "
                        "Be thorough and keep writing until the report is complete.")
        p = build_exact_prompt(self.tok, wl.input_tokens, corpus_kind, seed, self.tkw, question=question,
                               header=f"Session {nonce}.\n\n", extra_parts=extra, reserve_tokens=reserve)
        p["seed"] = seed
        p["image_reserve_tokens"] = reserve
        return p

    def body(self, prompt: dict[str, Any], out_tokens: int, stream: bool = True) -> tuple[str, dict[str, Any]]:
        a = self.args
        b: dict[str, Any] = {"model": self.model, "max_tokens": out_tokens, "temperature": a.temperature,
                             "stream": stream}
        if stream:
            b["stream_options"] = {"include_usage": True}
        if a.top_p is not None:
            b["top_p"] = a.top_p
        if a.seed is not None:
            b["seed"] = a.seed
        if self.fields["ignore_eos"]:
            b["ignore_eos"] = True
        if self.fields["min_tokens"]:
            b["min_tokens"] = out_tokens
        if a.api == "completions" and prompt.get("prompt_token_ids"):
            b["prompt"] = prompt["prompt_token_ids"]
            path = "/v1/completions"
        else:
            b["messages"] = prompt["messages"]
            b["chat_template_kwargs"] = self.tkw
            path = "/v1/chat/completions"
        b.update(self.extra_body)
        return path, b

    def probe_fields(self) -> None:
        """Send one tiny request; drop ignore_eos/min_tokens if the server rejects them (recorded)."""
        prompt = {"messages": [{"role": "user", "content": "Say hello."}],
                  "prompt_token_ids": self.tok.chat_ids([{"role": "user", "content": "Say hello."}], self.tkw)}
        for _ in range(3):
            path, b = self.body(prompt, 4, stream=False)
            try:
                http_post_json(self.base_url + path, b, timeout=600)
                return
            except Exception as e:
                msg = str(e).lower()
                hit = [f for f in ("min_tokens", "ignore_eos") if f in msg and self.fields[f]]
                if not hit:
                    print(f"[bench] probe request failed: {e}", file=sys.stderr)
                    return
                for f in hit:
                    self.fields[f] = False
                    self.dropped_fields.append(f)
                    print(f"[bench] server rejected '{f}', dropping it (recorded)", file=sys.stderr)

    # ------------------------------------------------------------------ phases
    def run_phase(self, prompts: list[dict[str, Any]], wl: Workload, c: int, t_ref: float) -> list[dict[str, Any]]:
        q: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue()
        for i, p in enumerate(prompts):
            q.put((i, p))
        results: list[dict[str, Any]] = []
        lock = threading.Lock()

        def worker(wid: int) -> None:
            while True:
                try:
                    i, p = q.get_nowait()
                except queue.Empty:
                    return
                path, b = self.body(p, wl.output_tokens)
                r = stream_completion(self.base_url, path, b, timeout=self.args.request_timeout)
                rec = self.request_record(i, wid, p, r, wl, t_ref)
                with lock:
                    results.append(rec)

        threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(c)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return sorted(results, key=lambda r: r["idx"])

    def request_record(self, i: int, wid: int, p: dict[str, Any], r: Any, wl: Workload, t_ref: float) -> dict[str, Any]:
        usage = r.usage or {}
        ct = usage.get("completion_tokens")
        src = "usage"
        if ct is None and r.ok:
            ct = len(self.tok.encode(r.reasoning + r.content))
            src = "retokenized-text (usage missing)"
        ttft, e2e = r.ttft_s, r.e2e_s
        decode_s = (e2e - ttft) if (ttft is not None and e2e is not None) else None
        tpot = (decode_s / (ct - 1)) if (decode_s is not None and ct and ct > 1) else None
        gaps = sorted(r.chunk_gaps_s)
        from benchmarks.common import percentile
        return {
            "idx": i, "worker": wid, "seed": p["seed"], "ok": r.ok, "error": r.error, "http_status": r.http_status,
            "prompt_tokens_target": wl.input_tokens, "prompt_tokens_local": p["prompt_tokens"],
            "prompt_tokens_server": usage.get("prompt_tokens"), "prompt_exact_local": p["exact"],
            "completion_tokens": ct, "completion_tokens_source": src if ct is not None else None,
            "output_honored": ct == wl.output_tokens if ct is not None else False,
            "finish_reason": r.finish_reason, "ttft_s": ttft, "e2e_s": e2e, "tpot_s": tpot,
            "decode_tok_s": (1.0 / tpot) if tpot else None,
            "e2e_output_tok_s": (ct / e2e) if (ct and e2e) else None,
            "prefill_tok_s": ((usage.get("prompt_tokens") or p["prompt_tokens"]) / ttft) if ttft else None,
            "n_chunks": r.n_chunks, "tokens_per_chunk": (ct / r.n_chunks) if (ct and r.n_chunks) else None,
            "chunk_gap_p50_s": percentile(gaps, 50), "chunk_gap_p95_s": percentile(gaps, 95),
            "chunk_gap_max_s": gaps[-1] if gaps else None,
            "t_start_rel_s": r.t_start_mono - t_ref, "t_end_rel_s": (r.t_start_mono - t_ref) + (e2e or 0.0),
            "content_chars": len(r.content), "reasoning_chars": len(r.reasoning),
            "output_head": (r.reasoning + r.content)[:160], "prompt_sha256": p["sha256"],
        }

    # ------------------------------------------------------------------ cell
    def run_cell(self, wl: Workload, c: int, run_id: str, label: str) -> dict[str, Any]:
        a = self.args
        cell_start = utc_now_iso()
        n_warm, n_meas = a.warmups * c, a.repeats * c
        base_seed = int(sha256_text(f"{a.seed_base}|{wl.name}|{c}")[:8], 16)
        t_build = time.monotonic()
        prompts = [self.make_prompt(wl, base_seed + i, f"{run_id[-4:]}-{wl.name}-c{c}-{i}")
                   for i in range(n_warm + n_meas)]
        build_s = time.monotonic() - t_build
        print(f"[bench] {label} {wl.name} c={c}: built {len(prompts)} prompts in {build_s:.1f}s "
              f"(exact={sum(p['exact'] for p in prompts)}/{len(prompts)})", flush=True)
        if n_warm:
            warm = self.run_phase(prompts[:n_warm], wl, c, time.monotonic())
            bad = [w for w in warm if not w["ok"]]
            if bad:
                print(f"[bench]   warmup errors: {bad[0]['error']}", file=sys.stderr)

        sampler = SystemSampler(power=not a.no_power)
        sampler.start()
        t_idle0 = time.monotonic()
        time.sleep(a.idle_sample_s)
        self.scraper.reset_json()
        before = self.scraper.snapshot()
        sampler.callback = self.scraper.poll if self.scraper.json_endpoints else None
        thermal_start = gpu_query() if not a.no_power else {}
        measure_start = utc_now_iso()
        t0 = time.monotonic()
        reqs = self.run_phase(prompts[n_warm:], wl, c, t0)
        t1 = time.monotonic()
        measure_end = utc_now_iso()
        sampler.callback = None
        self.scraper.poll()
        after = self.scraper.snapshot()
        thermal_end = gpu_query() if not a.no_power else {}
        time.sleep(0.3)
        sampler.stop()

        ok = [r for r in reqs if r["ok"]]
        total_out = sum(r["completion_tokens"] or 0 for r in ok)
        starts = [r["t_start_rel_s"] for r in ok]
        ends = [r["t_end_rel_s"] for r in ok]
        wall = (max(ends) - min(starts)) if ok else None
        honored = [r["output_honored"] for r in reqs]
        srv_pt = [r["prompt_tokens_server"] for r in ok if r["prompt_tokens_server"] is not None]
        exact_srv = [pt == wl.input_tokens for pt in srv_pt]
        summary = {
            "requests_total": len(reqs), "requests_ok": len(ok), "requests_failed": len(reqs) - len(ok),
            "total_output_tokens": total_out, "measure_wall_s": wall,
            "ttft_s": summarize(r["ttft_s"] for r in ok), "tpot_s": summarize(r["tpot_s"] for r in ok),
            "e2e_s": summarize(r["e2e_s"] for r in ok), "decode_tok_s": summarize(r["decode_tok_s"] for r in ok),
            "e2e_output_tok_s": summarize(r["e2e_output_tok_s"] for r in ok),
            "prefill_tok_s": summarize(r["prefill_tok_s"] for r in ok),
            "output_tok_s_aggregate": (total_out / wall) if wall else None,
            "output_len_honored_frac": (sum(honored) / len(honored)) if honored else None,
            "input_len_exact_local_frac": sum(p["exact"] for p in prompts[n_warm:]) / max(1, n_meas),
            "input_len_server_match_frac": (sum(exact_srv) / len(exact_srv)) if exact_srv else None,
            "server_prompt_tokens": summarize(srv_pt),
            "tokens_per_chunk": summarize(r["tokens_per_chunk"] for r in ok),
        }
        spec = self.scraper.spec_summary(before, after)
        spec["client_tokens_per_chunk_mean"] = summary["tokens_per_chunk"]["mean"]
        spec["client_tokens_per_chunk_note"] = ("client-side estimate only: tokens/stream-chunk ~ acceptance length "
                                                "when the server flushes once per decode step")
        invalid, warnings = [], []
        if len(ok) < len(reqs):
            invalid.append(f"{len(reqs) - len(ok)} request(s) failed: {next(r['error'] for r in reqs if not r['ok'])}")
        if not all(honored):
            invalid.append(f"output length not honored in {honored.count(False)}/{len(honored)} requests "
                           f"(dropped fields: {self.dropped_fields or 'none'})")
        if wl.kind != "image" and summary["input_len_exact_local_frac"] < 1.0:
            warnings.append("local prompt length not exact for some requests")
        if exact_srv and not all(exact_srv) and wl.kind != "image":
            warnings.append(f"server prompt_tokens != target for {exact_srv.count(False)}/{len(exact_srv)} requests "
                            f"(template/tokenizer mismatch?)")
        if any(r["completion_tokens_source"] and r["completion_tokens_source"] != "usage" for r in ok):
            warnings.append("usage missing from stream; completion tokens re-tokenized locally")
        if wl.kind == "image":
            warnings.append("image workload: input length not exact (visual tokens counted by the server)")
        rec = {
            "schema": SCHEMA, "run_id": run_id, "label": label, "framework": self.framework,
            "image_tag": a.image_tag, "config": self.config_meta, "notes": a.notes, "git": self.git,
            "host": self.host, "server": self.ident, "dry_run": self.dry_run,
            "request_params": {"api": a.api, "temperature": a.temperature, "top_p": a.top_p, "seed": a.seed,
                               "reasoning_mode": a.reasoning, "chat_template_kwargs": self.tkw,
                               "ignore_eos": self.fields["ignore_eos"], "min_tokens": self.fields["min_tokens"],
                               "dropped_fields": self.dropped_fields, "extra_body": self.extra_body,
                               "unique_prompt_per_request": True},
            "textgen_version": self.textgen_version, "tokenizer_path": self.tok.path,
            "workload": {"name": wl.name, "kind": wl.kind, "input_tokens": wl.input_tokens,
                         "output_tokens": wl.output_tokens},
            "concurrency": c, "repeats": a.repeats, "warmups": a.warmups,
            "timestamps": {"cell_start_utc": cell_start, "measure_start_utc": measure_start,
                           "measure_end_utc": measure_end, "cell_end_utc": utc_now_iso(),
                           "prompt_build_s": build_s, "idle_sample_s": a.idle_sample_s},
            "thermal": {"start": thermal_start, "end": thermal_end},
            "summary": summary, "spec_decode": spec,
            "power": power_summary(sampler, t_idle0, t0, t1, total_out) if not a.no_power else None,
            "memory": memory_summary(sampler, t_idle0, t0, t1),
            "valid": not invalid, "invalid_reasons": invalid, "warnings": warnings, "requests": reqs,
        }
        append_jsonl(self.out_path, rec)
        s = summary
        print(f"[bench]   ok={s['requests_ok']}/{s['requests_total']} ttft_p50={_fmt(s['ttft_s']['p50'])}s "
              f"tpot_p50={_fmt(s['tpot_s']['p50'], 4)}s decode_p50={_fmt(s['decode_tok_s']['p50'])} tok/s "
              f"agg={_fmt(s['output_tok_s_aggregate'])} tok/s accept={_fmt(spec['acceptance_rate'], 3)} "
              f"valid={rec['valid']}", flush=True)
        return rec


def _fmt(v: Any, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


# ============================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("HD_BASE_URL", "http://127.0.0.1:8000"))
    p.add_argument("--label", default=os.environ.get("HD_LABEL"), help="config label (required unless --dry-run)")
    p.add_argument("--framework", default=None, help="trtllm | vllm | sglang (default: guess from /v1/models)")
    p.add_argument("--image-tag", default=os.environ.get("HD_IMAGE_TAG"), help="container image tag")
    p.add_argument("--config", default=os.environ.get("HD_CONFIG"), help="server config file (hashed + embedded)")
    p.add_argument("--notes", default=None)
    p.add_argument("--run-id", default=os.environ.get("HD_RUN_ID"))
    p.add_argument("--out-dir", default=None, help=f"default {RESULTS_DIR}")
    p.add_argument("--workloads", default="standard", help="presets/specs, e.g. standard,code:16k-1024")
    p.add_argument("--concurrency", default="1,2,4")
    p.add_argument("--repeats", type=int, default=3, help="measured requests per concurrency slot")
    p.add_argument("--warmups", type=int, default=1, help="warmup requests per concurrency slot (excluded)")
    p.add_argument("--reasoning", choices=list(REASONING_MODES), default="off")
    p.add_argument("--api", choices=["chat", "completions"], default="chat",
                   help="completions sends the templated prompt as token ids (exact length guaranteed)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=None, help="sampling seed sent to the server")
    p.add_argument("--seed-base", type=int, default=20261020, help="prompt-corpus seed base")
    p.add_argument("--extra-body", default=None, help="JSON merged into every request body")
    p.add_argument("--no-ignore-eos", action="store_true")
    p.add_argument("--no-min-tokens", action="store_true")
    p.add_argument("--no-metrics", action="store_true")
    p.add_argument("--no-power", action="store_true")
    p.add_argument("--tokenizer", default=None, help="tokenizer dir (default models/super-vl-nvfp4)")
    p.add_argument("--image-token-estimate", type=int, default=None)
    p.add_argument("--idle-sample-s", type=float, default=2.0, help="idle power window before each cell")
    p.add_argument("--cooldown-s", type=float, default=0.0, help="sleep between cells")
    p.add_argument("--request-timeout", type=float, default=3600.0)
    p.add_argument("--no-lock", action="store_true", help="do not take the global GPU lock")
    p.add_argument("--lock-timeout", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="validate the whole pipeline against the mock server")
    p.add_argument("--list-workloads", action="store_true")
    return p


def run(args: argparse.Namespace, base_url: str, out_dir: Path, dry_run: bool = False) -> tuple[Path, list[dict[str, Any]]]:
    label = args.label
    run_id = args.run_id or make_run_id(label)
    out_path = out_dir / f"{run_id}.jsonl"
    runner = Runner(args, base_url, out_path, dry_run=dry_run)
    runner.probe_fields()
    workloads = parse_workloads(args.workloads)
    conc = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
    max_len = runner.ident.get("max_model_len")
    print(f"[bench] run_id={run_id} label={label} framework={runner.framework} model={runner.model} "
          f"workloads={[w.name for w in workloads]} concurrency={conc} -> {out_path}", flush=True)
    recs = []
    for wl in workloads:
        if max_len and wl.input_tokens + wl.output_tokens > max_len:
            print(f"[bench] skip {wl.name}: exceeds server max_model_len {max_len}", flush=True)
            continue
        for c in conc:
            recs.append(runner.run_cell(wl, c, run_id, label))
            if args.cooldown_s:
                time.sleep(args.cooldown_s)
    return out_path, recs


# ============================================================================ dry run
def dry_run(args: argparse.Namespace) -> int:
    """Start the mock server in-process and validate records end to end (CPU only, no GPU lock)."""
    from benchmarks import mock_server
    from benchmarks.textgen import resolve_tokenizer_dir
    tok_dir = resolve_tokenizer_dir(args.tokenizer)
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)
            print(f"  FAIL: {msg}")

    scenarios = [
        ("vllm", [], "text:4k-64,code:8k-96", "1,2", True),
        ("trtllm", [], "text:2k-48", "2", True),
        ("sglang", ["--draft-len", "0"], "text:1k-32", "1", True),
        ("none", ["--no-ignore-eos", "--natural-max", "20"], "text:1k-40", "1", False),
    ]
    if not args.no_image_dry:
        scenarios.append(("vllm", [], "image:2k-32", "1", True))
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    for flavor, extra, wls, conc, expect_valid in scenarios:
        margs = mock_server.build_parser().parse_args(
            ["--port", "0", "--flavor", flavor, "--tokenizer", tok_dir, "--prefill-tok-s", "200000",
             "--step-ms", "4", "--accept-prob", "0.6", "--draft-len", "3", *extra])
        srv = mock_server.make_server(margs)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        st = srv.state  # type: ignore[attr-defined]
        print(f"[dry-run] mock flavor={flavor} {wls} c={conc} at {url}")
        a = argparse.Namespace(**vars(args))
        a.label, a.framework, a.run_id = f"dryrun-{flavor}", None, None
        a.workloads, a.concurrency, a.repeats, a.warmups = wls, conc, 2, 1
        a.idle_sample_s, a.tokenizer = 0.5, tok_dir
        try:
            path, recs = run(a, url, DRYRUN_DIR, dry_run=True)
        finally:
            srv.shutdown()
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        check(len(lines) == len(recs) and len(recs) > 0, f"{flavor}: jsonl record count")
        for r in recs:
            tag = f"{flavor}/{r['workload']['name']}/c{r['concurrency']}"
            s = r["summary"]
            for k in ("schema", "run_id", "label", "framework", "git", "timestamps", "thermal", "summary",
                      "spec_decode", "power", "memory", "requests", "valid"):
                check(k in r, f"{tag}: missing key {k}")
            check(r["dry_run"] is True, f"{tag}: dry_run flag")
            check(s["requests_ok"] == s["requests_total"] == 2 * r["concurrency"], f"{tag}: request count")
            check(r["valid"] == expect_valid, f"{tag}: valid={r['valid']} expected {expect_valid} {r['invalid_reasons']}")
            check(s["ttft_s"]["p50"] is not None and s["ttft_s"]["p50"] > 0, f"{tag}: ttft")
            if expect_valid:
                check(s["output_len_honored_frac"] == 1.0, f"{tag}: output honored")
                check(s["total_output_tokens"] == s["requests_ok"] * r["workload"]["output_tokens"], f"{tag}: token total")
                check(s["tpot_s"]["p50"] is not None and s["decode_tok_s"]["p50"] > 0, f"{tag}: tpot/decode")
                for q in r["requests"]:
                    want = (q["e2e_s"] - q["ttft_s"]) / (q["completion_tokens"] - 1)
                    check(abs(q["tpot_s"] - want) < 1e-9, f"{tag}: tpot formula")
                if r["workload"]["kind"] != "image":
                    check(s["input_len_server_match_frac"] == 1.0, f"{tag}: server prompt_tokens == target")
            else:
                check(s["output_len_honored_frac"] < 1.0, f"{tag}: not-honored detection")
            sp = r["spec_decode"]
            if flavor in ("vllm", "trtllm"):
                check(sp["source"] is not None and sp["acceptance_rate"] is not None, f"{tag}: spec source {sp['source']}")
                if sp["acceptance_rate"] is not None:
                    check(0.0 < sp["acceptance_rate"] < 1.0, f"{tag}: acceptance in (0,1)")
                    check(abs(sp["mean_acceptance_length"] - 1 - sp["accepted_tokens"] / sp["drafts"]) < 1e-9,
                          f"{tag}: acceptance length consistency")
                    steps = sp["drafts"]
                    check(sp["accepted_tokens"] + steps == s["total_output_tokens"] or flavor == "trtllm"
                          or sp["accepted_tokens"] + steps >= s["total_output_tokens"],
                          f"{tag}: accepted+bonus covers output tokens")
            if flavor == "sglang":
                check(sp["drafts"] in (None, 0) and sp["acceptance_rate"] is None, f"{tag}: no spec when draft_len=0")
            m = r["memory"]
            check(m.get("samples", 0) >= 1 and m.get("peak_used_mb", 0) > 0, f"{tag}: memory samples")
            pw = r["power"]
            if pw and pw["samples"]:
                check(pw["energy_j"] is not None and pw["energy_j"] > 0, f"{tag}: energy integration")
                check(pw["j_per_output_token"] is not None, f"{tag}: J/token")
        # mock-side cross-check of spec counters for vllm flavor (whole server lifetime incl. warmup)
        if flavor == "vllm" and recs:
            check(st.counters["drafts"] > 0, "vllm: mock counted drafts")
    # unit check: trapezoid integration
    e = integrate_energy([(0.0, 10.0), (1.0, 20.0), (2.0, 20.0)], 0.5, 2.0)
    check(e is not None and abs(e - (0.5 * (15 + 20) / 2 + 20.0)) < 1e-9, "energy trapezoid unit test")
    print(f"[dry-run] results in {DRYRUN_DIR}")
    if failures:
        print(f"[dry-run] FAILED ({len(failures)} checks)")
        return 1
    print("[dry-run] PASS")
    return 0


def main() -> None:
    p = build_parser()
    p.add_argument("--no-image-dry", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.list_workloads:
        for k, v in PRESETS.items():
            print(f"{k}: {', '.join(w.name for w in parse_workloads(','.join(v)))}")
        return
    if args.dry_run:
        sys.exit(dry_run(args))
    if not args.label:
        p.error("--label is required (or set HD_LABEL)")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    with gpu_lock(enabled=not args.no_lock, timeout_s=args.lock_timeout):
        path, recs = run(args, args.base_url, out_dir)
    n_valid = sum(r["valid"] for r in recs)
    print(f"[bench] wrote {len(recs)} record(s) ({n_valid} valid) to {path}")


if __name__ == "__main__":
    main()

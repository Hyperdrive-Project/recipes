# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Shared helpers for Nemotron Hyperdrive tooling (standard library only).

Contents:
  * repo / git / host metadata
  * percentile summaries
  * a tiny OpenAI-compatible HTTP client with SSE streaming + timing
  * Prometheus text parsing and tolerant spec-decode metric extraction
  * the global GPU lock shared by bench.py, eval.py and autotune/lab.py
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_LOCK_PATH = "/tmp/nemotron-hyperdrive-gpu.lock"
GPU_LOCK_ENV = "HD_GPU_LOCK_HELD"
MODEL_REPO_ID = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
MODEL_REVISION = "59b8721646c38111a242da18218859fb2b454322"

# chat_template.jinja exposes `enable_thinking` (default True) and `medium_effort` (default False).
REASONING_MODES: dict[str, dict[str, bool]] = {
    "off": {"enable_thinking": False},
    "medium": {"enable_thinking": True, "medium_effort": True},
    "on": {"enable_thinking": True},
}


# --------------------------------------------------------------------------- metadata
def utc_now_iso() -> str:
    """Current UTC time, ISO-8601 with 'Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_stamp() -> str:
    """Compact UTC stamp for run ids."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize(s: str, maxlen: int = 64) -> str:
    """Make a string safe for file names."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    return s[:maxlen] or "x"


def make_run_id(label: str) -> str:
    """Run id: <utc stamp>-<label>-<4 hex>."""
    return f"{utc_stamp()}-{sanitize(label, 40)}-{os.urandom(2).hex()}"


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit hash and dirty flag of the project repo (None if not a git repo)."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None,
            "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


def sha256_file(path: str | Path) -> str | None:
    """sha256 of a file, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def host_info() -> dict[str, Any]:
    """Basic host description."""
    return {"hostname": socket.gethostname(), "kernel": platform.release(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version()}


# --------------------------------------------------------------------------- statistics
def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy 'linear' method). q in [0, 100]."""
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: Iterable[float | None]) -> dict[str, Any]:
    """n / mean / p50 / p95 / min / max / std of the non-None values (all None if empty)."""
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None, "std": None}
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
    return {"n": len(xs), "mean": mean, "p50": percentile(xs, 50), "p95": percentile(xs, 95),
            "min": min(xs), "max": max(xs), "std": std}


# --------------------------------------------------------------------------- GPU lock
@contextlib.contextmanager
def gpu_lock(enabled: bool = True, timeout_s: float | None = None, path: str = GPU_LOCK_PATH,
             announce: bool = True) -> Iterator[bool]:
    """Hold the global GPU lock (flock(2) on /tmp/nemotron-hyperdrive-gpu.lock).

    Compatible with `flock /tmp/nemotron-hyperdrive-gpu.lock <cmd>`. Re-entrant across processes via
    the HD_GPU_LOCK_HELD=1 environment variable: a child launched by a lock holder (e.g. bench.py run
    by autotune/lab.py) skips acquisition instead of deadlocking. Yields True if this call took the lock.
    """
    if not enabled or os.environ.get(GPU_LOCK_ENV) == "1":
        yield False
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        waited = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if announce and not waited:
                    print(f"[gpu-lock] waiting for {path} ...", flush=True)
                    waited = True
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"could not acquire GPU lock {path} within {timeout_s}s")
                time.sleep(1.0)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()} {utc_now_iso()}\n".encode())
        prev = os.environ.get(GPU_LOCK_ENV)
        os.environ[GPU_LOCK_ENV] = "1"
        try:
            yield True
        finally:
            if prev is None:
                os.environ.pop(GPU_LOCK_ENV, None)
            else:
                os.environ[GPU_LOCK_ENV] = prev
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------- HTTP client
class HTTPError(RuntimeError):
    """Non-2xx response, with body text."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


def http_get(url: str, timeout: float = 10.0) -> tuple[int, str, str]:
    """GET -> (status, content_type, text). Raises on connection errors."""
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", "") if e.headers else "", e.read().decode("utf-8", "replace")


def http_post_json(url: str, body: dict[str, Any], timeout: float = 600.0,
                   api_key: str | None = None) -> dict[str, Any]:
    """POST JSON, return parsed JSON. Raises HTTPError on non-2xx."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise HTTPError(e.code, e.read().decode("utf-8", "replace")) from None


@dataclass
class StreamResult:
    """Timing + content of one streamed completion. Times are seconds relative to request start."""
    ok: bool = False
    error: str | None = None
    t_start_wall: float = 0.0          # time.time() at send
    t_start_mono: float = 0.0          # time.monotonic() at send
    ttft_s: float | None = None        # first chunk carrying text / reasoning / tool-call delta
    e2e_s: float | None = None         # last byte of the stream
    n_chunks: int = 0                  # chunks carrying generated text
    chunk_gaps_s: list[float] = field(default_factory=list)
    content: str = ""
    reasoning: str = ""
    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    logprobs: list[dict[str, Any]] = field(default_factory=list)
    http_status: int | None = None


def stream_completion(base_url: str, path: str, body: dict[str, Any], timeout: float = 1800.0,
                      api_key: str | None = None) -> StreamResult:
    """POST a streaming chat/completions request and time it.

    Works for /v1/chat/completions (choices[].delta.{content,reasoning_content,reasoning,tool_calls})
    and /v1/completions (choices[].text). Never raises for HTTP/network errors: sets ok=False.
    """
    res = StreamResult()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + path, json.dumps(body).encode(), headers)
    res.t_start_wall = time.time()
    t0 = res.t_start_mono = time.monotonic()
    last_chunk_t: float | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res.http_status = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                now = time.monotonic()
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if d.get("usage"):
                    res.usage = d["usage"]
                if "error" in d and not d.get("choices"):
                    res.error = json.dumps(d["error"])[:500]
                got_text = False
                for ch in d.get("choices") or []:
                    if ch.get("finish_reason"):
                        res.finish_reason = ch["finish_reason"]
                    delta = ch.get("delta")
                    if delta is not None:  # chat
                        c = delta.get("content") or ""
                        rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        tc = delta.get("tool_calls") or []
                        res.content += c
                        res.reasoning += rc
                        res.tool_call_deltas.extend(tc)
                        got_text = got_text or bool(c or rc or tc)
                    elif "text" in ch:  # completions
                        t = ch.get("text") or ""
                        res.content += t
                        got_text = got_text or bool(t)
                    lp = ch.get("logprobs")
                    if isinstance(lp, dict) and isinstance(lp.get("content"), list):
                        res.logprobs.extend(lp["content"])
                if got_text:
                    if res.ttft_s is None:
                        res.ttft_s = now - t0
                    if last_chunk_t is not None:
                        res.chunk_gaps_s.append(now - last_chunk_t)
                    last_chunk_t = now
                    res.n_chunks += 1
        res.e2e_s = time.monotonic() - t0
        res.ok = res.error is None and res.ttft_s is not None
        if res.ttft_s is None and res.error is None:
            res.error = "stream ended without any generated text"
    except urllib.error.HTTPError as e:
        res.http_status = e.code
        res.error = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}"
        res.e2e_s = time.monotonic() - t0
    except Exception as e:  # network errors, timeouts
        res.error = f"{type(e).__name__}: {e}"
        res.e2e_s = time.monotonic() - t0
    return res


def get_model_id(base_url: str, timeout: float = 10.0) -> str:
    """First model id from /v1/models."""
    status, _, text = http_get(base_url.rstrip("/") + "/v1/models", timeout)
    if status != 200:
        raise HTTPError(status, text)
    return json.loads(text)["data"][0]["id"]


def server_identity(base_url: str) -> dict[str, Any]:
    """Model id, owned_by and a version string if the server exposes one (tolerant)."""
    out: dict[str, Any] = {"base_url": base_url}
    try:
        status, _, text = http_get(base_url.rstrip("/") + "/v1/models")
        d = json.loads(text)["data"][0]
        out["model_id"] = d.get("id")
        out["owned_by"] = d.get("owned_by")
        out["max_model_len"] = d.get("max_model_len")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    for path in ("/version", "/get_server_info"):
        try:
            status, _, text = http_get(base_url.rstrip("/") + path, timeout=5)
            if status == 200:
                try:
                    v = json.loads(text)
                except json.JSONDecodeError:
                    v = text[:200]
                if isinstance(v, dict) and path == "/get_server_info":
                    v = {k: v[k] for k in ("version", "tp_size", "speculative_algorithm",
                                           "speculative_num_draft_tokens", "model_path") if k in v}
                out["version_info" if path == "/version" else "server_info"] = v
        except Exception:
            pass
    return out


def guess_framework(ident: dict[str, Any]) -> str:
    """Best-effort framework name from /v1/models owned_by or /version."""
    owned = str(ident.get("owned_by") or "").lower()
    for key, name in (("vllm", "vllm"), ("sglang", "sglang"), ("tensorrt", "trtllm"), ("trt", "trtllm"),
                      ("mock", "mock")):
        if key in owned:
            return name
    if "server_info" in ident:
        return "sglang"
    return "unknown"


# --------------------------------------------------------------------------- Prometheus / metrics
_PROM_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+0-9.eEInfNa]+)(\s+\d+)?$')
_PROM_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition into {'name{k="v",...}': value} (labels sorted)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        if labels:
            pairs = sorted(_PROM_LABEL.findall(labels))
            labels = "{" + ",".join(f'{k}="{v_}"' for k, v_ in pairs) + "}"
        out[name + labels] = v
    return out


def split_series(key: str) -> tuple[str, dict[str, str]]:
    """'name{a="1"}' -> ('name', {'a': '1'})."""
    if "{" not in key:
        return key, {}
    name, rest = key.split("{", 1)
    return name, dict(_PROM_LABEL.findall(rest))


def metric_sum(snapshot: dict[str, float], name: str, exclude_labels: tuple[str, ...] = ("position",)) -> float | None:
    """Sum all series of a metric (accepts with/without `_total`), skipping series that carry any of
    `exclude_labels` (e.g. vLLM per-position acceptance). None if absent."""
    base = name[:-6] if name.endswith("_total") else name
    total, found = 0.0, False
    for key, v in snapshot.items():
        n, labels = split_series(key)
        if n in (base, base + "_total") and not any(lbl in labels for lbl in exclude_labels):
            total += v
            found = True
    return total if found else None


def prom_deltas(before: dict[str, float], after: dict[str, float], pattern: str | None = None) -> dict[str, float]:
    """Changed series between two snapshots (skips histogram _bucket and _created series)."""
    rx = re.compile(pattern, re.I) if pattern else None
    out = {}
    for k, v in after.items():
        name = k.split("{", 1)[0]
        if name.endswith("_bucket") or name.endswith("_created"):
            continue
        if rx and not rx.search(name):
            continue
        d = v - before.get(k, 0.0)
        if d != 0.0:
            out[k] = d
    return out


VLLM_SPEC = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
SPEC_NAME_RX = r"spec|draft|accept|mtp|eagle"


class MetricsScraper:
    """Tolerant scraper for server metrics.

    Probes `/metrics` and `/prometheus/metrics`. Prometheus text is snapshotted before/after a run and
    differenced. JSON payloads (trtllm-serve per-iteration stats, which are *drained* on every GET) are
    accumulated by calling poll() periodically during the run.
    """

    def __init__(self, base_url: str, enabled: bool = True):
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.prom_endpoints: list[str] = []
        self.json_endpoints: list[str] = []
        self.probe_notes: list[str] = []
        self.iter_stats: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if enabled:
            self._probe()

    def _probe(self) -> None:
        for path in ("/metrics", "/prometheus/metrics"):
            try:
                status, ctype, text = http_get(self.base_url + path, timeout=10)
            except Exception as e:
                self.probe_notes.append(f"{path}: {type(e).__name__}")
                continue
            if status != 200:
                self.probe_notes.append(f"{path}: HTTP {status}")
                continue
            stripped = text.lstrip()
            if "json" in ctype or stripped.startswith("[") or stripped.startswith("{"):
                self.json_endpoints.append(path)
                self._absorb_json(text)  # discard-by-drain happens in reset_json()
            elif parse_prometheus(text):
                self.prom_endpoints.append(path)
            else:
                self.probe_notes.append(f"{path}: unrecognised payload ({ctype})")

    def _absorb_json(self, text: str) -> int:
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return 0
        items = d if isinstance(d, list) else [d]
        items = [x for x in items if isinstance(x, dict)]
        with self._lock:
            self.iter_stats.extend(items)
        return len(items)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Prometheus snapshot of every prometheus endpoint."""
        out: dict[str, dict[str, float]] = {}
        for path in self.prom_endpoints:
            try:
                status, _, text = http_get(self.base_url + path, timeout=10)
                if status == 200:
                    out[path] = parse_prometheus(text)
            except Exception:
                pass
        return out

    def reset_json(self) -> None:
        """Drain JSON iteration stats accumulated so far (call right before the measured window)."""
        self.poll()
        with self._lock:
            self.iter_stats = []

    def poll(self) -> None:
        """Fetch (and thereby drain) JSON iteration stats."""
        for path in self.json_endpoints:
            try:
                status, _, text = http_get(self.base_url + path, timeout=10)
                if status == 200:
                    self._absorb_json(text)
            except Exception:
                pass

    def spec_summary(self, before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Spec-decode acceptance over the window from whatever the server exposes, plus raw deltas."""
        out: dict[str, Any] = {"source": None, "acceptance_rate": None, "mean_accepted_tokens_per_step": None,
                               "mean_acceptance_length": None, "drafts": None, "draft_tokens": None,
                               "accepted_tokens": None, "raw_prometheus_deltas": {}, "raw_gauges_after": {},
                               "iteration_stats": None, "probe": {"prometheus": self.prom_endpoints,
                                                                  "json": self.json_endpoints,
                                                                  "notes": self.probe_notes}}
        for path, a in after.items():
            b = before.get(path, {})
            deltas = prom_deltas(b, a)
            spec_deltas = {k: v for k, v in deltas.items() if re.search(SPEC_NAME_RX, k.split("{")[0], re.I)}
            other = {k: v for k, v in deltas.items() if k not in spec_deltas}
            out["raw_prometheus_deltas"][path] = {"spec": spec_deltas, "other": dict(list(other.items())[:200])}
            out["raw_gauges_after"][path] = {k: v for k, v in a.items()
                                             if re.search(SPEC_NAME_RX, k.split("{")[0], re.I)
                                             and not k.split("{")[0].endswith(("_total", "_bucket", "_sum",
                                                                               "_count", "_created"))}
            # vLLM names
            vals = {}
            for key, name in VLLM_SPEC.items():
                va, vb = metric_sum(a, name), metric_sum(b, name)
                vals[key] = (va - (vb or 0.0)) if va is not None else None
            if vals["draft_tokens"] is not None and out["source"] is None:
                out["source"] = f"vllm-prometheus{path}"
                out.update({k: vals[k] for k in vals})
                if vals["draft_tokens"]:
                    out["acceptance_rate"] = vals["accepted_tokens"] / vals["draft_tokens"]
                if vals["drafts"]:
                    out["mean_accepted_tokens_per_step"] = vals["accepted_tokens"] / vals["drafts"]
                    out["mean_acceptance_length"] = 1.0 + vals["accepted_tokens"] / vals["drafts"]
                pos = {}
                for k, v in a.items():
                    n, labels = split_series(k)
                    if n.startswith("vllm:spec_decode_num_accepted_tokens_per_pos") and "position" in labels:
                        pos[labels["position"]] = pos.get(labels["position"], 0.0) + v - b.get(k, 0.0)
                if pos and vals["drafts"]:
                    out["per_position_acceptance"] = {p: pos[p] / vals["drafts"]
                                                      for p in sorted(pos, key=lambda x: int(x) if x.isdigit() else 0)}
        if self.iter_stats:
            out["iteration_stats"] = summarize_iteration_stats(self.iter_stats)
            it = out["iteration_stats"]
            if out["source"] is None and it.get("num_draft_tokens"):
                out["source"] = "trtllm-iteration-stats"
                out["draft_tokens"] = it["num_draft_tokens"]
                out["accepted_tokens"] = it.get("num_accepted_tokens")
                out["drafts"] = it.get("num_requests_with_draft_tokens")
                if out["accepted_tokens"] is not None:
                    out["acceptance_rate"] = out["accepted_tokens"] / out["draft_tokens"]
                    if out["drafts"]:
                        out["mean_accepted_tokens_per_step"] = out["accepted_tokens"] / out["drafts"]
                        out["mean_acceptance_length"] = 1.0 + out["accepted_tokens"] / out["drafts"]
        if out["source"] is None:
            # SGLang-style gauges (e.g. sglang:spec_accept_length) are recorded as-reported only.
            for path, g in out["raw_gauges_after"].items():
                for k, v in g.items():
                    if "accept_length" in k and v > 0 and out["mean_acceptance_length"] is None:
                        out["mean_acceptance_length"] = v
                        out["source"] = f"gauge{path}:{k} (server-side window, not window-exact)"
        return out


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", k.lower())


def summarize_iteration_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum numeric fields of any spec-decoding sub-dict across TRT-LLM iteration-stat records.

    Field names are matched case/underscore-insensitively, e.g. specDecodingStats.numDraftTokens or
    spec_decoding_stats.num_draft_tokens.
    """
    sums: dict[str, float] = {}
    accept_len: list[float] = []
    n_with_spec = 0
    for it in items:
        spec = None
        for k, v in it.items():
            if isinstance(v, dict) and "spec" in k.lower():
                spec = v
                break
        if spec is None:
            continue
        n_with_spec += 1
        for k, v in spec.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nk = _norm_key(k)
                if nk == "acceptancelength":
                    if v > 0:
                        accept_len.append(float(v))
                    continue
                sums[nk] = sums.get(nk, 0.0) + float(v)
    out: dict[str, Any] = {"iterations": len(items), "iterations_with_spec_stats": n_with_spec,
                           "raw_sums": sums,
                           "mean_reported_acceptance_length": (sum(accept_len) / len(accept_len)) if accept_len else None}
    out["num_draft_tokens"] = sums.get("numdrafttokens")
    out["num_accepted_tokens"] = sums.get("numacceptedtokens")
    out["num_requests_with_draft_tokens"] = sums.get("numrequestswithdrafttokens")
    return out


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank/corrupt lines."""
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record (creates parent dirs)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=False) + "\n")

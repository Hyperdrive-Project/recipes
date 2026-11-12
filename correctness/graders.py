# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Scoring helpers for correctness/eval.py (standard library only).

* GSM8K numeric answer extraction + exact match
* HumanEval code extraction + sandboxed execution (subprocess, timeout, rlimits, network blocked)
* tool-call extraction (server `tool_calls` or client-side qwen3_coder XML fallback) + argument matching
* short-answer normalisation for image questions
* token-level diff vs a stored greedy reference
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from typing import Any

# ----------------------------------------------------------------------------- math
_NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def _to_decimal(s: str) -> Decimal | None:
    s = s.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def gsm8k_gold(answer: str) -> Decimal | None:
    """Gold value after '####'."""
    return _to_decimal(answer.split("####")[-1].strip())


def extract_number(text: str) -> tuple[Decimal | None, str]:
    """Prediction: last 'Answer: <n>' (also \\boxed{n}); else last number in the text. Returns (value, method)."""
    text = text.replace("**", "")
    m = re.findall(r"answer\s*(?:is|:)?\s*[:=]?\s*(" + _NUM + ")", text, flags=re.I)
    if m:
        return _to_decimal(m[-1]), "answer-tag"
    m = re.findall(r"\\boxed\{\s*(" + _NUM + r")\s*\}", text)
    if m:
        return _to_decimal(m[-1]), "boxed"
    m = re.findall(_NUM, text)
    if m:
        return _to_decimal(m[-1]), "last-number"
    return None, "none"


def numbers_equal(a: Decimal | None, b: Decimal | None) -> bool:
    return a is not None and b is not None and a == b


# ----------------------------------------------------------------------------- code
def extract_code(text: str) -> str:
    """First ```python block (or any ``` block); otherwise the raw text."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1)
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, flags=re.S | re.I)  # unterminated block
    return m.group(1) if m else text


def build_program(prompt: str, completion_code: str, test: str, entry_point: str) -> str:
    """Assemble a HumanEval program. If the completion defines the entry point, prepend only the prompt's
    import lines; otherwise treat it as a function body continuing the prompt."""
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", completion_code, flags=re.M):
        imports = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
        body = imports + "\n\n" + completion_code
    else:
        body = prompt + completion_code
    return body + "\n\n" + test + f"\n\ncheck({entry_point})\n"


_GUARD = r"""
import resource as _r
_r.setrlimit(_r.RLIMIT_AS, ({mem}, {mem}))
_r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu} + 1))
_r.setrlimit(_r.RLIMIT_FSIZE, (16 << 20, 16 << 20))
_r.setrlimit(_r.RLIMIT_CORE, (0, 0))
import builtins as _b, os as _os, shutil as _sh, subprocess as _sp, socket as _so, faulthandler as _fh
def _blocked(*a, **k):
    raise PermissionError("blocked by hyperdrive sandbox")
_so.socket = _blocked; _so.create_connection = _blocked; _so.socketpair = _blocked; _so.getaddrinfo = _blocked
for _n in ("system", "kill", "killpg", "fork", "forkpty", "execv", "execve", "execvp", "execvpe", "spawnv",
           "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "truncate", "putenv", "chmod",
           "chown", "setuid", "setgid"):
    if hasattr(_os, _n):
        setattr(_os, _n, _blocked)
_sh.rmtree = _blocked; _sh.move = _blocked; _sp.Popen = _blocked; _sp.run = _blocked; _sp.call = _blocked
_sp.check_output = _blocked; _sp.check_call = _blocked
_fh.disable()
del _b, _os, _sh, _sp, _so, _fh, _r
"""


_UNSHARE: bool | None = None


def netns_available() -> bool:
    """True if `unshare -rn` works for this user (unprivileged user+net namespace)."""
    global _UNSHARE
    if _UNSHARE is None:
        try:
            _UNSHARE = shutil.which("unshare") is not None and subprocess.run(
                ["unshare", "-rn", "true"], capture_output=True, timeout=5).returncode == 0
        except Exception:
            _UNSHARE = False
    return _UNSHARE


def sandbox_method() -> str:
    return ("unshare-netns + python guard + rlimits" if netns_available()
            else "python guard (socket/os/subprocess blocked) + rlimits (no unprivileged netns on this host)")


def run_sandboxed(program: str, timeout_s: float = 15.0, mem_mb: int = 2048) -> dict[str, Any]:
    """Execute untrusted Python in a fresh temp dir: `python -I -B`, empty env, CPU/AS/FSIZE rlimits set by
    the child itself (no preexec_fn, so safe with threads), a guard that blocks sockets/subprocess/destructive
    os calls, and the child's own process group killed on timeout."""
    tmp = tempfile.mkdtemp(prefix="hd-sandbox-")
    path = os.path.join(tmp, "prog.py")
    with open(path, "w") as f:
        f.write(_GUARD.format(mem=mem_mb << 20, cpu=int(timeout_s) + 1) + "\n" + program)
    cmd = [sys.executable, "-I", "-B", path]
    if netns_available():
        cmd = ["unshare", "-rn", *cmd]
    env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}
    try:
        p = subprocess.Popen(cmd, cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        try:
            out, err = p.communicate(timeout=timeout_s)
            status = "passed" if p.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)  # only the sandbox child's own process group
            out, err = p.communicate()
            status = "timeout"
        return {"status": status, "returncode": p.returncode, "stderr_tail": (err or "")[-600:],
                "stdout_tail": (out or "")[-200:]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- tools
_TC_BLOCK = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
_TC_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value: str, schema: dict[str, Any] | None) -> Any:
    t = (schema or {}).get("type")
    v = value.strip()
    try:
        if t == "integer":
            return int(v)
        if t == "number":
            f = float(v)
            return int(f) if f.is_integer() and "." not in v else f
        if t == "boolean":
            return {"true": True, "false": False}.get(v.lower(), v)
        if t in ("object", "array"):
            return json.loads(v)
    except (ValueError, json.JSONDecodeError):
        return v
    return v


def parse_xml_tool_calls(text: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Client-side fallback parser for the template's qwen3_coder format:
    <tool_call><function=name><parameter=p>value</parameter>...</function></tool_call>."""
    schemas = {}
    for t in tools:
        fn = t.get("function", t)
        schemas[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for name, inner in _TC_BLOCK.findall(text or ""):
        props = schemas.get(name, {})
        args = {k: _coerce(v, props.get(k)) for k, v in _TC_PARAM.findall(inner)}
        calls.append({"name": name, "arguments": args})
    return calls


def server_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise OpenAI `message.tool_calls` -> [{name, arguments(dict)}]."""
    out = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"__unparsed__": args}
        out.append({"name": fn.get("name"), "arguments": args if isinstance(args, dict) else {"__value__": args}})
    return out


def match_value(got: Any, spec: Any) -> bool:
    """Expected-arg matcher. spec is a literal (strings compared case-insensitively and stripped, numbers
    with tolerance) or {"any_of": [...]}, {"nonempty": true}, {"contains": "substr"}."""
    if isinstance(spec, dict):
        if "any_of" in spec:
            return any(match_value(got, s) for s in spec["any_of"])
        if spec.get("nonempty"):
            return got not in (None, "", [], {})
        if "contains" in spec:
            return isinstance(got, str) and spec["contains"].lower() in got.lower()
        return isinstance(got, dict) and all(k in got and match_value(got[k], v) for k, v in spec.items())
    if isinstance(spec, bool):
        if isinstance(got, str):
            return got.strip().lower() == str(spec).lower()
        return got is spec
    if isinstance(spec, (int, float)):
        try:
            return abs(float(got) - float(spec)) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(spec, str):
        return isinstance(got, str) and got.strip().lower() == spec.strip().lower()
    if isinstance(spec, list):
        return isinstance(got, list) and len(got) == len(spec) and all(match_value(g, s) for g, s in zip(got, spec))
    return got == spec


def grade_tool_case(calls: list[dict[str, Any]], expect: dict[str, Any]) -> dict[str, Any]:
    """expect = {"name": str|None, "args": {...}}; name None means 'no tool call expected'."""
    if expect.get("name") is None:
        return {"correct": not calls, "name_ok": not calls, "args_ok": not calls}
    if not calls:
        return {"correct": False, "name_ok": False, "args_ok": False}
    c = calls[0]
    name_ok = c["name"] == expect["name"]
    args = c["arguments"] or {}
    missing = [k for k in expect.get("args", {}) if k not in args]
    wrong = [k for k, v in expect.get("args", {}).items() if k in args and not match_value(args[k], v)]
    args_ok = name_ok and not missing and not wrong
    return {"correct": args_ok, "name_ok": name_ok, "args_ok": args_ok, "missing": missing, "wrong": wrong}


# ----------------------------------------------------------------------------- short answers
def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def short_answer_ok(text: str, expected: str, kind: str, alternatives: list[str] | None = None) -> bool:
    """Image/needle short answers.
    number: the first integer in the reply equals expected.
    letter: the first standalone capital letter in the reply equals expected.
    word:   expected appears as a whole word and none of the mutually exclusive `alternatives` does.
    code:   expected (alphanumerics only, case-insensitive) is a substring of the reply.
    """
    text = text or ""
    if kind == "number":
        m = re.findall(r"-?\d+", text.replace(",", ""))
        return bool(m) and m[0] == expected
    if kind == "letter":
        m = re.findall(r"\b([A-Z])\b", text)
        return bool(m) and m[0] == expected.upper()
    if kind == "code":
        squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
        return squash(expected) in squash(text)
    words = norm_words(text)
    return expected.lower() in words and not any(a.lower() in words for a in (alternatives or []))


# ----------------------------------------------------------------------------- determinism
def token_diff(ref: list[Any], cur: list[Any]) -> dict[str, Any]:
    """Prefix match length, first divergence, position-wise match rate over max(len)."""
    n = min(len(ref), len(cur))
    prefix = 0
    while prefix < n and ref[prefix] == cur[prefix]:
        prefix += 1
    same = sum(1 for a, b in zip(ref, cur) if a == b)
    denom = max(len(ref), len(cur)) or 1
    return {"ref_len": len(ref), "cur_len": len(cur), "prefix_match": prefix,
            "first_divergence": None if (prefix == len(ref) == len(cur)) else prefix,
            "positionwise_match_rate": same / denom, "prefix_match_rate": prefix / (len(ref) or 1),
            "exact": list(ref) == list(cur)}

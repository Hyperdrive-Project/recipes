# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Helpers for humaneval_instruct_think (see the YAML). pass_at_k is lm-eval's stock HumanEval metric."""
import re

from lm_eval.tasks.humaneval.utils import pass_at_k  # noqa: F401  (stock metric, re-exported for the YAML)

_BLOCK = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)```", re.S)


def extract_program(prompt: str, entry_point: str, resp: str | None) -> str:
    resp = resp or ""
    blocks = _BLOCK.findall(resp)
    if not blocks and "```" in resp:  # unterminated final block
        blocks = [re.split(r"```[ \t]*(?:python3?|py)?[ \t]*\n", resp)[-1]]
    defines = [b for b in blocks if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", b, re.M)]
    code = defines[-1] if defines else (blocks[-1] if blocks else resp)
    # The prompt's stub (signature + docstring) stays first so its imports and helper functions exist; a full definition in
    # the answer then replaces the stub, a body-only answer continues it (as in the stock completion-style task).
    return prompt + ("\n" + code if defines else code)


def build_predictions_think(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_program(doc["prompt"], doc["entry_point"], r) for r in resp] for resp, doc in zip(resps, docs)]

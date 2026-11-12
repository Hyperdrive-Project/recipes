# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Deterministic synthetic corpora and exact-token-length prompt construction.

* `prose(rng, n_chars)`: seeded, natural-reading English "archive documents" (towns, rivers, people,
  institutions, numbers, dates) so the model's continuation (a report about them) is realistic text
  for speculative-decoding acceptance.
* `python_code(rng, n_chars)`: seeded, plausible Python modules (classes, dataclasses, functions with
  docstrings/type hints/loops/error handling) for the coding workload.
* `ChatTokenizer`: model tokenizer + chat template (via transformers) that counts tokens exactly as an
  OpenAI chat server does (template rendered with the same chat_template_kwargs, add_special_tokens=False).
* `fit_exact()`: chooses the corpus prefix length so the *rendered* prompt has exactly N tokens.

Everything here is pure Python except ChatTokenizer (transformers + tokenizers).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

TEXTGEN_VERSION = "1"

# ----------------------------------------------------------------------------- vocabulary
_SYL_A = ["al", "bran", "cor", "dun", "el", "fal", "gar", "hal", "is", "kel", "lor", "mar", "nor", "or",
          "pen", "quar", "ros", "sel", "tor", "ul", "ver", "wen", "yar", "zan", "ash", "bel", "cal", "dor"]
_SYL_B = ["ford", "wick", "mere", "ton", "by", "dale", "holm", "stead", "bury", "mouth", "field", "brook",
          "haven", "moor", "gate", "ridge", "wood", "port", "ley", "worth", "combe", "thorpe"]
_FIRST = ["Amara", "Bennett", "Clara", "Dmitri", "Elena", "Farid", "Greta", "Hugo", "Ines", "Jonah", "Keiko",
          "Lucia", "Marcus", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Samuel", "Talia", "Umar", "Vera",
          "Wendell", "Ximena", "Yusuf", "Zora", "Adele", "Bruno", "Celeste", "Desmond", "Esther", "Felix"]
_LAST = ["Abernathy", "Brightwater", "Castellan", "Delacroix", "Eastwood", "Fairbanks", "Goldfinch",
         "Hargrove", "Ingram", "Jansen", "Kowalski", "Lindqvist", "Mbeki", "Navarro", "Okafor", "Pemberton",
         "Quinlan", "Rasmussen", "Sorensen", "Takahashi", "Underhill", "Valdez", "Whitlock", "Yamamoto"]
_INST = ["Historical Society", "Water Board", "Agricultural Cooperative", "Railway Company", "Public Library",
         "Chamber of Commerce", "Observatory", "Teachers' College", "Harbour Authority", "Textile Guild",
         "Botanical Garden", "Geological Survey", "Mutual Aid Society", "Orchestra", "Hospital Trust"]
_GOODS = ["wool", "barley", "copper", "timber", "salt", "glassware", "dried fish", "apples", "linen", "slate",
          "honey", "rope", "leather", "tin", "pottery", "cheese", "paper", "dye", "coal", "oats"]
_TOPICS = ["irrigation", "bridge repair", "school enrolment", "harbour dredging", "grain storage",
           "flood defences", "street lighting", "the new rail spur", "public health", "forest management",
           "market regulation", "postal routes", "mine safety", "orchard disease", "census methods"]
_ADJ = ["modest", "remarkable", "steady", "uneven", "ambitious", "cautious", "prolonged", "unexpected",
        "gradual", "vigorous", "disputed", "careful", "costly", "successful", "short-lived", "lasting"]
_SEASON = ["spring", "summer", "autumn", "winter"]
_MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
_WEATHER = ["a dry summer", "heavy autumn rains", "an unusually cold winter", "late spring frosts",
            "a mild and wet year", "a long drought", "repeated storms along the coast"]

_SENTENCES = [
    "{town} lies on the {dir} bank of the {river}, about {n2} kilometres from {town2}.",
    "In {year}, the {town} {inst} recorded {n3} members, a {adj} increase over the previous decade.",
    "{person}, who chaired the {inst} from {year} to {year2}, argued that {topic} deserved priority.",
    "Records from {month} {year} show that {n2} carts of {good} left {town} for the markets of {town2}.",
    "The council debated {topic} for {n1} sessions before approving a budget of {money} crowns.",
    "After {weather}, the price of {good} rose by {pct} percent, and several families left for {town2}.",
    "A survey led by {person} counted {n3} households and {n2} working mills along the {river}.",
    "Local accounts describe the {adj} reconstruction of the old bridge, completed in the {season} of {year}.",
    "{person} wrote in a letter that the {inst} \"cannot keep pace with the demands of {topic}\".",
    "Trade in {good} and {good2} accounted for nearly {pct} percent of the town's income by {year}.",
    "The {inst} published its first annual report in {year}, listing {n2} projects and {n1} disputes.",
    "During the {season} fair, merchants from {town2} exchanged {good} for {good2} at a fixed rate.",
    "A {adj} dispute over {topic} between {town} and {town2} was settled by arbitration in {year2}.",
    "Population estimates for {town} range from {n3} to {n3b}, depending on whether outlying farms are counted.",
    "{person} and {person2} founded a small school near the {river} that taught {n2} pupils in its first year.",
    "Engineers measured the {river} at {n1}.{d} metres deep near the mill weir during {month}.",
    "By {year2}, the {inst} had raised {money} crowns through subscriptions and a lottery.",
    "The minutes note that {person} objected to the cost, while {person2} insisted that {topic} could not wait.",
    "Maps drawn in {year} place the market square {n1}00 metres north of the present town hall.",
    "Shipments of {good} were delayed for {n1} weeks after {weather} damaged the road to {town2}.",
    "Older residents recall that the {river} froze completely in {year}, allowing carts to cross on the ice.",
    "An inventory from {year} lists {n2} barrels of {good}, {n2b} bales of {good2}, and {n1} crates of tools.",
    "The {adj} growth of {town} after {year} is usually credited to {topic} and to the arrival of the railway.",
    "Visitors in the {season} of {year2} described the streets as narrow, busy and lined with {good} stalls.",
]
_TITLES = ["A Short History of {town}", "Minutes of the {town} {inst}", "Trade Ledger of {town}, {year}",
           "Survey of the {river} Valley", "Letters of {person}", "Annual Report of the {inst}",
           "Notes on {topic} in {town}", "Recollections of {town2} and {town}"]


def _name(rng: random.Random) -> str:
    return (rng.choice(_SYL_A) + rng.choice(_SYL_B)).capitalize()


def _fill(rng: random.Random, tpl: str, ctx: dict[str, str]) -> str:
    year = rng.randint(1780, 1935)
    vals = {
        "town": ctx["town"], "town2": rng.choice(ctx["towns"]), "river": ctx["river"],
        "inst": rng.choice(_INST), "person": rng.choice(ctx["people"]), "person2": rng.choice(ctx["people"]),
        "year": str(year), "year2": str(year + rng.randint(2, 30)), "month": rng.choice(_MONTH),
        "season": rng.choice(_SEASON), "good": rng.choice(_GOODS), "good2": rng.choice(_GOODS),
        "topic": rng.choice(_TOPICS), "adj": rng.choice(_ADJ), "weather": rng.choice(_WEATHER),
        "dir": rng.choice(["northern", "southern", "eastern", "western"]),
        "n1": str(rng.randint(2, 9)), "n2": str(rng.randint(12, 95)), "n2b": str(rng.randint(12, 95)),
        "n3": f"{rng.randint(1, 9)},{rng.randint(100, 999)}", "n3b": f"{rng.randint(10, 19)},{rng.randint(100, 999)}",
        "pct": str(rng.randint(3, 60)), "money": f"{rng.randint(2, 90)},{rng.randint(0, 9)}00", "d": str(rng.randint(1, 9)),
    }
    return tpl.format(**vals)


def prose(rng: random.Random, n_chars: int) -> str:
    """Seeded natural-reading archive documents, at least n_chars long."""
    parts: list[str] = []
    total = 0
    doc = 0
    while total < n_chars:
        doc += 1
        towns = [_name(rng) for _ in range(4)]
        ctx = {"town": towns[0], "towns": towns[1:], "river": "River " + _name(rng),
               "people": [f"{rng.choice(_FIRST)} {rng.choice(_LAST)}" for _ in range(4)]}
        title = _fill(rng, rng.choice(_TITLES), ctx)
        paras = []
        for _ in range(rng.randint(2, 5)):
            sents = [_fill(rng, rng.choice(_SENTENCES), ctx) for _ in range(rng.randint(3, 7))]
            paras.append(" ".join(sents))
        block = f"Document {doc}: {title}\n\n" + "\n\n".join(paras)
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- code
_NOUNS = ["order", "invoice", "customer", "shipment", "sensor", "reading", "account", "ticket", "batch",
          "record", "route", "station", "booking", "item", "payment", "report", "event", "job", "user", "asset"]
_VERBS = ["load", "parse", "validate", "compute", "merge", "filter", "group", "normalize", "summarize",
          "export", "resolve", "schedule", "reconcile", "score", "rank", "index", "update", "archive"]
_FIELDS = ["id", "name", "status", "created_at", "amount", "quantity", "region", "priority", "owner",
           "tags", "weight", "price", "currency", "updated_at", "notes", "category", "duration", "score"]
_TYPES = {"id": "int", "name": "str", "status": "str", "created_at": "float", "amount": "float",
          "quantity": "int", "region": "str", "priority": "int", "owner": "str", "tags": "list[str]",
          "weight": "float", "price": "float", "currency": "str", "updated_at": "float", "notes": "str",
          "category": "str", "duration": "float", "score": "float"}


def _cls(n: str) -> str:
    return "".join(p.capitalize() for p in n.split("_"))


def _func(rng: random.Random, noun: str, fields: list[str]) -> str:
    verb = rng.choice(_VERBS)
    f1, f2 = rng.sample(fields, 2)
    cls = _cls(noun)
    kind = rng.randint(0, 4)
    name = f"{verb}_{noun}s"
    if kind == 0:
        return (f"def {name}(items: list[{cls}], threshold: float = {rng.randint(1, 99)}.0) -> list[{cls}]:\n"
                f'    """{verb.capitalize()} {noun}s whose {f1} passes the threshold, sorted by {f2}."""\n'
                f"    selected = []\n"
                f"    for item in items:\n"
                f"        value = getattr(item, \"{f1}\", None)\n"
                f"        if value is None:\n"
                f"            continue\n"
                f"        if isinstance(value, (int, float)) and value >= threshold:\n"
                f"            selected.append(item)\n"
                f"    selected.sort(key=lambda x: str(getattr(x, \"{f2}\", \"\")))\n"
                f"    return selected\n")
    if kind == 1:
        return (f"def {name}_by_{f1}(items: Iterable[{cls}]) -> dict[str, list[{cls}]]:\n"
                f'    """Group {noun}s by their {f1} field."""\n'
                f"    groups: dict[str, list[{cls}]] = defaultdict(list)\n"
                f"    for item in items:\n"
                f"        key = str(getattr(item, \"{f1}\", \"unknown\")).strip().lower()\n"
                f"        groups[key].append(item)\n"
                f"    return dict(groups)\n")
    if kind == 2:
        return (f"def {name}_from_rows(rows: list[dict[str, Any]]) -> list[{cls}]:\n"
                f'    """Build {cls} objects from raw rows, skipping malformed entries."""\n'
                f"    result: list[{cls}] = []\n"
                f"    for line_no, row in enumerate(rows, start=1):\n"
                f"        try:\n"
                f"            result.append({cls}.from_dict(row))\n"
                f"        except (KeyError, ValueError) as exc:\n"
                f"            logger.warning(\"row %d skipped: %s\", line_no, exc)\n"
                f"    return result\n")
    if kind == 3:
        return (f"def {name}_stats(items: Sequence[{cls}]) -> dict[str, float]:\n"
                f'    """Return count, mean and maximum of the {f1} field."""\n'
                f"    values = [float(getattr(i, \"{f1}\", 0) or 0) for i in items]\n"
                f"    if not values:\n"
                f"        return {{\"count\": 0.0, \"mean\": 0.0, \"max\": 0.0}}\n"
                f"    return {{\n"
                f"        \"count\": float(len(values)),\n"
                f"        \"mean\": sum(values) / len(values),\n"
                f"        \"max\": max(values),\n"
                f"    }}\n")
    return (f"def {name}_to_csv(items: list[{cls}], path: str) -> int:\n"
            f'    """Write {noun}s to a CSV file and return the number of rows written."""\n'
            f"    columns = [\"{f1}\", \"{f2}\"]\n"
            f"    with open(path, \"w\", newline=\"\") as handle:\n"
            f"        writer = csv.writer(handle)\n"
            f"        writer.writerow(columns)\n"
            f"        for item in items:\n"
            f"            writer.writerow([getattr(item, c, \"\") for c in columns])\n"
            f"    return len(items)\n")


def python_code(rng: random.Random, n_chars: int) -> str:
    """Seeded plausible Python source, at least n_chars long, split into '# file:' sections."""
    parts: list[str] = []
    total = 0
    while total < n_chars:
        noun = rng.choice(_NOUNS)
        cls = _cls(noun)
        fields = rng.sample(_FIELDS, rng.randint(4, 7))
        hdr = (f"# file: {noun}_{rng.choice(_VERBS)}er.py\n"
               f'"""Utilities for working with {noun} records."""\n'
               "from __future__ import annotations\n\nimport csv\nimport logging\n"
               "from collections import defaultdict\nfrom dataclasses import dataclass, field\n"
               "from typing import Any, Iterable, Sequence\n\nlogger = logging.getLogger(__name__)\n\n\n")
        body = [f"@dataclass\nclass {cls}:\n    \"\"\"A single {noun} record.\"\"\"\n"]
        for f_ in fields:
            t = _TYPES[f_]
            default = " = field(default_factory=list)" if t.startswith("list") else ""
            body.append(f"    {f_}: {t}{default}\n")
        body.append(f"\n    @classmethod\n    def from_dict(cls, row: dict[str, Any]) -> \"{cls}\":\n"
                    f"        \"\"\"Parse a raw dictionary into a {cls}.\"\"\"\n        return cls(\n")
        for f_ in fields:
            t = _TYPES[f_]
            conv = {"int": "int", "float": "float", "str": "str"}.get(t, "list")
            body.append(f"            {f_}={conv}(row[\"{f_}\"]),\n")
        body.append("        )\n\n\n")
        for _ in range(rng.randint(2, 4)):
            body.append(_func(rng, noun, fields) + "\n\n")
        block = hdr + "".join(body)
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


QUESTIONS = {
    "text": ("Write a detailed, well-organized report on the archive documents above. Go through every "
             "document in order, summarize its key facts, figures, people and places, and then discuss the "
             "connections between the towns. Be thorough and keep writing until the report is complete."),
    "code": ("Refactor the Python modules above into a cleaner, fully type-annotated package. Rewrite every "
             "class and function completely (do not abbreviate or skip any), improve error handling and "
             "docstrings, and then write a comprehensive pytest test suite covering all of them."),
}


def corpus(kind: str, seed: int, n_chars: int) -> str:
    """Deterministic corpus of the given kind ('text'/'image' -> prose, 'code' -> python)."""
    rng = random.Random(seed)
    return python_code(rng, n_chars) if kind == "code" else prose(rng, n_chars)


# ----------------------------------------------------------------------------- tokenizer
def resolve_tokenizer_dir(path: str | None) -> str:
    """Return a directory containing tokenizer.json: explicit path, else models/super-vl-nvfp4, else the
    pinned HF cache snapshot (local files only, no network)."""
    from benchmarks.common import MODEL_REPO_ID, MODEL_REVISION, REPO_ROOT
    cands = [Path(path)] if path else [REPO_ROOT / "models" / "super-vl-nvfp4"]
    for c in cands:
        if (c / "tokenizer.json").exists():
            return str(c)
    if path and not Path(path).exists():
        return path  # maybe a HF repo id; let transformers resolve it
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(MODEL_REPO_ID, revision=MODEL_REVISION, local_files_only=True,
                                 allow_patterns=["tokenizer*", "chat_template.jinja", "*.json"])
        if (Path(snap) / "tokenizer.json").exists():
            return snap
    except Exception:
        pass
    raise FileNotFoundError(f"no tokenizer.json found in {cands[0]} (model download incomplete?) and no "
                            f"cached snapshot of {MODEL_REPO_ID}@{MODEL_REVISION[:8]}; pass --tokenizer")


class ChatTokenizer:
    """Counts prompt tokens exactly as an OpenAI chat server would (template + add_special_tokens=False)."""

    def __init__(self, path: str | None = None):
        import logging
        import os
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        from transformers import AutoTokenizer  # type: ignore
        self.path = resolve_tokenizer_dir(path)
        self.tok = AutoTokenizer.from_pretrained(self.path)

    def render(self, messages: list[dict[str, Any]], template_kwargs: dict[str, Any] | None = None,
               tools: list[dict[str, Any]] | None = None) -> str:
        kw = dict(template_kwargs or {})
        if tools:
            kw["tools"] = tools
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)

    def encode(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def count_chat(self, messages: list[dict[str, Any]], template_kwargs: dict[str, Any] | None = None) -> int:
        return len(self.encode(self.render(messages, template_kwargs)))

    def chat_ids(self, messages: list[dict[str, Any]], template_kwargs: dict[str, Any] | None = None) -> list[int]:
        return self.encode(self.render(messages, template_kwargs))


def fit_exact(count: Callable[[int], int], target: int, n0: int, max_n: int) -> tuple[int, int]:
    """Find n (corpus-token prefix length) so that count(n) == target.

    `count(n)` is the rendered-prompt token count when the first n corpus tokens are used. It is close to
    n + constant but boundary merges make it non-additive, so we iterate on the error and then scan locally.
    Returns (n, count(n)); count(n) != target only if no exact n exists nearby.
    """
    cache: dict[int, int] = {}

    def c(n: int) -> int:
        n = max(0, min(n, max_n))
        if n not in cache:
            cache[n] = count(n)
        return cache[n]

    n = max(0, min(n0, max_n))
    for _ in range(12):
        got = c(n)
        if got == target:
            return n, got
        nxt = max(0, min(n + (target - got), max_n))
        if nxt == n or nxt in cache:
            break
        n = nxt
    for radius in range(1, 9):
        for cand in (n - radius, n + radius):
            if 0 <= cand <= max_n and c(cand) == target:
                return cand, target
    best = min(cache, key=lambda k: (abs(cache[k] - target), -k))
    return best, cache[best]


def build_exact_prompt(tok: ChatTokenizer, target_tokens: int, kind: str, seed: int,
                       template_kwargs: dict[str, Any], question: str | None = None,
                       header: str = "", make_content: Callable[[str], str] | None = None,
                       extra_parts: list[dict[str, Any]] | None = None,
                       reserve_tokens: int = 0) -> dict[str, Any]:
    """Build a single-user-message chat prompt whose rendered token count is exactly `target_tokens`.

    The user content is `header + corpus_prefix + "\\n\\n" + question` (or `make_content(corpus_prefix)`).
    `extra_parts` (e.g. an image part) are appended to the content list; `reserve_tokens` is subtracted from
    the text budget for them (their real token cost is only known to the server).
    Returns {"messages", "prompt_tokens" (local count), "exact", "corpus_tokens", "sha256"}.
    """
    from benchmarks.common import sha256_text
    question = question if question is not None else QUESTIONS.get(kind, QUESTIONS["text"])
    text_target = target_tokens - reserve_tokens

    def content_for(body: str) -> str:
        return make_content(body) if make_content else f"{header}{body}\n\n{question}"

    def messages_for(body: str) -> list[dict[str, Any]]:
        c = content_for(body)
        if extra_parts:
            return [{"role": "user", "content": [*extra_parts, {"type": "text", "text": c}]}]
        return [{"role": "user", "content": c}]

    def text_messages(body: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": content_for(body)}]

    overhead = tok.count_chat(text_messages(""), template_kwargs)
    budget = max(0, text_target - overhead)
    ids = tok.encode(corpus(kind, seed, int(budget * 6) + 2000))
    while len(ids) < budget + 64:
        ids = tok.encode(corpus(kind, seed, int(len(ids) * 8) + 4000))

    # Whitespace runs (code indentation) can merge with the following separator so that no prefix length
    # hits the target exactly; retry with a few deterministic boundary treatments.
    trims: list[Callable[[str], str]] = [lambda s: s, lambda s: s.rstrip(), lambda s: s.rstrip() + " ...",
                                         lambda s: s.rstrip() + ".", lambda s: s.rstrip() + "\n#"]
    for trim in trims:
        def count(n: int, trim: Callable[[str], str] = trim) -> int:
            return tok.count_chat(text_messages(trim(tok.decode(ids[:n]))), template_kwargs)

        n, got = fit_exact(count, text_target, budget, len(ids))
        if got == text_target:
            break
    body = trim(tok.decode(ids[:n]))
    msgs = messages_for(body)
    rendered = tok.render(text_messages(body), template_kwargs)
    return {"messages": msgs, "prompt_tokens": got + reserve_tokens, "text_tokens": got,
            "exact": got == text_target, "corpus_tokens": n, "sha256": sha256_text(rendered),
            "prompt_token_ids": None if extra_parts else tok.encode(rendered)}

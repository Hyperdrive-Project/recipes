# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Task definitions for correctness/eval.py: data loading, hand-written tool cases, procedural images,
needle-in-a-haystack prompts and fixed greedy determinism prompts."""
from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# max_tokens per task per reasoning mode (reasoning and answer share the budget)
MAX_TOKENS = {
    "math": {"off": 1024, "medium": 4096, "on": 8192},
    "code": {"off": 1536, "medium": 4096, "on": 8192},
    "tools": {"off": 512, "medium": 2048, "on": 4096},
    "image": {"off": 128, "medium": 2048, "on": 4096},
    "needle": {"off": 96, "medium": 2048, "on": 4096},
    "determinism": {"off": 128, "medium": 128, "on": 128},
}


# ----------------------------------------------------------------------------- math / code
def load_jsonl(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: tools-venv/bin/python correctness/fetch_data.py")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


MATH_PROMPT = ("Solve the following math problem. Show brief reasoning, then give the final answer on its own "
               "line in the form 'Answer: <number>'.\n\n{question}")
CODE_PROMPT = ("Complete the following Python function. Respond with the complete function implementation "
               "(including the signature and any needed imports) in a single ```python code block.\n\n"
               "```python\n{prompt}```")


# ----------------------------------------------------------------------------- tools
def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


_WEATHER = _fn("get_weather", "Get the current weather for a city.",
               {"city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city", "unit"])
_INVENTORY = _fn("get_inventory", "Return the inventory count for a product.",
                 {"product": {"type": "string", "enum": ["keyboard", "monitor", "mouse"]}}, ["product"])
_CURRENCY = _fn("convert_currency", "Convert an amount between currencies (ISO 4217 codes).",
                {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO code, e.g. USD"},
                 "to_currency": {"type": "string", "description": "ISO code, e.g. EUR"}},
                ["amount", "from_currency", "to_currency"])
_CAL = _fn("create_calendar_event", "Create a calendar event.",
           {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
            "duration_minutes": {"type": "integer"}}, ["title", "date", "duration_minutes"])
_FLIGHTS = _fn("search_flights", "Search flights between airports (IATA codes).",
               {"origin": {"type": "string"}, "destination": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"}, "passengers": {"type": "integer"}},
               ["origin", "destination", "date", "passengers"])
_TIP = _fn("calculate_tip", "Calculate a restaurant tip.",
           {"bill_amount": {"type": "number"}, "tip_percent": {"type": "number"}}, ["bill_amount", "tip_percent"])
_EMAIL = _fn("send_email", "Send an email.",
             {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
             ["to", "subject", "body"])
_STOCK = _fn("get_stock_price", "Get the latest stock price for a ticker symbol.",
             {"symbol": {"type": "string"}}, ["symbol"])
_NEWS = _fn("get_company_news", "Get recent news items about a company by ticker symbol.",
            {"symbol": {"type": "string"}, "limit": {"type": "integer", "description": "number of items"}},
            ["symbol", "limit"])
_THERMO = _fn("set_thermostat", "Set the thermostat.",
              {"temperature": {"type": "number", "description": "degrees Celsius"},
               "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}}, ["temperature", "mode"])
_ORDER = _fn("lookup_order", "Look up an order by id.",
             {"order_id": {"type": "string"}, "include_items": {"type": "boolean"}}, ["order_id", "include_items"])

TOOL_CASES: list[dict[str, Any]] = [
    {"id": "weather", "tools": [_WEATHER], "prompt": "What's the weather like in Paris right now? Use celsius.",
     "expect": {"name": "get_weather", "args": {"city": "Paris", "unit": "celsius"}}},
    {"id": "inventory", "tools": [_INVENTORY], "prompt": "How many monitors do we have in stock?",
     "expect": {"name": "get_inventory", "args": {"product": "monitor"}}},
    {"id": "currency", "tools": [_CURRENCY], "prompt": "Convert 250 US dollars to euros.",
     "expect": {"name": "convert_currency", "args": {"amount": 250, "from_currency": "USD", "to_currency": "EUR"}}},
    {"id": "calendar", "tools": [_CAL],
     "prompt": "Schedule a 45 minute meeting titled 'Design review' on 2026-11-03.",
     "expect": {"name": "create_calendar_event",
                "args": {"title": "Design review", "date": "2026-11-03", "duration_minutes": 45}}},
    {"id": "flights", "tools": [_FLIGHTS], "prompt": "Find flights from SFO to JFK on 2026-12-15 for 2 passengers.",
     "expect": {"name": "search_flights",
                "args": {"origin": "SFO", "destination": "JFK", "date": "2026-12-15", "passengers": 2}}},
    {"id": "tip", "tools": [_TIP], "prompt": "What is an 18% tip on a $64.50 bill?",
     "expect": {"name": "calculate_tip", "args": {"bill_amount": 64.5, "tip_percent": 18}}},
    {"id": "email", "tools": [_EMAIL],
     "prompt": "Email alice@example.com with the subject 'Quarterly report' telling her the report is attached.",
     "expect": {"name": "send_email", "args": {"to": "alice@example.com", "subject": "Quarterly report",
                                                "body": {"nonempty": True}}}},
    {"id": "choose_news", "tools": [_STOCK, _NEWS], "prompt": "Show me the 3 latest news items about NVIDIA (NVDA).",
     "expect": {"name": "get_company_news", "args": {"symbol": "NVDA", "limit": 3}}},
    {"id": "thermostat", "tools": [_THERMO], "prompt": "Set the thermostat to 21.5 degrees in heat mode.",
     "expect": {"name": "set_thermostat", "args": {"temperature": 21.5, "mode": "heat"}}},
    {"id": "order", "tools": [_ORDER], "prompt": "Look up order A-10423 and include the line items.",
     "expect": {"name": "lookup_order", "args": {"order_id": "A-10423", "include_items": True}}},
    {"id": "no_tool_needed", "tools": [_WEATHER, _STOCK], "prompt": "What is 12 plus 30? Just answer the question.",
     "expect": {"name": None}},
]


# ----------------------------------------------------------------------------- images
def _font(size: int):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def make_images() -> dict[str, bytes]:
    """Deterministic procedural test images (PIL, bundled default font)."""
    from PIL import Image, ImageDraw
    out: dict[str, bytes] = {}

    img = Image.new("RGB", (768, 320), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([60, 80, 220, 240], fill=(220, 30, 30))
    d.rectangle([304, 80, 464, 240], fill=(30, 60, 220))
    d.polygon([(628, 80), (548, 240), (708, 240)], fill=(30, 170, 60))
    out["shapes"] = _png(img)

    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    rng = random.Random(7)
    cells = rng.sample([(c, r) for c in range(4) for r in range(3)], 7)
    for c, r in cells:
        x, y = 40 + c * 150 + rng.randint(0, 30), 30 + r * 150 + rng.randint(0, 30)
        d.ellipse([x, y, x + 90, y + 90], fill=(240, 140, 20), outline="black", width=3)
    out["count_circles"] = _png(img)

    img = Image.new("RGB", (720, 260), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 60), "73914", fill="black", font=_font(140))
    out["digits"] = _png(img)

    img = Image.new("RGB", (720, 260), (235, 240, 250))
    d = ImageDraw.Draw(img)
    d.text((60, 70), "NEBULA", fill=(90, 20, 120), font=_font(120))
    out["word"] = _png(img)

    img = Image.new("RGB", (800, 560), "white")
    d = ImageDraw.Draw(img)
    vals = {"A": 3, "B": 7, "C": 5, "D": 9, "E": 2}
    f = _font(30)
    for i, (k, v) in enumerate(vals.items()):
        x0 = 110 + i * 130
        top = 480 - v * 42
        d.rectangle([x0, top, x0 + 80, 480], fill=(60, 110, 200))
        d.text((x0 + 28, 490), k, fill="black", font=f)
        d.text((x0 + 30, top - 38), str(v), fill="black", font=f)
    d.line([80, 480, 780, 480], fill="black", width=3)
    d.line([80, 40, 80, 480], fill="black", width=3)
    out["bar_chart"] = _png(img)

    img = Image.new("RGB", (768, 320), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([90, 90, 230, 230], fill=(30, 60, 220))
    d.rectangle([540, 90, 680, 230], fill=(220, 30, 30))
    out["position"] = _png(img)
    return out


IMAGE_CASES: list[dict[str, Any]] = [
    {"id": "circle_color", "image": "shapes", "kind": "word", "expected": "red",
     "alternatives": ["blue", "green"], "prompt": "What color is the circle? Reply with one word."},
    {"id": "triangle_color", "image": "shapes", "kind": "word", "expected": "green",
     "alternatives": ["red", "blue"], "prompt": "What color is the triangle? Reply with one word."},
    {"id": "count_circles", "image": "count_circles", "kind": "number", "expected": "7",
     "prompt": "How many circles are in this image? Reply with the number only."},
    {"id": "digits", "image": "digits", "kind": "number", "expected": "73914",
     "prompt": "What number is written in the image? Reply with the number only."},
    {"id": "word", "image": "word", "kind": "word", "expected": "nebula",
     "prompt": "What word is written in the image? Reply with the word only."},
    {"id": "tallest_bar", "image": "bar_chart", "kind": "letter", "expected": "D",
     "prompt": "Which letter labels the tallest bar in this chart? Reply with the letter only."},
    {"id": "bar_value", "image": "bar_chart", "kind": "number", "expected": "7",
     "prompt": "What value is shown above bar B? Reply with the number only."},
    {"id": "position", "image": "position", "kind": "word", "expected": "right", "alternatives": ["left"],
     "prompt": "Is the red square to the left or to the right of the blue circle? Reply with one word: left or right."},
]


def image_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


# ----------------------------------------------------------------------------- needle
NEEDLE_DEPTHS = [0.1, 0.5, 0.9]


def needle_case(length: int, depth: float, idx: int) -> dict[str, Any]:
    rng = random.Random(9000 + idx)
    vault = rng.choice(["Kestrel", "Marlowe", "Juniper", "Halcyon", "Osprey", "Tamarind", "Vesper", "Wren"])
    code = f"{rng.randint(10, 99)}-{rng.choice(['AMBER', 'COBALT', 'SAFFRON', 'TEAL', 'UMBER'])}-{rng.randint(1000, 9999)}"
    return {"id": f"needle-{length}-{int(depth * 100)}", "length": length, "depth": depth, "seed": 7000 + idx,
            "needle": f"The access code for the {vault} vault is {code}.",
            "question": (f"According to the documents above, what is the access code for the {vault} vault? "
                         "Reply with the access code only."),
            "expected": code}


def insert_needle(body: str, needle: str, depth: float) -> str:
    """Insert the needle at the sentence boundary nearest to depth * len(body)."""
    pos = int(len(body) * depth)
    cut = body.rfind(". ", 0, max(pos, 1))
    cut = cut + 2 if cut >= 0 else pos
    return body[:cut] + needle + " " + body[cut:]


# ----------------------------------------------------------------------------- determinism
DETERMINISM_PROMPTS: list[dict[str, str]] = [
    {"id": "fridge", "prompt": "Explain in one short paragraph how a refrigerator keeps food cold."},
    {"id": "palindrome", "prompt": "Write a Python function that checks whether a string is a palindrome, with a docstring."},
    {"id": "mars", "prompt": "List five facts about the planet Mars as bullet points."},
    {"id": "french", "prompt": "Translate into French: 'The library opens at nine in the morning and closes at six in the evening.'"},
    {"id": "lighthouse", "prompt": "Write a four-sentence story about a lighthouse keeper who finds a message in a bottle."},
    {"id": "tcp_udp", "prompt": "What are the main differences between TCP and UDP? Answer concisely."},
    {"id": "sql", "prompt": "Write a SQL query that returns the ten customers with the highest total order value."},
    {"id": "arith", "prompt": "Compute 37 * 43 step by step."},
]

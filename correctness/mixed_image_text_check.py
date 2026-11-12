# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Mixed image + text check (written 2026-10-22 for the pcg-textonly patch: prefill CUDA graphs for text-only batches).

usage: mixed_image_text_check.py <base_url> [rounds]   (needs Pillow)

Exercises the paths the patch splits between graph and eager prefill:
  followup   image turn, then a text follow-up in the same conversation (image prefix reused from cache, so the new
             chunk holds no image tokens) with a ~1.5K-token filler and a code to recall (graph bucket > 128 tokens).
  concurrent one image request and two text-only requests fired together, `rounds` times (mixed batches: image
             chunks must stay eager, text-only batches may replay graphs).
  text_only  short greedy text prompts alone (graph path).
Every case checks required substrings in the answer. Prints one JSON line per case and a final summary line.
"""
import base64
import io
import json
import sys
import threading
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

base = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def img_url(im):
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def shapes():
    im = Image.new("RGB", (512, 512), "white"); d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 200, 200), fill="red"); d.rectangle((280, 60, 460, 220), fill="blue")
    d.polygon([(256, 300), (150, 470), (362, 470)], fill="green")
    return img_url(im)


def invoice(n):
    im = Image.new("RGB", (512, 256), "white"); d = ImageDraw.Draw(im)
    d.text((30, 90), f"INVOICE {n}", fill="black", font=font(56))
    return img_url(im)


def chat(messages, max_tokens=128):
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        return (r["choices"][0]["message"]["content"] or ""), r.get("usage", {}), round(time.time() - t0, 2), None
    except Exception as e:  # noqa: BLE001
        return "", {}, round(time.time() - t0, 2), repr(e)[:300]


results = []


def record(case, ans, must, usage, lat, err, any_of=None):
    ok = err is None and all(m.lower() in ans.lower() for m in must)
    if any_of:
        ok = ok and any(m.lower() in ans.lower() for m in any_of)
    results.append(ok)
    print(json.dumps({"case": case, "pass": ok, "latency_s": lat, "prompt_tokens": usage.get("prompt_tokens"),
                      "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "answer": ans.strip()[:160], "error": err}), flush=True)


# 1. image turn + text follow-up in the same conversation
q1 = "Name the three shapes in this image and their colors."
m1 = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": shapes()}}, {"type": "text", "text": q1}]}]
a1, u, lat, err = chat(m1)
record("followup_turn1_image", a1, ["red", "blue", "green"], u, lat, err)
filler = " ".join(f"Note {i}: the weather log for day {i} was unremarkable and nothing changed." for i in range(110))
q2 = (filler + "\nThe secret code is 4417.\n"
      "Question: what is the secret code, and what color was the triangle in the image? Answer in one short line.")
m2 = m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": q2}]
a2, u, lat, err = chat(m2)
record("followup_turn2_text", a2, ["4417", "green"], u, lat, err)
q3 = "Which shape in the image was blue? One word."
m3 = m2 + [{"role": "assistant", "content": a2}, {"role": "user", "content": q3}]
a3, u, lat, err = chat(m3, max_tokens=16)
record("followup_turn3_text", a3, [], u, lat, err, any_of=["rectangle", "square"])

# 2. image and text requests fired together
for r in range(rounds):
    n = 7342 + r
    jobs = [
        ("concurrent_image", [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": invoice(n)}},
            {"type": "text", "text": "What text is written in this image? Reply with the text only."}]}], [str(n)], None),
        ("concurrent_text_math", [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
         ["391"], None),
        ("concurrent_text_capital", [{"role": "user", "content": "Name the capital of Australia in one word."}],
         ["canberra"], None),
    ]
    out = [None] * len(jobs)

    def run(i, msgs):
        out[i] = chat(msgs)
    th = [threading.Thread(target=run, args=(i, j[1])) for i, j in enumerate(jobs)]
    [t.start() for t in th]
    [t.join() for t in th]
    for (case, _, must, any_of), (ans, u, lat, err) in zip(jobs, out):
        record(f"{case}_r{r}", ans, must, u, lat, err, any_of)

# 3. text-only prompts alone
for case, msg, must in (("text_only_math", "What is 12 * 12? Answer with just the number.", ["144"]),
                        ("text_only_prime", "List the first five prime numbers separated by commas.", ["2", "3", "5", "7", "11"])):
    ans, u, lat, err = chat([{"role": "user", "content": msg}])
    record(case, ans, must, u, lat, err)

print(json.dumps({"passed": sum(results), "total": len(results)}))

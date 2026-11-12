# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Phase 1 image-understanding check: three procedurally drawn images with known answers.

usage: image_check.py <base_url>   (needs Pillow)
"""
import base64
import io
import json
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

base = sys.argv[1]
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def shapes():
    im = Image.new("RGB", (512, 512), "white"); d = ImageDraw.Draw(im)
    d.ellipse((40, 40, 200, 200), fill="red"); d.rectangle((280, 60, 460, 220), fill="blue")
    d.polygon([(256, 300), (150, 470), (362, 470)], fill="green")
    return im, "Name the three shapes in this image and their colors.", ["red", "blue", "green", "circle", "triangle"]


def text():
    im = Image.new("RGB", (512, 256), "white"); d = ImageDraw.Draw(im)
    d.text((30, 90), "INVOICE 7342", fill="black", font=font(56))
    return im, "What text is written in this image? Reply with the text only.", ["7342"]


def bars():
    im = Image.new("RGB", (512, 384), "white"); d = ImageDraw.Draw(im)
    vals = {"A": 3, "B": 9, "C": 5}
    for i, (k, v) in enumerate(vals.items()):
        x = 80 + i * 140
        d.rectangle((x, 340 - v * 30, x + 80, 340), fill="gray")
        d.text((x + 30, 345), k, fill="black", font=font(28))
    return im, "This is a bar chart with bars labeled A, B and C. Which bar is tallest? Reply with the letter only.", ["B"]


ok = 0
for make in (shapes, text, bars):
    im, q, must = make()
    buf = io.BytesIO(); im.save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"model": model, "temperature": 0.0, "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": q}]}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        ans = r["choices"][0]["message"]["content"] or ""
        err = None
    except Exception as e:
        ans, err = "", repr(e)[:300]
    passed = all(m.lower() in ans.lower() for m in must)
    ok += passed
    print(json.dumps({"case": make.__name__, "pass": passed, "latency_s": round(time.time() - t0, 2),
                      "answer": ans.strip()[:160], "error": err}), flush=True)
print(json.dumps({"passed": ok, "total": 3}))

# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Mamba-2 in_proj / out_proj: static FP8 -> NVFP4 (W4A4), made from the checkpoint's own FP8 weights, written as a new
checkpoint directory next to the original. A model change, kept only if the reasoning-on
quality gate holds.

NVFP4 as ModelOpt writes it (the layout of the checkpoint's NVFP4 experts, which TRT-LLM loads as is):
  weight_scale_2  g = amax(W) / (6 * 448)                       FP32 per-tensor
  weight_scale    s = E4M3(amax(16-element block) / 6 / g)      [N, K/16], linear layout (TRT interleaves on load)
  weight          nearest E2M1(W / (s * g)), two codes a byte, even element in the low nibble, sign in bit 3
  input_scale     amax(x) / (6 * 448) = the FP8 input_scale / 6 (the FP8 one is amax(x) / 448: same calibration)
Checks before anything is written: the codec reproduces TRT-LLM's own fp4_quantize codes on random data and an
existing NVFP4 tensor of the checkpoint (scales and codes of every block whose largest code is 6), the targets are
static FP8 with scalar input scales, and the disk has room.
TRT-LLM reads every *.safetensors in the directory concurrently (no index), so a converted tensor cannot just sit in
a second file. --mode overlay (default): DST symlinks every file of SRC and holds the converted tensors in
nhd-m4-overlay.st, which patches/weight-overlay.diff (NHD_WEIGHT_OVERLAY) applies after the shards: ~2 GB instead of a
70 GB copy. --mode full: a standalone checkpoint, the shards holding a target rewritten whole, the others symlinked.
Either way the targets' quantized_layers entries become the NVFP4 entry the checkpoint already uses
(hf_quant_config.json, config.json, quant_cfg.json, wherever a quantized_layers dict is).
Usage: python3 m4_convert.py SRC DST [--targets in_proj,out_proj] [--mode overlay|full] [--dry]   RESULT convert lines."""
import argparse
import json
import os
import re
import shutil
import statistics
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--targets", default="in_proj,out_proj")
ap.add_argument("--mode", choices=("overlay", "full"), default="overlay")
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
t0 = time.time()
R = lambda *x: print("RESULT convert", *x, flush=True)  # noqa: E731
dev = "cuda" if torch.cuda.is_available() else "cpu"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
JSONS = ("hf_quant_config.json", "config.json", "quant_cfg.json")
INDEX = "model.safetensors.index.json"
OVERLAY = "nhd-m4-overlay.st"


def fail(msg):
    R(f"FAILED: {msg}")
    sys.exit(1)


def quant(w, g):
    """w float32 [N, K] on dev, g float32 0-d -> (packed uint8 [N, K/2], E4M3 block scales [N, K/16]).
    Rounds like the hardware cvt.rn.satfinite.e2m1x2.f32 that TRT-LLM's fp4_quantize uses: nearest, ties to the even
    code, and the sign bit kept for values that round to zero (-0 = code 8), as the checkpoint's own NVFP4 tensors."""
    n, k = w.shape
    b = w.reshape(n, k // 16, 16)
    s = (b.abs().amax(-1) / 6 / g).clamp(max=448).to(torch.float8_e4m3fn)
    x = b / (s.float().clamp(min=2 ** -9)[..., None] * g)
    v = x.abs()
    mag = torch.bucketize(v, MIDS)  # nearest E2M1 magnitude; an exact midpoint lands on the smaller code
    mag = mag + ((mag % 2 == 1) & (v == MIDS[mag.clamp(max=6)])).long()  # ties to the even code
    code = (mag | (torch.signbit(x).long() << 3)).reshape(n, k).to(torch.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4), s


def codes(packed):
    return torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).long()


def dequant(packed, s, g):
    c = codes(packed)
    v = E2M1[c & 7] * (1 - 2 * (c >> 3)).float()
    n = c.shape[0]
    return (v.reshape(n, -1, 16) * s.float()[..., None] * g).reshape(n, -1)


def rel_err(w, g):
    p, s = quant(w, g)
    return ((dequant(p, s, g) - w).norm() / w.norm()).item()


src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
if not os.path.exists(os.path.join(src, INDEX)):
    fail(f"no {INDEX} in {src}")
index = json.load(open(os.path.join(src, INDEX)))
wmap = index["weight_map"]
handles = {}


def h(k):
    f = wmap[k]
    if f not in handles:
        handles[f] = safe_open(os.path.join(src, f), "pt", device="cpu")
    return handles[f]


def dt(k):
    return h(k).get_slice(k).get_dtype()


def shp(k):
    return list(h(k).get_slice(k).get_shape())


def get(k):
    return h(k).get_tensor(k)


want = [t.strip() for t in a.targets.split(",") if t.strip()]
pat = re.compile(r"(^|\.)layers\.(\d+)\.mixer\.(" + "|".join(map(re.escape, want)) + r")\.weight$")
lid = lambda m: (int(re.search(r"layers\.(\d+)\.", m).group(1)), m)  # noqa: E731
mods = sorted({k[:-len(".weight")] for k in wmap if pat.search(k) and "mtp" not in k}, key=lid)
if not mods:
    fail(f"no keys match {pat.pattern}")
kinds = {}
for m in mods:
    kinds.setdefault(dt(m + ".weight"), []).append(m)
targets = kinds.get("F8_E4M3", [])
R(f"{len(mods)} modules match ({', '.join(f'{k} x{len(v)}' for k, v in kinds.items())}); converting {len(targets)}")
if not targets:
    fail("no FP8 targets left (already converted?)")
done = os.path.join(dst, ".m4done")
if not a.dry and os.path.exists(done) and json.load(open(done)).get("targets") == targets \
        and json.load(open(done)).get("mode") == a.mode:
    R(f"OK (already converted at {dst})")
    sys.exit(0)
m0 = targets[0]
for k in (m0 + ".weight", m0 + ".weight_scale", m0 + ".input_scale"):
    if k not in wmap:
        fail(f"{k} missing: not a static-FP8 Linear")
for t in want:
    mt = next((m for m in targets if m.endswith("." + t)), None)
    if mt:
        R(f"{t} e.g. {mt}: " + "; ".join(f"{s} {dt(mt + s)} {shp(mt + s)}" for s in (".weight", ".weight_scale", ".input_scale")
                                        if mt + s in wmap))
R(f"first target {m0}: " + "; ".join(f"{s} {dt(m0 + s)} {shp(m0 + s)}" for s in (".weight", ".weight_scale", ".input_scale")))

# Reference: an NVFP4 module already in the checkpoint (dtypes and shapes of its scalars are copied).
ref = next((k[:-len(".weight_scale_2")] for k in sorted(wmap) if k.endswith(".weight_scale_2")
            and dt(k[:-len("_scale_2")]) == "U8" and len(shp(k[:-len("_scale_2")])) == 2), None)
if ref is None:
    fail("no 2-D NVFP4 reference module (*.weight_scale_2 with a U8 weight)")
if dt(ref + ".weight_scale") != "F8_E4M3":
    fail(f"reference block scales are {dt(ref + '.weight_scale')}, not F8_E4M3")
R(f"reference NVFP4 {ref}: " + "; ".join(f"{s} {dt(ref + s)} {shp(ref + s)}" for s in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
                                       if ref + s in wmap))
ws2_ref, is_ref = get(ref + ".weight_scale_2"), get(ref + ".input_scale") if ref + ".input_scale" in wmap else None

# Check 1: codes equal TRT-LLM's fp4_quantize (the activation path the GEMM pairs with the weights) on random data.
trt_ok = None
try:
    import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
    torch.manual_seed(0)
    x = (torch.randn(512, 4096, device="cuda") * torch.rand(512, 1, device="cuda") * 3).to(torch.bfloat16)
    gs = (448 * 6 / x.float().abs().amax()).reshape(1)
    p_trt = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)[0]
    p_me = quant(x.float(), 1 / gs.reshape(()))[0]
    ct, cm = codes(p_trt.view(torch.uint8)), codes(p_me)
    match = (ct == cm).float().mean().item()
    mval = (torch.where(ct & 7 == 0, 0, ct) == torch.where(cm & 7 == 0, 0, cm)).float().mean().item()
    trt_ok = match > 0.999
    R(f"codec vs TRT-LLM fp4_quantize on random data: {match * 100:.3f}% of codes equal, "
      f"{mval * 100:.3f}% of values (+0 = -0)")
except Exception as e:  # noqa: BLE001
    R(f"codec vs TRT-LLM fp4_quantize: not run ({type(e).__name__}: {str(e)[:150]})")
# Check 2: the checkpoint's own NVFP4 tensor survives dequant -> quant with its global scale.
P, S = get(ref + ".weight").to(dev), get(ref + ".weight_scale").to(dev)
G = ws2_ref.float().to(dev).reshape(())
P2, S2 = quant(dequant(P, S, G), G)
c1, c2 = codes(P), codes(P2)
full = (c1.reshape(c1.shape[0], -1, 16) & 7).amax(-1) == 7  # blocks whose largest code is 6
same_c = (c1 == c2).reshape(c1.shape[0], -1, 16).all(-1)
same_s = S.view(torch.uint8) == S2.view(torch.uint8)
rt = (same_c & same_s)[full].float().mean().item()
R(f"codec round trip on {ref}.weight: {full.float().mean().item() * 100:.2f}% of blocks reach code 6, "
  f"{rt * 100:.4f}% of those exact (scales and codes); all blocks {(same_c & same_s).float().mean().item() * 100:.3f}%")
# How the checkpoint itself scaled its NVFP4 blocks (information only; the GEMM takes any 1x16 block scales).
amax_s = (dequant(P, S, G).reshape(P.shape[0], -1, 16).abs().amax(-1) / 6 / G).clamp(max=448).to(torch.float8_e4m3fn)
rows16 = S.shape[0] // 16 * 16
tiles = S.view(torch.uint8)[:rows16].reshape(-1, 16, S.shape[1])
R(f"checkpoint NVFP4 scales on {ref}: {(amax_s.view(torch.uint8) == S.view(torch.uint8)).float().mean().item() * 100:.2f}% "
  f"of blocks = E4M3(block amax / 6 / g); {(tiles == tiles[:, :1]).all(1).float().mean().item() * 100:.2f}% of 16-row "
  f"column groups share one scale (2D 16x16 tiles); codes with the -0 sign bit {((c1 == 8).float().mean().item() * 100):.2f}%")
if rt < 0.9999 or trt_ok is False:
    fail("codec does not reproduce the checkpoint / TRT-LLM NVFP4 layout")
gw = torch.randn(4096, 4096, device=dev)
R(f"NVFP4 error on a Gaussian 4096x4096 (reference): {rel_err(gw, gw.abs().amax() / 2688):.4f}")

# quantized_layers entries to change, per json file.
edits = {}
for name in JSONS:
    p = os.path.join(src, name)
    if not os.path.exists(p):
        continue
    doc = json.load(open(p))
    stack, qls = [doc], []
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                (qls.append(v) if k == "quantized_layers" and isinstance(v, dict) else stack.append(v))
        elif isinstance(o, list):
            stack.extend(o)
    if not qls:
        continue
    for ql in qls:
        tmpl = next((dict(v) for v in ql.values() if isinstance(v, dict) and v.get("quant_algo") == "NVFP4"),
                    {"quant_algo": "NVFP4", "group_size": 16})
        for m in targets:
            hits = [k for k in ql if k == m or m.endswith("." + k) or k.endswith("." + m)]
            if len(hits) != 1:
                fail(f"{name}: {len(hits)} quantized_layers keys for {m} (e.g. {list(ql)[:3]})")
            if m == targets[0]:
                R(f"{name}: {hits[0]} {json.dumps(ql[hits[0]])} -> {json.dumps(tmpl)}")
            ql[hits[0]] = dict(tmpl)
    edits[name] = doc
if not edits:
    fail("no quantized_layers dict in " + "/".join(JSONS))

# Convert.
new, errs, isc = {}, [], []
for m in targets:
    w8 = get(m + ".weight").to(dev)
    ws = get(m + ".weight_scale").float().to(dev)
    s_in = get(m + ".input_scale").float()
    if ws.numel() == 1:
        W = w8.float() * ws.reshape(())
    elif ws.numel() == w8.shape[0]:
        W = w8.float() * ws.reshape(-1, 1)
    else:
        fail(f"{m}.weight_scale has {ws.numel()} elements for {w8.shape[0]} rows")
    if s_in.numel() != 1:
        fail(f"{m}.input_scale is not a scalar ({list(s_in.shape)})")
    if W.shape[1] % 16:
        fail(f"{m}: in_features {W.shape[1]} not a multiple of 16")
    g = (W.abs().amax() / (6 * 448)).clamp(min=1e-12)
    p, s = quant(W, g)
    errs.append(((dequant(p, s, g) - W).norm() / W.norm()).item())
    isc.append(s_in.item())
    new[m + ".weight"] = p.cpu()
    new[m + ".weight_scale"] = s.cpu()
    new[m + ".weight_scale_2"] = g.reshape(ws2_ref.shape).to(ws2_ref.dtype).cpu()
    new[m + ".input_scale"] = (s_in / 6).reshape(is_ref.shape if is_ref is not None else s_in.shape).to(
        is_ref.dtype if is_ref is not None else torch.float32)
    del w8, W, p, s
for t in sorted(set(m.rsplit(".", 1)[1] for m in targets)):
    e = [x for m, x in zip(targets, errs) if m.endswith("." + t)]
    R(f"{t}: {len(e)} layers, FP4 vs FP8 weight error median {statistics.median(e):.4f}, max {max(e):.4f} "
      f"(layer {targets[errs.index(max(e))]})")
R(f"FP8 input_scale range {min(isc):.3e}..{max(isc):.3e} (NVFP4 input_scale = that / 6)")

# Plan the write.
wmap_new = dict(wmap)
for m in targets:
    wmap_new[m + ".weight_scale_2"] = wmap[m + ".weight"]
shards = sorted({wmap_new[k] for k in new}) if a.mode == "full" else []
need = (sum(os.path.getsize(os.path.join(src, f)) for f in shards) if a.mode == "full"
        else sum(t.numel() * t.element_size() for t in new.values()))
os.makedirs(os.path.dirname(dst), exist_ok=True)
free = shutil.disk_usage(os.path.dirname(dst)).free
R(f"mode {a.mode}: writes {need / 2**30:.2f} GiB ({len(shards)} of {len(set(wmap.values()))} shards rewritten), "
  f"free {free / 2**30:.1f} GiB")
if a.dry:
    R(f"dry run: nothing written ({time.time() - t0:.0f}s)")
    sys.exit(0)
if os.path.exists(dst):
    shutil.rmtree(dst)  # a partial earlier run: only its own files and symlinks
if free < need + 5 * 2**30:
    fail(f"not enough disk: need {need / 2**30:.1f} GiB + 5")
os.makedirs(dst)
for f in shards:
    with safe_open(os.path.join(src, f), "pt", device="cpu") as fh:
        meta = fh.metadata() or {"format": "pt"}
        out = {k: (new[k] if k in new else fh.get_tensor(k)) for k in fh.keys()}
    out.update({k: v for k, v in new.items() if wmap_new[k] == f and k not in out})
    save_file(out, os.path.join(dst, f + ".tmp"), metadata=meta)
    os.rename(os.path.join(dst, f + ".tmp"), os.path.join(dst, f))
    del out
if a.mode == "overlay":
    save_file(new, os.path.join(dst, OVERLAY + ".tmp"), metadata={"format": "pt"})
    os.rename(os.path.join(dst, OVERLAY + ".tmp"), os.path.join(dst, OVERLAY))
for entry in sorted(os.listdir(src)):
    if entry.startswith(".") or entry in shards or entry in edits or (entry == INDEX and a.mode == "full"):
        continue
    os.symlink(os.path.join(src, entry), os.path.join(dst, entry))
if a.mode == "full":
    index["weight_map"] = wmap_new
    json.dump(index, open(os.path.join(dst, INDEX), "w"), indent=2)
for name, doc in edits.items():
    json.dump(doc, open(os.path.join(dst, name), "w"), indent=2)
json.dump({"targets": targets, "mode": a.mode, "src": src, "errors": dict(zip(targets, errs))}, open(done, "w"))
R(f"OK {len(targets)} modules to NVFP4 ({a.mode}"
  f"{': ' + OVERLAY + ' + NHD_WEIGHT_OVERLAY' if a.mode == 'overlay' else f', {len(shards)} shards rewritten'}) "
  f"in {time.time() - t0:.0f}s -> {dst}")

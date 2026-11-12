#!/usr/bin/env python3
# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""Standalone repro: b12x NVFP4 W4A4 fused-MoE kernels vs an independent reference.

Synthetic ModelOpt-NVFP4 experts at Nemotron 3.5 Super VL's routed-expert shape
(hidden/latent 1024, intermediate 2688, 512 experts, top-22). Block-scale and
weight-global-scale magnitudes follow Super VL layer 1 (block scales median
~180, weight_scale_2 ~2e-5; input scales 0.002046 FC1 / 0.0340 FC2). Runs the
ReLU-squared (ungated) case and a gated-SiLU control.

Every available path is compared with one reference computed here from the
checkpoint-form tensors, using the arithmetic b12x's own moe_reference_nvfp4
documents for its micro kernel: per-16 E4M3 activation block scales from the
static input scale, E2M1 round-to-nearest-even, FP32 GEMMs, BF16 intermediate,
FP32 per-route accumulation, BF16 output.

Paths (each runs only where its API exists):
  planned      b12x planned API (b12x.moe.fused_moe / flashinfer.experimental.b12x),
               weights in the ModelOpt NVFP4 contract.
  wrapper-trt  flashinfer.B12xMoEWrapper with TensorRT-LLM 1.4's
               NVFP4CuteDslB12xFusedMoEMethod conversion (block scales multiplied
               by weight_scale_2 and re-cast to E4M3; alphas = input scale).
               This is TensorRT-LLM's CuteDslB12xFusedMoE decode call.
  wrapper-exact flashinfer.B12xMoEWrapper with unmodified block scales,
               w1_alpha = weight_scale_2 and input_global_scale (documented
               alternative), to separate kernel error from scale conversion.
  wrapper-trt[forced=X]  the functional b12x_fused_moe with moe_dispatch's
               test hook _FORCED_BACKEND = micro / static / dynamic.

Usage: python3 b12x_moe_repro.py [--rows 1,2,4,8,17,64] [--activations relu2,silu]
       [--paths planned,wrapper-trt,wrapper-exact,forced] [--experts 512]
Prints one line per (activation, path, rows): relative L2 error vs the reference,
the worst per-row relative error, and the backend that ran; then truth= (relative
L2 vs the same NVFP4 weights with unquantized FP32 activations, i.e. the total
activation-quantization error) and, where the planned API exists, or_<mode>=
(vs b12x's moe_reference_nvfp4 with quant_scale_math micro / dynamic_precise /
dynamic_fast).
"""

import argparse
import importlib
import importlib.metadata as metadata
import math
import sys
import time
import traceback

import torch

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
HIDDEN, INTER, TOPK = 1024, 2688, 22
IN_UP, IN_DOWN = 0.0020461310632526875, 0.0340401791036129


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item(), ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()


def swizzle(scale):
    """128x4 block-scale swizzle (FlashInfer/b12x NVFP4 layout); input [E, rows, cols]."""
    batch, rows, cols = scale.shape
    rp, cp = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = torch.zeros((batch, rp, cp), dtype=scale.dtype, device=scale.device)
    padded[:, :rows, :cols] = scale
    out = padded.reshape(batch, rp // 128, 4, 32, cp // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous()
    return out.reshape(batch, rp, cp)


# ---------------------------------------------------------------- synthetic weights
def make_experts(activation, experts, device, seed=5):
    g = torch.Generator(device=device).manual_seed(seed)
    gated = activation != "relu2"

    def projection(rows, cols):
        codes = torch.randint(0, 256, (experts, rows, cols // 2), dtype=torch.uint8, device=device, generator=g)
        # Block scales: log-normal around 180, clamped to E4M3's range (Super VL layer 1: median ~180, max 448).
        logs = torch.randn((experts, rows, cols // 16), device=device, generator=g) * .6 + math.log(180.)
        scales = logs.exp().clamp(.002, 448.).to(torch.float8_e4m3fn)
        ws2 = 2e-5 * (1 + .1 * torch.rand((experts,), device=device, generator=g))
        return codes, scales, ws2

    up = projection(INTER, HIDDEN)
    gate = projection(INTER, HIDDEN) if gated else None
    if gated:  # ModelOpt shares weight_scale_2 between gate and up for fused FC1.
        gate = (gate[0], gate[1], up[2])
    down = projection(HIDDEN, INTER)
    return dict(up=up, gate=gate, down=down, gated=gated, activation=activation, experts=experts)


def load_experts(checkpoint, layer, device):
    """Real ModelOpt-NVFP4 relu2 experts of one Nemotron-H MoE layer (e.g. Super VL)."""
    import json
    from pathlib import Path
    from safetensors import safe_open

    global IN_UP, IN_DOWN
    root = Path(checkpoint)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"language_model.backbone.layers.{layer}.mixer.experts."
    experts = 1 + max(int(n[len(prefix):].split(".")[0]) for n in index if n.startswith(prefix))
    names = [f"{prefix}{e}.{p}.{t}" for e in range(experts) for p in ("up_proj", "down_proj")
             for t in ("weight", "weight_scale", "weight_scale_2", "input_scale")]
    tensors = {}
    for shard in sorted({index[n] for n in names}):
        with safe_open(str(root / shard), "pt") as f:
            for n in names:
                if index[n] == shard:
                    tensors[n] = f.get_tensor(n)

    def stack(p, t):
        return torch.stack([tensors[f"{prefix}{e}.{p}.{t}"].reshape(-1) if t in ("weight_scale_2", "input_scale")
                            else tensors[f"{prefix}{e}.{p}.{t}"] for e in range(experts)]).to(device)

    inputs = {p: stack(p, "input_scale").float().reshape(-1) for p in ("up_proj", "down_proj")}
    if any(bool((v != v[0]).any()) for v in inputs.values()):
        raise ValueError("per-expert input scales differ; this repro assumes one per projection")
    IN_UP, IN_DOWN = inputs["up_proj"][0].item(), inputs["down_proj"][0].item()
    projections = {p: (stack(p, "weight"), stack(p, "weight_scale"), stack(p, "weight_scale_2").float().reshape(-1))
                   for p in ("up_proj", "down_proj")}
    return dict(up=projections["up_proj"], gate=None, down=projections["down_proj"], gated=False,
                activation="relu2", experts=experts)


def dequant(codes, scales, ws2):
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=codes.device)
    values = lut[torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()]
    rows, cols = values.shape
    return (values.view(rows, cols // 16, 16) * scales.float()[..., None] * ws2).view(rows, cols)


def round_e2m1(values):
    magnitude = values.abs().clamp(max=6)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=values.device)
    below = torch.searchsorted(bounds, magnitude.contiguous(), right=False)
    tie = magnitude == bounds[below.clamp(max=6)]
    codes = below + (tie & (below % 2 == 1)).int()
    lut = torch.tensor(FP4_VALUES[:8], dtype=torch.float32, device=values.device)
    return lut[codes] * values.sign()


def quantize_activation(values, input_scale):
    """NVFP4 activation QDQ with a static input scale (b12x 'micro' scale math)."""
    gs = torch.tensor(1.0 / input_scale, dtype=torch.float32, device=values.device)
    groups = values.float().reshape(*values.shape[:-1], -1, 16)
    raw = (groups.abs().amax(-1, keepdim=True).double() * gs.double() / 6.0).float()
    scale = raw.clamp(max=448.).to(torch.float8_e4m3fn).float()
    effective = (scale / gs).clamp(min=1e-30)
    # The micro kernel multiplies by the reciprocal; at E2M1 ties (frequent for BF16 inputs) division differs.
    return (round_e2m1(groups * effective.reciprocal()) * effective).reshape(values.shape)


def reference(w, x, ids, weights, quantize=True):
    """Independent FP32 reference of the checkpoint arithmetic.

    quantize=False keeps the activations unquantized (the same NVFP4 weights, FP32 activations):
    the "truth" against which each path's total activation-quantization error is measured."""
    out = torch.zeros((x.shape[0], HIDDEN), dtype=torch.float32, device=x.device)
    qdq = quantize_activation if quantize else (lambda values, scale: values.float())
    xq = qdq(x, IN_UP)
    for e in ids.unique().tolist():
        rows, cols = torch.where(ids == e)
        up = xq[rows] @ dequant(w["up"][0][e], w["up"][1][e], w["up"][2][e]).T
        if w["gated"]:
            gate = xq[rows] @ dequant(w["gate"][0][e], w["gate"][1][e], w["gate"][2][e]).T
            mid = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
        else:
            mid = torch.relu(up).square().to(torch.bfloat16).float()
        down = qdq(mid, IN_DOWN) @ dequant(w["down"][0][e], w["down"][1][e], w["down"][2][e]).T
        out.index_add_(0, rows, down * weights[rows, cols, None])
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------- paths
def b12x_module():
    for name in ("b12x.moe.fused_moe", "flashinfer.experimental.b12x.moe.fused_moe"):
        try:
            return name, importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
    return None, None


def planned_experts(w):
    """Prepare experts through the b12x planned API, as an NVFP4 model loader does."""
    name, moe = b12x_module()
    if moe is None:
        raise RuntimeError("no b12x planned fused_moe API in this environment")
    E = w["experts"]
    w13 = torch.cat((w["up"][0], w["gate"][0]), 1) if w["gated"] else w["up"][0]
    s13 = torch.cat((w["up"][1], w["gate"][1]), 1) if w["gated"] else w["up"][1]
    plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(mode="a4", nonlinearity=w["activation"], io_dtype=torch.bfloat16),
        geometry=moe.MoEGeometry(num_experts=E, hidden_size=HIDDEN, intermediate_size=INTER))
    device = w13.device
    packed = moe.PackedWeights(
        w13=w13.contiguous(), w2=w["down"][0].contiguous(),
        w13_block_scales=swizzle(s13), w2_block_scales=swizzle(w["down"][1]),
        w13_global_scales=w["up"][2].float().contiguous(), w2_global_scales=w["down"][2].float().contiguous(),
        input_scale=torch.full((E,), 1 / IN_UP, device=device), intermediate_scale=torch.full((E,), 1 / IN_DOWN, device=device),
        immutable_input_scales=True)
    return name, moe, moe.prepare_weights(plan=plan, weights=packed)


def run_planned(moe, experts, x, ids, weights):
    from importlib import import_module
    prep = import_module(moe.__name__.replace("moe.fused_moe", "preparation"))
    try:
        workloads = import_module(moe.__name__ + ".workloads")
        invocation = prep.FrozenMapping({"tuning_route_pattern": workloads.TUNING_WORKLOAD_VERSION,
                                         "tuning_route_source": "b12x"})
    except Exception:  # noqa: BLE001
        invocation = prep.FrozenMapping({})
    out = torch.empty_like(x)
    plan = moe.plan_execution(experts=experts, capacity=moe.ExecutionCapacity(
        max_tokens=x.shape[0], top_k=ids.shape[1], route_num_experts=0), invocation=invocation)

    def scratch(specs):
        (spec,) = tuple(specs)
        return torch.empty(spec.nbytes, dtype=torch.uint8, device=x.device)

    def prepare_call(state):
        buffer = scratch(state.scratch.scratch_specs())
        binding = state.bind(a=x, topk_ids=torch.zeros_like(ids), topk_weights=torch.full_like(weights, 1 / ids.shape[1]),
                             output=out, scratch=buffer, input_scales_static=True)
        return prep.PreparedCall(run=lambda: state.run(binding), owners=(buffer, binding))

    with prep.PreparationSession(device=x.device, autotune=False) as session:
        session.prepare((plan.request(name=f"moe{x.shape[0]}", prepare_call=prepare_call),))
        binding = moe.bind(plan, a=x, topk_ids=ids, topk_weights=weights, output=out,
                           scratch=scratch(plan.scratch_specs()), input_scales_static=True)
        moe.run(binding=binding)
        torch.cuda.synchronize()
        return out.clone(), getattr(binding, "implementation", "?")


def trtllm_weights(w):
    """TensorRT-LLM 1.4 NVFP4CuteDslB12xFusedMoEMethod.process_weights_after_loading (W4A4 branch)."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    E = w["experts"]
    # TRT-LLM keeps fc31_input_scale = 1/max(input_scale) and fc31_alpha = weight_scale_2 * max(input_scale),
    # recovers weight_scale_2 = fc31_alpha * fc31_input_scale, and multiplies the swizzled block scales by it.
    w13 = torch.cat((w["up"][0], w["gate"][0]), 1) if w["gated"] else w["up"][0]
    s13 = torch.cat((w["up"][1], w["gate"][1]), 1) if w["gated"] else w["up"][1]
    normalized = swizzle(s13).float()
    s13 = normalized * w["up"][2].view(-1, 1, 1)
    s2 = swizzle(w["down"][1]).float() * w["down"][2].view(-1, 1, 1)
    cast = s13.to(torch.float8_e4m3fn).float()
    live = normalized > 0
    stats = dict(zeroed=((cast == 0) & live).float().sum().item() / live.float().sum().item(),
                 scale_error=((cast - s13).abs() / s13.clamp_min(1e-30))[live].mean().item())
    out = dict(
        w1_weight=w13.contiguous(),
        w1_weight_sf=convert_sf_to_mma_layout(s13.to(torch.float8_e4m3fn).reshape(-1, s13.shape[-1]), m=w13.shape[1], k=HIDDEN, num_groups=E),
        w1_alpha=torch.full((E,), IN_UP, dtype=torch.float32, device=w13.device),
        w2_weight=w["down"][0].contiguous(),
        w2_weight_sf=convert_sf_to_mma_layout(s2.to(torch.float8_e4m3fn).reshape(-1, s2.shape[-1]), m=HIDDEN, k=INTER, num_groups=E),
        w2_alpha=torch.full((E,), IN_DOWN, dtype=torch.float32, device=w13.device),
        fc2_input_scale=torch.tensor(IN_DOWN, dtype=torch.float32, device=w13.device))
    return out, stats


def exact_weights(w):
    """Unmodified ModelOpt block scales; w1_alpha carries weight_scale_2, input_global_scale the input scale."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    E = w["experts"]
    w13 = torch.cat((w["up"][0], w["gate"][0]), 1) if w["gated"] else w["up"][0]
    s13 = torch.cat((w["up"][1], w["gate"][1]), 1) if w["gated"] else w["up"][1]
    device = w13.device
    return dict(
        w1_weight=w13.contiguous(),
        w1_weight_sf=convert_sf_to_mma_layout(swizzle(s13).reshape(-1, s13.shape[-1]), m=w13.shape[1], k=HIDDEN, num_groups=E),
        w1_alpha=w["up"][2].float().contiguous(),
        input_global_scale=torch.full((E,), IN_UP, dtype=torch.float32, device=device),
        w2_weight=w["down"][0].contiguous(),
        w2_weight_sf=convert_sf_to_mma_layout(swizzle(w["down"][1]).reshape(-1, w["down"][1].shape[-1]), m=HIDDEN, k=INTER, num_groups=E),
        w2_alpha=(w["down"][2] * IN_DOWN).float().contiguous(),
        fc2_input_scale=torch.tensor(IN_DOWN, dtype=torch.float32, device=device))


def run_wrapper(w, prepared, x, ids, weights):
    import flashinfer
    wrapper = flashinfer.B12xMoEWrapper(num_experts=w["experts"], top_k=ids.shape[1], hidden_size=HIDDEN,
        intermediate_size=INTER, use_cuda_graph=False, max_num_tokens=max(64, x.shape[0]),
        activation=w["activation"], quant_mode="nvfp4")
    out = wrapper.run(x=x, token_selected_experts=ids.int(), token_final_scales=weights, **prepared)
    torch.cuda.synchronize()
    dispatch = importlib.import_module("flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch")
    backend = dispatch.select_sm120_moe_backend(num_tokens=x.shape[0], num_topk=ids.shape[1])
    return out.clone(), f"auto({backend} workspace)"


def run_forced(w, prepared, x, ids, weights, backend):
    import flashinfer
    dispatch = importlib.import_module("flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch")
    previous = dispatch._FORCED_BACKEND
    dispatch._FORCED_BACKEND = backend
    try:
        out = flashinfer.fused_moe.b12x_fused_moe(x, num_experts=w["experts"], top_k=ids.shape[1],
            token_selected_experts=ids.int(), token_final_scales=weights, activation=w["activation"],
            quant_mode="nvfp4", **prepared)
        torch.cuda.synchronize()
    finally:
        dispatch._FORCED_BACKEND = previous
    return out.clone(), backend


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", default="1,2,4,8,17,64")
    parser.add_argument("--activations", default="relu2,silu")
    parser.add_argument("--paths", default="planned,wrapper-trt,wrapper-exact,forced")
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint", help="use one real MoE layer's relu2 experts from this ModelOpt NVFP4 checkpoint")
    parser.add_argument("--layer", type=int, default=1)
    args = parser.parse_args()
    rows = [int(r) for r in args.rows.split(",")]
    paths = set(args.paths.split(","))
    device = torch.device("cuda", 0)
    print(f"python {sys.version.split()[0]} torch {torch.__version__} cuda {torch.version.cuda} "
          f"device {torch.cuda.get_device_name(device)} cc {torch.cuda.get_device_capability(device)}")
    print("packages: " + ", ".join(f"{n}={version(n)}" for n in (
        "flashinfer-python", "b12x", "nvidia-cutlass-dsl", "tensorrt_llm", "apache-tvm-ffi")))
    print(f"shape: hidden {HIDDEN} intermediate {INTER} experts {args.experts} top-{TOPK}; "
          f"input scales FC1 {IN_UP:.6g} FC2 {IN_DOWN:.6g}; weight_scale_2 ~2e-5; block scales ~180")
    for activation in (["relu2"] if args.checkpoint else args.activations.split(",")):
        if args.checkpoint:
            w = load_experts(args.checkpoint, args.layer, device)
            args.experts = w["experts"]
            print(f"real experts: {args.checkpoint} layer {args.layer}: {w['experts']} experts, input scales "
                  f"FC1 {IN_UP:.6g} FC2 {IN_DOWN:.6g}, weight_scale_2 {w['up'][2].min().item():.3g}-{w['up'][2].max().item():.3g}")
        else:
            w = make_experts(activation, args.experts, device)
        prepared = {}
        if "planned" in paths:
            try:
                prepared["planned"] = planned_experts(w)
            except Exception as error:  # noqa: BLE001
                print(f"{activation} planned: unavailable ({type(error).__name__}: {str(error).splitlines()[0][:160]})")
        if paths & {"wrapper-trt", "forced"}:
            try:
                prepared["trt"], stats = trtllm_weights(w)
                print(f"{activation} wrapper-trt: FC1 block scales x weight_scale_2 re-cast to E4M3: "
                      f"{stats['zeroed'] * 100:.1f}% become 0, mean relative scale error {stats['scale_error'] * 100:.1f}%")
            except Exception as error:  # noqa: BLE001
                print(f"{activation} wrapper-trt: unavailable ({type(error).__name__}: {str(error).splitlines()[0][:160]})")
        if "wrapper-exact" in paths:
            try:
                prepared["exact"] = exact_weights(w)
            except Exception as error:  # noqa: BLE001
                print(f"{activation} wrapper-exact: unavailable ({type(error).__name__}: {str(error).splitlines()[0][:160]})")
        g = torch.Generator(device=device).manual_seed(11)
        for count in rows:
            x = (torch.randn((count, HIDDEN), device=device, generator=g) * .5).bfloat16()
            ids = torch.stack([torch.randperm(args.experts, device=device, generator=g)[:TOPK] for _ in range(count)]).int()
            weights = torch.rand((count, TOPK), device=device, generator=g) * .1 + .02
            expected = reference(w, x, ids.long(), weights)
            truth = reference(w, x, ids.long(), weights, quantize=False)
            oracles = {}
            runs = []
            if "planned" in prepared:
                name, moe, experts = prepared["planned"]
                runs.append(("planned", lambda: run_planned(moe, experts, x, ids, weights)))
                oracle = importlib.import_module(moe.__name__.replace("fused_moe", "_shared.kernels.reference"))
                impl = experts._impl
                for mode in ("micro", "dynamic_precise", "dynamic_fast"):
                    try:
                        oracles[mode] = oracle.moe_reference_nvfp4(x, impl.w1_fp4, impl.w1_blockscale, impl.w1_alphas,
                            impl.w2_fp4, impl.w2_blockscale, impl.w2_alphas, impl.a1_gscale, impl.a2_gscale, ids.long(),
                            weights, args.experts, HIDDEN, INTER, activation=activation, quant_scale_math=mode)
                    except Exception as error:  # noqa: BLE001
                        print(f"{activation} rows={count} b12x-oracle[{mode}]: unavailable ({type(error).__name__}: {error})")
                        continue
                    label = "b12x-oracle" if mode == "micro" else f"b12x-oracle[{mode}]"
                    runs.append((label, lambda r=oracles[mode], m=mode: (r, f"moe_reference_nvfp4({m})")))
            if "trt" in prepared and "wrapper-trt" in paths:
                runs.append(("wrapper-trt", lambda: run_wrapper(w, prepared["trt"], x, ids, weights)))
            if "exact" in prepared:
                runs.append(("wrapper-exact", lambda: run_wrapper(w, prepared["exact"], x, ids, weights)))
            if "trt" in prepared and "forced" in paths:
                for backend in ("micro", "static", "dynamic"):
                    if backend == "micro" and count > 8:
                        continue
                    runs.append((f"wrapper-trt[forced={backend}]",
                                 lambda b=backend: run_forced(w, prepared["trt"], x, ids, weights, b)))
                if "exact" in prepared:
                    for backend in ("micro", "static", "dynamic"):
                        if backend == "micro" and count > 8:
                            continue
                        runs.append((f"wrapper-exact[forced={backend}]",
                                     lambda b=backend: run_forced(w, prepared["exact"], x, ids, weights, b)))
            for label, call in runs:
                start = time.time()
                try:
                    out, backend = call()
                    error, worst = rel(out, expected)
                    # Extra columns: distance to the unquantized-activation truth (total error budget) and to
                    # b12x's oracle in each native scale-math mode (which evaluation order the kernel follows).
                    extra = f" truth={rel(out, truth)[0]:.4f}" + "".join(
                        f" or_{m}={rel(out, r)[0]:.4f}" for m, r in oracles.items())
                    print(f"{activation:5s} rows={count:3d} {label:30s} rel={error:.4f} worst_row={worst:.4f} "
                          f"backend={backend} ({time.time() - start:.1f}s){extra}", flush=True)
                except Exception as error:  # noqa: BLE001
                    message = str(error).splitlines()[0][:200] if str(error) else ""
                    print(f"{activation:5s} rows={count:3d} {label:30s} FAILED {type(error).__name__}: {message}", flush=True)
                    if args.verbose:
                        traceback.print_exc()
        del w, prepared
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

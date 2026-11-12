# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""F5: fused add + RMSNorm + static FP8 quantization for the input of a per-tensor FP8 Linear (Mamba in_proj).

Replaces TRT-LLM's two decode/prefill kernels on SM121
    flashinfer fused_add_rmsnorm (CuTe DSL, writes BF16)  ->  trtllm static_quantize_e4m3_per_tensor (scaleMatrixPerTensorVec)
with one kernel (megakernel/nhd_mk/fi_norm_exact.py, generated from FlashInfer 0.6.18 by bench/gen_norm_exact.py) that
keeps FlashInfer's reduction, rounds to BF16 like the unfused path, and scales with the same reciprocal, so the FP8
output and the updated residual are meant to be bit-identical (verified by bench/mk_f5.py; recip mode chosen there).
"""
import os

import torch

from .fi_norm_exact import fused_add_rmsnorm_quant_cute

RECIP_MODE = int(os.environ.get("NHD_F5_RECIP", "1"))  # 1: IEEE rcp.rn (matches TRT's 1.0f/s), 0: rcp.approx.ftz
PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"   # TRT passes get_env_enable_pdl() to flashinfer's norm


@torch.library.custom_op("nhd_mk::add_rmsnorm_fp8", mutates_args=("out", "residual"))
def add_rmsnorm_fp8(out: torch.Tensor, input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                    scale: torch.Tensor, eps: float, recip_mode: int) -> None:
    """residual += input; out = fp8(bf16(rmsnorm(residual) * weight) * (1/scale)), satfinite."""
    fused_add_rmsnorm_quant_cute(out, input, residual, weight, scale, eps, weight_bias=0.0, enable_pdl=PDL,
                                 recip_mode=recip_mode, round_input_dtype=True)


@add_rmsnorm_fp8.register_fake
def _(out, input, residual, weight, scale, eps, recip_mode):
    return None


def norm_fp8(norm, hidden, residual, scale, recip_mode=None):
    """TRT RMSNorm module `norm` + FP8 static `scale` -> (fp8 hidden, residual). residual is updated in place."""
    out = torch.empty(hidden.shape, dtype=torch.float8_e4m3fn, device=hidden.device)
    torch.ops.nhd_mk.add_rmsnorm_fp8(out, hidden, residual, norm.weight, scale.reshape(1).float(), float(norm.variance_epsilon),
                                     RECIP_MODE if recip_mode is None else recip_mode)
    return out, residual


def eligible(norm, linear, hidden, residual):
    """True when the fused path reproduces norm -> Linear's static FP8 input quantization for this call."""
    w = getattr(linear, "weight", None)
    s = getattr(linear, "input_scale", None)
    return (isinstance(residual, torch.Tensor) and w is not None and w.dtype == torch.float8_e4m3fn
            and isinstance(s, torch.Tensor) and s.numel() == 1
            and hidden.dtype == torch.bfloat16 and hidden.dim() == 2 and hidden.is_contiguous()
            and residual.is_contiguous() and not getattr(norm, "use_gemma", False)
            and getattr(linear, "lora", None) is None)

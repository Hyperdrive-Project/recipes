# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""F6 / F7 helpers (B300 kernel round, 2026-10-24). Kept apart from norm_quant.py so the as-run F5 files stay unchanged.

F6: the F5 fused norm with a second input, so the MoE layer's `shared_output + routed_output` (a separate elementwise
    kernel in rc1) is summed inside the next Mamba layer's add+RMSNorm+FP8 kernel. Bit-exact by construction: the sum is
    rounded to BF16 exactly like the torch add, then the F5 math runs unchanged (checked by patches/kernels/test_f6_f7.py).
F7: the unchanged F5 kernel applied to the attention qkv_proj / MLP up_proj inputs when those Linears are FP8
    per-tensor (eligible() decides per layer at runtime; NVFP4 or BF16 Linears keep the stock path).
"""
import torch

from .norm_quant import PDL, RECIP_MODE, eligible, norm_fp8  # noqa: F401  (norm_fp8 re-exported for F7)


@torch.library.custom_op("nhd_mk::add2_rmsnorm_fp8", mutates_args=("out", "residual"))
def add2_rmsnorm_fp8(out: torch.Tensor, input: torch.Tensor, input2: torch.Tensor, residual: torch.Tensor,
                     weight: torch.Tensor, scale: torch.Tensor, eps: float, recip_mode: int) -> None:
    """residual += bf16(input + input2); out = fp8(bf16(rmsnorm(residual) * weight) * (1/scale)), satfinite."""
    from .fi_norm_add2 import fused_add2_rmsnorm_quant_cute
    fused_add2_rmsnorm_quant_cute(out, input, input2, residual, weight, scale, eps, weight_bias=0.0, enable_pdl=PDL,
                                  recip_mode=recip_mode, round_input_dtype=True)


@add2_rmsnorm_fp8.register_fake
def _(out, input, input2, residual, weight, scale, eps, recip_mode):
    return None


def norm_fp8_add2(norm, hidden, hidden2, residual, scale, recip_mode=None):
    """F6: norm_fp8 with hidden := bf16(hidden + hidden2) folded into the kernel. residual is updated in place."""
    out = torch.empty(hidden.shape, dtype=torch.float8_e4m3fn, device=hidden.device)
    torch.ops.nhd_mk.add2_rmsnorm_fp8(out, hidden, hidden2, residual, norm.weight, scale.reshape(1).float(),
                                      float(norm.variance_epsilon), RECIP_MODE if recip_mode is None else recip_mode)
    return out, residual


def eligible2(norm, linear, hidden, hidden2, residual):
    """eligible() plus a second input of the same shape/dtype, contiguous, and no int32 offset overflow."""
    return (eligible(norm, linear, hidden, residual) and isinstance(hidden2, torch.Tensor)
            and hidden2.shape == hidden.shape and hidden2.dtype == hidden.dtype and hidden2.is_contiguous()
            and residual.shape == hidden.shape and hidden.numel() <= 2**31 - 1)

# Part of Hyperdrive recipes, https://github.com/Hyperdrive-Project/recipes
# Apache-2.0. If this helps you, a credit to Hyperdrive recipes is appreciated.
"""
Generated (generator script not included) from nhd_mk/fi_norm_exact.py -- do not edit by hand.

Nemotron Hyperdrive F6: the F5 kernel (FlashInfer fused add+RMSNorm with a bit-exact FP8 epilogue) with a second input
that is summed (rounded to BF16) before the residual add. Licensed under the Apache License, Version 2.0 (FlashInfer).
"""

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from .fi_norm_exact import (  # noqa: F401  (names used by the generated kernel body)
    COPY_BITS, FusedAddRMSNormKernel, RMSNormKernel, _torch_dtype_to_str, cvt_and_store_2xf32_to_fp8_hw,
    cvt_and_store_4xf32_to_fp8_hw, cvt_and_store_8xf32_to_fp8_hw, cvt_and_store_f32_to_fp8_hw,
    cvt_and_store_f32_to_fp8_sw, get_cutlass_dtype, get_fp8_max, get_ptr_as_int64, get_sm_version, has_hw_fp8_cvt,
    predicate_k, rcp_approx_ftz, rcp_rn, row_reduce_sum_multirow)


class FusedAdd2RMSNormQuantKernel:
    """
    Fused two-input Residual Add + RMSNorm + FP8 Quantization Kernel (Nemotron Hyperdrive F6).

    Computes:
    1. residual = bf16(input + input2) + residual (in-place update); the inner sum is rounded to the input dtype first,
       exactly like a separate torch add of two BF16 tensors
    2. output = residual / sqrt(mean(residual^2) + eps) * weight / scale,
       clamped to the finite range of the FP8 output dtype (E4M3 or E5M2)

    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
        use_hw_fp8: bool = True,
        sm_version: int | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.weight_bias = weight_bias
        self.use_hw_fp8 = use_hw_fp8
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        self.cluster_n = FusedAddRMSNormKernel._compute_cluster_n(
            H, dtype, self.sm_version
        )
        self.H_per_cta = H // self.cluster_n

        elem_bytes = dtype.width // 8
        max_vec_size = COPY_BITS // 8 // elem_bytes

        h_align = self.H_per_cta & (-self.H_per_cta)
        self.vec_size = min(h_align, max_vec_size)
        self.copy_bits = self.vec_size * dtype.width

        self.threads_per_row = RMSNormKernel._compute_threads_per_row(self.H_per_cta)
        self.num_threads = RMSNormKernel._compute_num_threads(self.H_per_cta)
        if self.H_per_cta > 8192 and self.num_threads < 256:
            self.num_threads = 256
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        if self.copy_bits >= 32:
            tile_bytes_2 = 2 * self.rows_per_block * self.cols_per_tile * elem_bytes
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.use_async_copy = (
                tile_bytes_2 <= props.shared_memory_per_block_optin // 2
            )
        else:
            self.use_async_copy = False

    def _smem_size_in_bytes(self) -> int:
        if self.use_async_copy:
            tile_bytes = (
                2 * self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            )
        else:
            tile_bytes = 0

        if self.cluster_n == 1:
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            reduction_bytes = (
                self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            )

        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes

    @cute.jit
    def __call__(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mX2: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int64,
        mS: cute.Tensor,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(mY, mX, mX2, mR, mW, M, mS, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mY: cute.Tensor,
        mX: cute.Tensor,
        mX2: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        M: Int64,
        mS: cute.Tensor,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        cols_per_tile = self.cols_per_tile
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        vec_size = self.vec_size
        num_vec_blocks = self.num_vec_blocks
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        if cutlass.const_expr(self.recip_mode == 1):
            inv_scale = rcp_rn(mS[0])
        else:
            inv_scale = rcp_approx_ftz(mS[0])

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gX2 = cute.local_tile(mX2, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        # Partition residual (same load tiled copy)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition second input (sync load, same partitioning as the weight copy)
        tX2gX2 = thr_copy_W.partition_S(gX2)
        tX2rX2 = cute.make_fragment_like(tX2gX2)
        tXrX2 = thr_copy_X.retile(tX2rX2)

        # Partition residual store destination (match non-quant kernel pattern)
        tRgO = thr_copy_O.partition_D(gR)
        tXgO_r = thr_copy_O.partition_D(gX)
        tRrO = cute.make_fragment_like(tXgO_r)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        tX2rX2.store(cute.zeros_like(tX2rX2, dtype=mX2.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_sync, tX2gX2, tX2rX2, pred=tWpW)

        # ===== Pass 1: Load input + residual, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = (tXrX.load().to(Float32) + tXrX2.load().to(Float32)).to(mX.element_type).to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        # Write h to residual (global)
        tRrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tRrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize, quantize, and store FP8 output =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))
        if cutlass.const_expr(self.round_input_dtype):
            y = y.to(mX.element_type).to(Float32)
        y = y * inv_scale

        tYrY_f32 = cute.make_rmem_tensor(tXrX.shape, Float32)
        tYrY_f32.store(y)

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row
        # Compute actual_row in int64 so that, with M now widened to Int64,
        # bidx * rows_per_block does not overflow int32 before being compared
        # against M or used in the address arithmetic below.
        actual_row = Int64(bidx) * rows_per_block + row_in_block
        col_offset = lane_in_row * vec_size
        fp8_max = get_fp8_max(mY.element_type)

        if cutlass.const_expr(self.use_hw_fp8 and vec_size == 8):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 8 <= H and actual_row < M:
                    base = v * 8
                    cvt_and_store_8xf32_to_fp8_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        tYrY_f32[base + 2],
                        tYrY_f32[base + 3],
                        tYrY_f32[base + 4],
                        tYrY_f32[base + 5],
                        tYrY_f32[base + 6],
                        tYrY_f32[base + 7],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                        mY.element_type,
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-fp8_max))
                            clamped = min(clamped, Float32(fp8_max))
                            cvt_and_store_f32_to_fp8_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                                mY.element_type,
                            )
        elif cutlass.const_expr(self.use_hw_fp8 and vec_size == 4):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 4 <= H and actual_row < M:
                    base = v * 4
                    cvt_and_store_4xf32_to_fp8_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        tYrY_f32[base + 2],
                        tYrY_f32[base + 3],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                        mY.element_type,
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-fp8_max))
                            clamped = min(clamped, Float32(fp8_max))
                            cvt_and_store_f32_to_fp8_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                                mY.element_type,
                            )
        elif cutlass.const_expr(self.use_hw_fp8 and vec_size == 2):
            for v in cutlass.range_constexpr(num_vec_blocks):
                local_col = col_offset + v * threads_per_row * vec_size
                abs_col = cluster_y * cols_per_tile + local_col
                if abs_col + 2 <= H and actual_row < M:
                    base = v * 2
                    cvt_and_store_2xf32_to_fp8_hw(
                        tYrY_f32[base],
                        tYrY_f32[base + 1],
                        get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        ),
                        mY.element_type,
                    )
                else:
                    for e in cutlass.range_constexpr(vec_size):
                        abs_col_e = cluster_y * cols_per_tile + local_col + e
                        if abs_col_e < H and actual_row < M:
                            flat_idx = v * vec_size + e
                            clamped = max(tYrY_f32[flat_idx], Float32(-fp8_max))
                            clamped = min(clamped, Float32(fp8_max))
                            cvt_and_store_f32_to_fp8_hw(
                                clamped,
                                get_ptr_as_int64(
                                    mY,
                                    cute.crd2idx(
                                        (Int64(actual_row), Int32(abs_col_e)),
                                        mY.layout,
                                    ),
                                ),
                                mY.element_type,
                            )
        else:
            for v in cutlass.range_constexpr(num_vec_blocks):
                for e in cutlass.range_constexpr(vec_size):
                    local_col = col_offset + v * threads_per_row * vec_size + e
                    abs_col = cluster_y * cols_per_tile + local_col
                    if abs_col < H and actual_row < M:
                        flat_idx = v * vec_size + e
                        clamped = max(tYrY_f32[flat_idx], Float32(-fp8_max))
                        clamped = min(clamped, Float32(fp8_max))
                        out_ptr = get_ptr_as_int64(
                            mY,
                            cute.crd2idx(
                                (Int64(actual_row), Int32(abs_col)), mY.layout
                            ),
                        )
                        if self.use_hw_fp8:
                            cvt_and_store_f32_to_fp8_hw(
                                clamped, out_ptr, mY.element_type
                            )
                        else:
                            cvt_and_store_f32_to_fp8_sw(
                                clamped, out_ptr, mY.element_type
                            )

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def _get_compiled_fused_add2_rmsnorm_quant_kernel(
    dtype_str: str,
    out_dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    use_hw_fp8: bool = True,
    sm_version: int = 80,
    recip_mode: int = 1,
    round_input_dtype: bool = True,
):
    """Compiled F6 kernel (contiguous tensors only; the caller falls back to the unfused path otherwise)."""
    dtype = get_cutlass_dtype(dtype_str)
    out_dtype = get_cutlass_dtype(out_dtype_str)
    kernel_obj = FusedAdd2RMSNormQuantKernel(dtype, H, weight_bias, use_hw_fp8=use_hw_fp8, sm_version=sm_version)
    kernel_obj.recip_mode = recip_mode
    kernel_obj.round_input_dtype = round_input_dtype
    sym_m = cute.sym_int(64)
    in_align = math.gcd(128, H * (dtype.width // 8))
    out_align = math.gcd(128, H * (out_dtype.width // 8))
    y_fake = cute.runtime.make_fake_compact_tensor(out_dtype, (sym_m, H), stride_order=(1, 0), assumed_align=out_align)
    x_fake = cute.runtime.make_fake_compact_tensor(dtype, (sym_m, H), stride_order=(1, 0), assumed_align=in_align)
    x2_fake = cute.runtime.make_fake_compact_tensor(dtype, (sym_m, H), stride_order=(1, 0), assumed_align=in_align)
    r_fake = cute.runtime.make_fake_compact_tensor(dtype, (sym_m, H), stride_order=(1, 0), assumed_align=in_align)
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    s_fake = cute.runtime.make_fake_compact_tensor(Float32, (1,), assumed_align=4)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel_obj, y_fake, x_fake, x2_fake, r_fake, w_fake, Int64(1), s_fake, Float32(1e-6), enable_pdl,
                        stream_fake, options="--enable-tvm-ffi")


def fused_add2_rmsnorm_quant_cute(
    out: torch.Tensor,
    input: torch.Tensor,
    input2: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    weight_bias: float = 0.0,
    enable_pdl: bool = False,
    recip_mode: int = 1,
    round_input_dtype: bool = True,
) -> None:
    """residual += bf16(input + input2); out = fp8 quant of the RMSNorm output (see FusedAdd2RMSNormQuantKernel)."""
    M, H = input.shape
    assert input2.shape == input.shape == residual.shape and out.shape == input.shape
    assert all(t.is_contiguous() for t in (out, input, input2, residual)) and M * H <= 2**31 - 1
    kernel = _get_compiled_fused_add2_rmsnorm_quant_kernel(
        _torch_dtype_to_str(input.dtype), _torch_dtype_to_str(out.dtype), H, weight_bias, enable_pdl,
        use_hw_fp8=has_hw_fp8_cvt(input.device), sm_version=get_sm_version(input.device),
        recip_mode=recip_mode, round_input_dtype=round_input_dtype)
    kernel(out, input, input2, residual, weight, M, scale, eps)

# ------------------------- Triton stuff -------------------------------

# type: ignore

import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel_forward(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, **meta):
    x = tl.zeros((100,), dtype=tl.float16)


def fused_forward(input, weight, target):
    out = torch.empty((), dtype=input.dtype, device=input.device)
    # Enqueue kernel.
    y = input
    x = input
    n_rows = 1024
    n_cols = 1024
    num_warps = 32
    BLOCK_SIZE = 32
    fused_kernel_forward[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_cols, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

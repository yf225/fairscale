# ------------------------- Triton stuff -------------------------------

# flake8: noqa
# type: ignore

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK_SIZE_DMODEL = 32


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_TOK": 16, "BLOCK_SIZE_VOC": 16, "BLOCK_SIZE_DMODEL": 32}, num_stages=2, num_warps=2),
    ],
    key=["tokens", "vocabs", "d_model"],
)
@triton.jit
def fused_kernel_forward(
    # fmt: off
    input, weight, output,
    tokens, vocabs, d_model,
    stride_tok, stride_d_tok,
    stride_voc, stride_d_voc,
    stride_out_tok, stride_out_voc,
    **meta
    # fmt: on
):
    """ fused kernel to compute the max. """

    # Extract config-parameters.
    block_tokens = meta["BLOCK_SIZE_TOK"]
    block_vocabs = meta["BLOCK_SIZE_VOC"]
    block_d_model = meta["BLOCK_SIZE_DMODEL"]

    # Get thread id.
    pid_tok = tl.program_id(axis=0)
    pid_voc = tl.program_id(axis=1)

    # Get offsets.
    off_tok = pid_tok * block_tokens + tl.arange(0, block_tokens)
    off_voc = pid_voc * block_vocabs + tl.arange(0, block_vocabs)
    off_d_model = tl.arange(0, block_d_model)

    # Get initial ptrs.
    ptr_tok = input + (off_tok[:, None] * stride_tok + off_d_model[None, :] * stride_d_tok)
    ptr_voc = weight + (off_d_model[:, None] * stride_d_voc + off_voc[None, :] * stride_voc)

    # Output accumulator (in fp32).
    acc = tl.zeros((block_tokens, block_vocabs), dtype=tl.float32)

    # Matmul loop.
    for k in range(0, d_model, block_d_model):
        # load. No mask and assume d_model is a multiply of block_d_model.
        t = tl.load(ptr_tok)
        v = tl.load(ptr_voc)
        # accumulate
        acc += tl.dot(t, v).to(tl.float32)
        # advance the block and offset for computing the masks.
        ptr_tok += block_d_model * stride_d_tok
        ptr_voc += block_d_model * stride_d_voc
        off_d_model += block_d_model

    # Get max for each token.

    # Write back (fp16).
    acc = acc.to(tl.float16)
    out_ptr = output + (stride_out_tok * off_tok[:, None] + stride_out_voc * off_voc[None, :])
    mask = (off_tok[:, None] < tokens) & (off_voc[None, :] < vocabs)
    tl.store(out_ptr, acc, mask=mask)


def fused_forward(input, weight, target):
    """ function to invoke fused kernels."""
    # check the args
    tokens, d1 = input.shape
    vocabs, d2 = weight.shape
    assert d1 == d2
    d_model = d1
    assert d_model % BLOCK_SIZE_DMODEL == 0, "d_model must be multiply of 32"
    assert input.is_contiguous()
    assert weight.is_contiguous()
    assert input.dtype == weight.dtype == torch.float16
    assert (tokens,) == target.shape
    assert target.dtype == torch.long
    assert input.stride(1) == weight.stride(1) == 1

    # Get the max output
    output = torch.empty((vocabs,), dtype=input.dtype, device=input.device)

    output = torch.ones((tokens, vocabs), dtype=input.dtype, device=input.device)

    # Enqueue kernel.
    grid = lambda META: (triton.cdiv(tokens, META["BLOCK_SIZE_TOK"]), triton.cdiv(vocabs, META["BLOCK_SIZE_VOC"]))
    fused_kernel_forward[grid](
        # fmt: off
        input, weight, output, # ptrs
        tokens, vocabs, d_model, # dims
        # strides
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        # fmt: on
    )
    return output

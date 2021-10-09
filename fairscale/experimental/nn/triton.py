# ------------------------- Triton stuff -------------------------------

# flake8: noqa
# type: ignore

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

MAX_BLOCK_SIZE_TOK = 128
MAX_BLOCK_SIZE_DMODEL = 32


@triton.autotune(
    configs=[
        # NOTE: MAX_BLOCK_SIZE_TOK due to padding below.
        triton.Config({"BLOCK_SIZE_TOK": 16, "BLOCK_SIZE_VOC": 16, "BLOCK_SIZE_DMODEL": 32}, num_stages=2, num_warps=2),
    ],
    key=["tokens", "vocabs", "d_model"],
)
@triton.jit
def fused_kernel_forward(
    input,
    weight,
    output,  # ptrs
    tokens,
    vocabs,
    d_model,  # dims
    stride_tok,
    stride_d_tok,
    stride_voc,
    stride_d_voc,
    stride_out_tok,
    stride_out_voc,
    **meta
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
    ptr_voc = weight + (off_voc[:, None] * stride_voc + off_d_model[None, :] * stride_d_voc)

    # Output accumulator (in fp32).
    acc = tl.zeros((block_tokens, block_vocabs), dtype=tl.float32)

    # Matmul loop.
    for k in range(0, d_model, block_d_model):
        # load
        t = tl.load(ptr_tok)  # no mask since we padded.
        mask = (off_voc[:, None] < vocabs) & (off_d_model[None, :] < d_model)
        v = tl.load(ptr_voc, mask=mask, other=0)
        v = tl.reshape(v, (v.shape[1], v.shape[0]))
        # accumulate
        acc += tl.dot(t, v).to(tl.float32)
        # advance the block and offset for computing the masks.
        ptr_tok += block_d_model * stride_d_tok
        ptr_voc += block_d_model * stride_d_voc
        off_d_model += block_d_model

    # Write back (fp16).
    acc = acc.to(tl.float16)
    out_ptr = output + (stride_out_tok * off_tok[:, None] + stride_out_voc * off_voc[None, :])
    # mask = (off_tok[:, None] < 3) & (off_voc[None, :] < 3)
    # tl.store(out_ptr, acc, mask=mask)
    tl.store(out_ptr, acc)


def fused_forward(input, weight, target):
    """ function to invoke fused kernels."""
    # check the args
    tokens, d1 = input.shape
    vocabs, d2 = weight.shape
    assert d1 == d2
    d_model = d1
    assert input.is_contiguous()
    assert weight.is_contiguous()
    assert input.dtype == weight.dtype == torch.float16
    assert (tokens,) == target.shape
    assert target.dtype == torch.long
    assert input.stride(1) == weight.stride(1)

    # pad the input since tl.load() mask above seems to be broken.
    # see: https://github.com/openai/triton/issues/330
    tokens, d1 = input.shape
    dim_0_pad = triton.cdiv(tokens, MAX_BLOCK_SIZE_TOK) * MAX_BLOCK_SIZE_TOK - tokens
    dim_1_pad = triton.cdiv(d_model, MAX_BLOCK_SIZE_DMODEL) * MAX_BLOCK_SIZE_DMODEL - d_model
    print("pad", tokens, d1, "to", tokens + dim_0_pad, d1 + dim_1_pad)
    input = F.pad(input, (0, dim_0_pad, 0, dim_1_pad), "constant", 0)

    # Get the max output
    output = torch.empty((vocabs,), dtype=input.dtype, device=input.device)

    output = torch.ones((8, 16), dtype=input.dtype, device=input.device)

    # Enqueue kernel.
    grid = lambda META: (triton.cdiv(tokens, META["BLOCK_SIZE_TOK"]), triton.cdiv(vocabs, META["BLOCK_SIZE_VOC"]))
    fused_kernel_forward[grid](
        input,
        weight,
        output,  # ptrs
        tokens,
        vocabs,
        d_model,  # dims
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),  # strides
        output.stride(0),
        output.stride(1),  # tmp
    )
    return output

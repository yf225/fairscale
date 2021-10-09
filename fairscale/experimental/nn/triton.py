# ------------------------- Triton stuff -------------------------------

# flake8: noqa
# type: ignore

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Enable to do autotune
_searching = False

# d_model dim must be a multiply of this.
BLOCK_SIZE_DMODEL = 32


def get_configs():
    if _searching:
        block_sizes_tok = [32, 64, 128, 256]
        block_sizes_voc = [32, 64, 128, 256]
        stages = [3, 4, 5]
        warps = [2, 4, 8]
    else:
        block_sizes_voc = [32, 64, 128, 256]
        block_sizes_voc = [32, 64, 128, 256]
        stages = [3, 4, 5]
        warps = [2, 4, 8]
    cfgs = []
    for block_size_tok in [32, 64, 128, 256]:
        for block_size_voc in [32, 64, 128, 256]:
            for num_stages in [3, 4, 5]:
                for num_warps in [2, 4, 8]:
                    c = triton.Config(
                        {
                            "BLOCK_SIZE_TOK": block_size_tok,
                            "BLOCK_SIZE_VOC": block_size_voc,
                            "BLOCK_SIZE_DMODEL": 32,
                            "GROUP_SIZE_TOK": 8,
                        },
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                    cfgs.append(c)
    return cfgs


@triton.autotune(
    configs=get_configs(), key=["tokens", "vocabs", "d_model"],
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
    group_size_tok = meta["GROUP_SIZE_TOK"]

    # Compute the group of blocks this thread is in.
    pid = tl.program_id(axis=0)
    grid_tokens = tl.cdiv(tokens, block_tokens)
    grid_vocabs = tl.cdiv(vocabs, block_vocabs)
    group_id = pid // grid_vocabs // group_size_tok
    group_first_pid_tok = group_id * group_size_tok
    group_size_tok = min(grid_tokens - (group_id * group_size_tok), group_size_tok)  # in case this is the last group

    # Get my pid along tok and voc dimensions.
    pid_tok = group_first_pid_tok + pid % group_size_tok  # go down columns for better perf
    pid_voc = (pid - group_first_pid_tok * grid_vocabs) // group_size_tok

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

    # Kernel grid
    def grid(meta):
        # Use 1D grid so that we can control the groups for L2 cache access pattern.
        if _searching:
            print(meta)
        return (triton.cdiv(tokens, meta["BLOCK_SIZE_TOK"]) * triton.cdiv(vocabs, meta["BLOCK_SIZE_VOC"]),)

    # Kernel launch
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

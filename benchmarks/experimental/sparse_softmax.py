# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
from pprint import pprint
import time

import torch
from torch.cuda import Event

from fairscale.experimental.nn import (  # noqa: F401
    BaselineSoftmax,
    BaselineSoftmaxNllLoss,
    InplaceSoftmax,
    TiledSoftmax,
    TopKFaissSoftmax,
    TopKSoftmax,
    TopKTiledSoftmax,
    TorchFuseAllTiled,
    TritonFuseAll,
    TritonSoftmax,
)
from fairscale.experimental.nn.sparse_softmax import get_data
from fairscale.utils.testing import get_smi_memory

""" Benchmarking various softmax kernels. Some are dense and some are with label and top-K sparsity. """

# TODO:
#   From Naman: d_model varies between [2k, 12k]
#               input: 8 * 2K = 16K -- 2K is seq len, 8 is batch size
#               vocab: 256K
SHAPES = [
    # name, activation, FC weights
    # ("1k_128h_256k", (1024, 128), (128, 256 * 1024)),
    # ("4k_128h_256k", (4096, 128), (128, 256 * 1024)),
    # ("8k_4k_32k", (4 * 2048, 4 * 1024), (4 * 1024, 32 * 1024)),
    # ("24k_4k_50k", (12 * 2048, 4 * 1024), (4 * 1024, 50 * 1024)),
    # ("8k_4k_256k", (4 * 2048, 4 * 1024), (4 * 1024, 256 * 1024)),
    # ("8k_4k_256008", (4 * 2048, 4 * 1024), (4 * 1024, 256008)),  # max seq len for base is 2100, 2300 for top-k
    ("xk_4k_256008", (8 * 2048, 4 * 1024), (4 * 1024, 256008)),  # max seq len for base is 2100, 2300 for top-k
]
KERNELS = [
    # BaselineSoftmax,
    # BaselineSoftmaxNllLoss,  # bs2=16G, bs4=28G, bs8=illegal mem
    # TritonFuseAll,
    TorchFuseAllTiled,  # bs2=12G, bs4=16G, bs8=23.4G
    #    TritonSoftmax,
    #    InplaceSoftmax,
    # TiledSoftmax,
    #    TopKSoftmax,
    # TopKTiledSoftmax,
    #    TopKFaissSoftmax,
]


def my_nll_loss(lprobs, target):
    """ Like that in fairseq, when lprobs.numel > 2r9, it uses a loss like this. """
    target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    return nll_loss.mean()


def run_on_gpu(kernel, data, repeats, no_grad, fwd_bwd):
    input, weight, target = data

    # Ensure GPU memory is minimal and get_smi_memory is good
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cur_mem_before = round(torch.cuda.memory_allocated() / 1024 / 1024)
    smi_mem_before = get_smi_memory()
    assert cur_mem_before == 0, cur_mem_before

    # Move tensors to GPU.
    input, weight.data, target = input.cuda(), weight.data.cuda(), target.cuda()

    # Create the kernel
    k = kernel(
        weight, k=200, tile_factor=16
    )  # 16 is good for TorchFuseAllTiled, 8 and 32 are both slower. Power of 2 is much better. memory wise, 16 is good enough and their seems to be a floor of 2.xGB no matter what with no_grad

    # Get the events
    events = [Event(enable_timing=True) for _ in range(repeats)]

    # Queue the ops to GPU
    cpu_start_time = time.time()
    for i in range(repeats):
        context = contextlib.suppress()
        if no_grad:
            context = torch.no_grad()
        with context:
            events[i].record()
            out = k(input, target)
            if fwd_bwd:
                if kernel not in [BaselineSoftmaxNllLoss, TorchFuseAllTiled]:
                    my_nll_loss(out, target).backward()
                else:
                    out.backward()
            del out
    # Cpu is done
    cpu_time = time.time() - cpu_start_time
    # Might wait for gpu here
    torch.cuda.synchronize()

    # Get the durations
    durations = [cpu_time * 1000]  # convert seconds to ms.
    for x, y in zip(events, events[1:]):
        durations.append(x.elapsed_time(y))

    # Free memory
    del k
    input, weight.data, target = input.cpu(), weight.data.cpu(), target.cpu()
    weight.grad = None
    cur_mem_after = round(torch.cuda.memory_allocated() / 1024 / 1024)
    assert cur_mem_after == 0, cur_mem_after

    # Get peak mem
    peak_mem_after = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
    smi_mem_after = get_smi_memory()
    peak_mem = peak_mem_after - cur_mem_before
    smi_peak_mem = smi_mem_after - smi_mem_before

    return peak_mem, smi_peak_mem, durations


def main():
    parser = argparse.ArgumentParser("Benchmarking softmax kernels")

    parser.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--grad", type=str, choices=["grad", "no_grad"], default="grad")
    parser.add_argument("--fwd_bwd", action="store_true", default=False)
    args = parser.parse_args()

    repeats = 9
    results = {}
    results["peak cached"] = {}
    results["peak smi"] = {}
    results["durations"] = {}
    for shape in SHAPES:
        name = shape[0]
        results["peak cached"][name] = {}
        results["peak smi"][name] = {}
        results["durations"][name] = {}
        dtype = torch.float32 if args.dtype == "fp32" else torch.float16
        data = get_data(shape[1:], dtype, "cpu")  # Use cpu memory to ensure we always start with empty GPU
        for kernel in KERNELS:
            k_name = kernel.__name__
            no_grad = args.grad
            print(f"Running {k_name} with {name} {dtype} {no_grad} data")
            peak_mem, smi_peak_mem, durations = run_on_gpu(kernel, data, repeats, no_grad == "no_grad", args.fwd_bwd)
            results["peak cached"][name][k_name] = peak_mem
            results["peak smi"][name][k_name] = smi_peak_mem
            results["durations"][name][k_name] = durations
    pprint(results)


if __name__ == "__main__":
    main()

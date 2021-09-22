# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pprint
import time

import torch
from torch.cuda import Event

from fairscale.experimental.nn import BaselineSoftmax, TopKSoftmax, TopKSoftmaxFaiss
from fairscale.experimental.nn.sparse_softmax import get_data

""" Benchmarking various softmax kernels with label and top-K sparsity. """

# TODO: measure mixed precisions.

SHAPES = [
    # name, activation, FC weights
    ("4k_128h_256k", (4096, 128), (128, 256 * 1024)),
]
KERNELS = [BaselineSoftmax, TopKSoftmax, TopKSoftmaxFaiss]


def run_on_gpu(kernel, data, repeats):
    input, weight, target = data

    # Ensure GPU memory is 0

    # Create the kernel
    k = kernel(weight, 200)

    # Get the events
    events = [Event(enable_timing=True) for _ in range(repeats)]

    # Queue the ops to GPU
    cpu_start_time = time.time()
    for i in range(repeats):
        events[i].record()
        k(input, target)
    cpu_time = time.time() - cpu_start_time
    torch.cuda.synchronize()

    # Get the durations
    durations = [cpu_time * 1000]  # convert seconds to ms.
    for x, y in zip(events, events[1:]):
        durations.append(x.elapsed_time(y))

    # Get peak mem
    peak_mem = 0

    return peak_mem, durations


def main():
    repeats = 9
    results = {}
    results["peak_mem"] = {}
    results["durations"] = {}
    for shape in SHAPES:
        name = shape[0]
        results["peak_mem"][name] = {}
        results["durations"][name] = {}
        data = get_data(shape[1:])
        for kernel in KERNELS:
            k_name = kernel.__name__
            peak_mem, durations = run_on_gpu(kernel, data, repeats)
            results["peak_mem"][name][k_name] = peak_mem
            results["durations"][name][k_name] = durations
    pprint(results)


if __name__ == "__main__":
    main()

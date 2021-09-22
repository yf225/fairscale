# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from pprint import pprint

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
    k = kernel(weight)

    # Queue the ops to GPU
    for _ in range(repeats):
        k(input, target)

    # Get the durations

    # Get peak mem
    return peak_mem, durations


def main():
    repeats = 10
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

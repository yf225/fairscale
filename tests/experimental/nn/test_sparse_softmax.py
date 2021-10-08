# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import pytest
import torch
from torch import nn

from fairscale.experimental.nn import (
    BaselineSoftmax,
    InplaceSoftmax,
    TiledSoftmax,
    TopKFaissSoftmax,
    TopKSoftmax,
    TopKTiledSoftmax,
    TritonSoftmax,
)
from fairscale.experimental.nn.sparse_softmax import get_data
from fairscale.utils.testing import skip_if_no_cuda


@pytest.fixture(scope="session", params=[torch.float16, torch.float32])
def input_data(request):
    shape = ((2, 3), (3, 4))
    return get_data(shape, dtype=request.param)


_dense_out = {}
_dense_grad = {}


@skip_if_no_cuda
@pytest.mark.parametrize("kernel", [BaselineSoftmax, TritonSoftmax, InplaceSoftmax, TiledSoftmax])
def test_dense(input_data, kernel):
    # Prepare
    input, weight, target = input_data
    weight.grad = None

    if kernel is TritonSoftmax:
        pytest.skip("skip TritonSoftmax since it takes too long")

    if kernel is InplaceSoftmax and input.dtype == torch.float16:
        pytest.skip("skip InplaceSoftmax and fp16 until it is implemented")

    if kernel is TiledSoftmax:
        tile_factor = 2
        sm = kernel(weight, tile_factor=tile_factor)
    else:
        sm = kernel(weight)

    # Forward
    out = sm(input, target)
    if kernel is TiledSoftmax:
        orig_out = out
        out = torch.cat(out, dim=0)

    # Check
    assert out.shape == (2, 4)
    global _dense_out
    if out.dtype not in _dense_out:
        _dense_out[out.dtype] = out
        print(out)
    else:
        torch.allclose(_dense_out[out.dtype], out)

    # Backward
    if kernel is InplaceSoftmax:
        # Inplace can't do autograd
        return
    loss = nn.CrossEntropyLoss()
    if kernel is TiledSoftmax:
        out = orig_out
        out = [loss(o, t).unsqueeze(dim=0) for o, t in zip(out, torch.split(target, target.shape[0] // tile_factor, 0))]
        out = torch.cat(out, dim=0)
        out.mean().backward()
    else:
        loss(out, target).backward()

    # Check
    global _dense_grad
    if weight.dtype not in _dense_grad:
        _dense_grad[weight.dtype] = weight.grad
        print(weight.grad)
    else:
        torch.allclose(_dense_grad[weight.dtype], weight.grad)


@skip_if_no_cuda
def test_topk(input_data):
    input, weight, target = input_data

    if input.dtype == torch.float16:
        pytest.skip("TopKSoftmax and fp16 not yet implemented")

    sm = TopKSoftmax(weight, k=2)
    out = sm(input, target)
    assert out.shape == (2, 4)
    print(out)
    loss = nn.CrossEntropyLoss()
    loss(out, target).backward()
    print(weight.grad)


@skip_if_no_cuda
def test_topk_tiled(input_data):
    input, weight, target = input_data
    weight.grad = None
    sm = TopKTiledSoftmax(weight, k=2, tile_factor=2)
    out = sm(input, target)
    print(out)
    assert out.shape == (2, 4)
    loss = nn.CrossEntropyLoss()
    loss(out, target).backward()
    print(weight.grad)


@skip_if_no_cuda
def test_topk_tiled_uneven():
    print()
    shape = ((4, 3), (3, 5))
    input, weight, target = get_data(shape, dtype=torch.float16)
    weight.grad = None
    sm = TopKTiledSoftmax(weight, k=2, tile_factor=2)
    out = sm(input, target)
    print("kernel output", out)
    assert out.shape == (4, 5)
    loss = nn.CrossEntropyLoss()
    loss(out, target).backward()
    assert weight.grad.shape == weight.data.shape
    print("grad", weight.grad)


@skip_if_no_cuda
def test_topk_faiss(input_data):
    pytest.skip("skip TopKFaissSoftmax until it is actually used")
    input, weight, target = input_data
    sm = TopKFaissSoftmax(weight, k=2)
    out = sm(input, target)
    assert out.shape == (2, 2)

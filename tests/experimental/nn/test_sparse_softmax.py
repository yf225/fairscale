# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from fairscale.experimental.nn import BaselineSoftmax, TopKSoftmax, TopKSoftmaxFaiss
from fairscale.experimental.nn.sparse_softmax import get_data
from fairscale.utils.testing import skip_if_no_cuda


@skip_if_no_cuda
def test_baseline():
    shape = ((2, 3), (3, 4))
    input, weight, target = get_data(shape)
    sm = BaselineSoftmax(weight)
    out = sm(input, target)
    assert out.shape == (2, 4)


@skip_if_no_cuda
def test_topk():
    shape = ((2, 3), (3, 4))
    input, weight, target = get_data(shape)
    sm = TopKSoftmax(weight, k=2)
    out = sm(input, target)
    assert out.shape == (2, 4)


@skip_if_no_cuda
def test_topk_faiss():
    shape = ((2, 3), (3, 4))
    input, weight, target = get_data(shape)
    sm = TopKSoftmaxFaiss(weight, k=2)
    out = sm(input, target)
    assert out.shape == (2, 2)

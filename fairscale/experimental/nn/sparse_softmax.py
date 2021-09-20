# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

import faiss
import faiss.contrib.torch_utils  # noqa: F401, actually used just by importing
import torch
from torch import nn

#  res = faiss.StandardGpuResources()
#  res.setTempMemory(1024*1024*100)
#  D, I = faiss.knn_gpu(res, q, b.T, 2)


class BaselineSoftmax(nn.Module):
    """ Baseline softmax that does an output projection and a softmax. """

    def __init__(self, proj_weight: torch.Tensor, log_prob: bool):
        super().__init__()
        # todo

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        assert kwargs == {}
        input, target = input
        return input, target


class TopKSoftmax(nn.Module):
    """ TopK softmax that does an project and take the top-K and then softmax.

        Peak GPU memory is not reduced since entire output is generated and then
        top-K is applied.

        TODO: can also implement a variant that does reduce GPU memory by tiled
              views of the output.
    """

    pass


class TopKSoftmaxFaiss(nn.Module):
    """ TopK softmax that uses FAISS's fused project & top-K to reduce GPU memory. """

    pass

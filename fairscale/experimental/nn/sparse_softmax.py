# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Tuple

import faiss
import faiss.contrib.torch_utils  # noqa: F401, actually used just by importing
import torch
from torch import nn
import torch.nn.functional as F


def get_data(shape: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[torch.Tensor, torch.nn.Parameter, torch.Tensor]:
    """ Utility function for getting some tensors for testing and benchmarking."""
    (tokens, d1), (d2, vocabs) = shape
    assert d1 == d2
    input = torch.rand(tokens, d1, device="cuda")
    weight = nn.Linear(d2, vocabs, bias=False, device="cuda").weight
    target = (torch.rand(tokens, device="cuda") * vocabs).long()
    return input, weight, target


class BaselineSoftmax(nn.Module):
    """ Baseline softmax that does an output projection and a softmax. """

    def __init__(self, proj_weight: torch.nn.Parameter, k: int = 0):  # k is ignored.
        super().__init__()
        in_dim, out_dim = proj_weight.shape
        self.fc = nn.Linear(in_dim, out_dim, bias=False, device="cuda")
        self.fc.weight = proj_weight

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        assert kwargs == {}
        input, target = input
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        x = self.fc(input)
        x = F.softmax(x, dim=-1)
        return x


class TopKSoftmax(nn.Module):
    """ TopK softmax that does an project and take the top-K and then softmax.

        Peak GPU memory is not reduced since entire output is generated and then
        top-K is applied.

        TODO: can also implement a variant that does reduce GPU memory by tiled
              views of the FC output.
    """

    def __init__(self, proj_weight: torch.nn.Parameter, k: int):
        super().__init__()
        in_dim, out_dim = proj_weight.shape
        self.fc = nn.Linear(in_dim, out_dim, bias=False, device="cuda")
        self.fc.weight = proj_weight
        self.k = k

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        assert kwargs == {}
        input, target = input
        assert isinstance(input, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        # Get matmul output.
        x = self.fc(input)
        # Get the top-K index.
        D, I = torch.topk(x, self.k)
        # Add in the targets.
        I = torch.cat([I, target.reshape(-1, 1)], dim=1)
        # Generate a mask.
        mask = torch.ones_like(x) * float("-inf")
        mask.scatter_(1, I, 0.0)
        # Mask x and softmax it.
        x = x + mask
        x = F.softmax(x, dim=-1)
        return x


class TopKSoftmaxFaiss(nn.Module):
    """ TopK softmax that uses FAISS's fused project & top-K to reduce GPU memory.

        Note, the output of this kernel is reduced in size. It is no longer the
        [tokens, vocabs] shape. Instead, it is in the shape of [k+1, vocabs].
    """

    def __init__(self, proj_weight: torch.nn.Parameter, k: int):
        super().__init__()
        self.k = k
        self.res = faiss.StandardGpuResources()
        self.res.setTempMemory(1024 * 1024 * 100)
        self.b = proj_weight.data

    def forward(self, *input: Any, **kwargs: Any) -> Any:
        assert kwargs == {}
        input, target = input
        D, I = faiss.knn_gpu(self.res, input, self.b, self.k)
        # TODO: add in target
        return D

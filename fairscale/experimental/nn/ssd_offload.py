from __future__ import annotations

from functools import reduce
import io
import os
import pickle
from typing import IO, Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import _functional as F
from torch.serialization import DEFAULT_PROTOCOL as DEFAULT_PROTOCOL
from torch.utils._pytree import tree_map

DEFAULT_CHUNK_SIZE = 1024 * 1024


def _get_num_chunks(t: torch.Tensor, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> int:
    size_in_bytes = t.nelement() * t.element_size()
    num_chunks = (size_in_bytes + (chunk_size_bytes - 1)) // chunk_size_bytes
    return num_chunks


def _tensor_to_bytes_chunks(t: torch.Tensor, chunk_idx: int, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> bytes:
    size_in_bytes = t.nelement() * t.element_size()
    assert chunk_idx < _get_num_chunks(t, chunk_size_bytes)
    t_np = t.detach().numpy().view(np.uint8).reshape(-1)
    chunk_start = chunk_idx * chunk_size_bytes
    chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
    return t_np[chunk_start:chunk_end].tobytes()


def write(t: torch.Tensor, filename: str, file_offset_bytes: int = 0) -> None:
    num_chunks = _get_num_chunks(t)
    with open(filename, "wb") as f:
        f.seek(file_offset_bytes)
        for i in range(num_chunks):
            f.write(_tensor_to_bytes_chunks(t, i))


def read(t: torch.Tensor, filename: str, file_offset_bytes: int = 0) -> None:
    size_in_bytes = t.nelement() * t.element_size()
    chunk_size_bytes = DEFAULT_CHUNK_SIZE
    num_chunks = _get_num_chunks(t)
    t_np = t.detach().numpy()
    t_mv = memoryview(t_np.view(dtype=np.uint8).reshape(-1))
    fixed_mv = t_mv[0:chunk_size_bytes]
    with io.open(filename, "rb") as f:
        f.seek(file_offset_bytes)
        for i in range(num_chunks):
            chunk_start = i * chunk_size_bytes
            chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
            data_read = f.readinto(t_mv[chunk_start:chunk_end])
            assert data_read == chunk_end - chunk_start


# PJ: WIP
"""
class SsdAdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False) -> None:
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(SsdAdamOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state) -> None:
        super(SsdAdamOptimizer, self).__setstate_(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[int]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            F.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )
        return loss
"""


class SsdTensorHandle(torch.Tensor):
    @staticmethod
    def __new__(
        cls: SsdTensorHandle, shape: Tuple[int, ...], dtype: torch.dtype, requires_grad: bool = False
    ) -> SsdTensorHandle:
        r = torch.Tensor._make_subclass(cls, torch.empty(shape, dtype=dtype).to("meta"), requires_grad)
        return r

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, requires_grad: bool) -> None:
        self._shape = shape
        if len(shape) == 0:
            self._numel = 0
        else:
            self._numel = reduce((lambda x, y: x * y), shape)
        self._dtype = dtype
        # valid if offloaded to file
        self.filename = ""
        self.offset = -1
        # valid if loaded to memory
        self.tensor: Optional[torch.Tensor] = None
        self.requires_grad = requires_grad

    @classmethod
    def from_file(
        cls, shape: Tuple[int, ...], dtype: torch.dtype, filename: str, requires_grad: bool = False
    ) -> SsdTensorHandle:
        handle = cls(shape=shape, dtype=dtype, requires_grad=requires_grad)
        handle.filename = filename
        return handle

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> SsdTensorHandle:
        # TODO(anj): figure out why requires_grad property does not flow.
        handle = cls(shape=tensor.shape, dtype=tensor.dtype, requires_grad=tensor.requires_grad)
        handle.tensor = tensor
        return handle

    def is_available(self) -> bool:
        return self.tensor is not None

    def get_tensor(self) -> torch.Tensor:
        assert self.tensor is not None
        return self.tensor

    def set_file_params(self, filename: str, offset: int) -> None:
        self.filename = filename
        self.offset = offset

    def point_to_file(self, filename: str, offset: int) -> None:
        self.set_file_params(filename, offset)
        self.tensor = None

    def point_to_tensor(self, tensor: torch.Tensor) -> None:
        assert self.tensor is None
        assert self._shape == tensor.shape
        assert self._dtype == tensor.dtype
        self.tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        if self.tensor is not None:
            return self.tensor
        else:
            result_tensor = torch.empty(size=self._shape, dtype=self._dtype, requires_grad=self.requires_grad)
            self.copy_into_tensor(result_tensor)
            self.tensor = result_tensor
            return self.tensor

    def to_file(self, release_tensor_after_write: bool = True) -> None:
        assert self.tensor is not None
        write(self.tensor, self.filename, self.offset * self.tensor.element_size())
        if release_tensor_after_write:
            self.tensor = None

    def copy_into_tensor(self, tensor: torch.Tensor) -> None:
        """
        if self.is_available(), this copies the Handle's tensor
        into the passed in tensor. Otherwise, if !is_available(),
        this reads from file into tensor, using the read() function.
        Does not modify modify self.tensor unlike to_tensor() function.
        This can be useful for calls like named_parameters() when
        the tensor is already offloaded to disk.
        """
        assert self._shape == tensor.shape
        assert self._dtype == tensor.dtype
        if self.tensor is not None:
            tensor.copy_(self.tensor)
        else:
            read(tensor, self.filename, self.offset * tensor.element_size())

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        ssd_tensor_handles = []

        def unwrap(e):
            if isinstance(e, SsdTensorHandle):
                t = e.to_tensor()
                ssd_tensor_handles.append((e, t._version))
                return t
            else:
                return e

        # need to test if tensor is modified
        r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        for e, saved_version in ssd_tensor_handles:
            # TODO: version counter trick doesn't work
            """
            if saved_version != e.tensor._version:
                r.to_file()
            """
            e.to_file()
        return r


# Class supporting a single SSD file backing one or more tensors
class SsdBuffer:
    def __init__(self, num_elems: int, filename: str) -> None:
        # TODO(anj): add an option of passing the dtype of the buffer
        # we want to track.
        self.buffer: Optional[torch.Tensor] = torch.empty((num_elems,))
        self.filename = filename
        self.offset = 0
        self.tensors: Dict[int, SsdTensorHandle] = {}

    def allocate(self, n: int) -> SsdTensorHandle:
        assert n > 0
        assert list(self.buffer.size()) != [1]
        assert self.can_alloc(n)

        tensor = self.buffer.narrow(0, self.offset, n)

        tensor_offset = self.offset
        handle = SsdTensorHandle.from_tensor(tensor)
        self.tensors[tensor_offset] = handle
        handle.set_file_params(self.filename, tensor_offset)
        self.offset += n

        return handle

    def insert(self, tensor: torch.Tensor) -> SsdTensorHandle:
        assert list(self.buffer.size()) != [1]
        # For the non sharded case, the tensor will not be flattened
        tensor = tensor.reshape(-1)
        assert self.buffer.dtype == tensor.dtype
        handle = self.allocate(tensor.numel())
        handle.get_tensor().copy_(tensor)
        return handle

    def can_alloc(self, n: int) -> bool:
        assert list(self.buffer.size()) != [1]
        return (self.offset + n) <= self.buffer.numel()

    def get_tensors(self) -> List[SsdTensorHandle]:
        return [t for t in self.tensors.values()]

    def to_disk(self) -> None:
        assert list(self.buffer.size()) != [1]
        # TODO(anj): Add comment about why we use `narrow`.
        valid_data = self.buffer.narrow(0, 0, self.offset)
        write(valid_data, self.filename)

        # Remove all Tensor references
        for offset, t in self.tensors.items():
            t.point_to_file(self.filename, offset)

        # TODO(anj-s): Setting this to None does not result in GC picking
        # this reference up.
        self.buffer = torch.empty((1))

    def from_disk(self, num_elems: int) -> None:
        if num_elems < self.offset:
            raise RuntimeError(
                f"Attempted to load from file ssdbuffer of size: {self.offset} into a buffer that is of size: {num_elems}"
            )
        self.buffer = torch.empty((num_elems,))
        valid_data = self.buffer.narrow(0, 0, self.offset)
        read(valid_data, self.filename)

        # Restore Tensor References
        # We are book-keeping the offset in two places -
        # both in the SSDTensorHandle and the SSD buffer.
        # TODO(anj): Do we need to do this twice?
        for offset, t in self.tensors.items():
            t.point_to_tensor(self.buffer.narrow(0, t.offset, t._numel))


# Classes supporting torch.save/load
class TorchSaver:
    def __init__(self) -> None:
        self.pickle_module = DisableMemoizationPicklerModule

    def save(
        self, obj: Any, f: Union[str, os.PathLike, BinaryIO, IO[bytes]], pickle_protocol: int = DEFAULT_PROTOCOL
    ) -> None:
        torch.serialization.save(
            obj, f, self.pickle_module, pickle_protocol=pickle_protocol, _use_new_zipfile_serialization=False
        )


class DisableMemoizationPicklerModule:
    @classmethod
    def Pickler(cls, data_buf: io.BytesIO, protocol: int) -> pickle.Pickler:
        p = pickle.Pickler(data_buf, protocol)
        p.fast = True
        return p

    @classmethod
    def dump(cls, obj: Any, f: io.BytesIO, protocol: int) -> None:
        pickle.dump(obj, f, protocol)


class FileChunkingIterator:
    """
    chunk_size_bytes determines how large each chunk that we break the file
    into. It is important to consider limiting the size because by when
    python unpickles an object, by default it will read up to 1000 list
    elements at a time. So memory usage while unpickling will be on the
    order of O(min(file_size, 1000 * chunk_size_bytes)).
    """

    def __init__(self, filename: str, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> None:
        self.filename = filename
        self.file: Optional[Union[BinaryIO, IO[bytes]]] = None
        self.chunk_size_bytes = chunk_size_bytes
        self.num_chunks_read = 0

    def __iter__(self) -> Iterator[bytes]:
        self.file = io.open(self.filename, "rb", buffering=0)
        self.num_chunks_read = 0
        return self

    def __next__(self) -> bytes:
        assert self.file
        next_chunk = self.file.read(self.chunk_size_bytes)

        if len(next_chunk) == 0:
            raise StopIteration
        self.num_chunks_read += 1

        return next_chunk


class SsdTensor:
    def __init__(self, shape: Tuple[int, ...], filename: str, dtype: torch.dtype = torch.float) -> None:
        self.filename = filename
        self.f: Optional[Union[BinaryIO, IO[bytes]]] = None
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def __unpickle__(cls, shape: Tuple[int, ...], filename: str, dtype: torch.dtype) -> SsdTensor:
        result = cls(shape, filename, dtype)
        result.f = io.open(result.filename, "wb")
        return result

    @classmethod
    def fromtensor(cls, tensor: torch.Tensor, filename: str) -> SsdTensor:
        result = cls(tensor.shape, filename, tensor.dtype)
        write(tensor, result.filename)
        return result

    def __reduce_ex__(self, protocol: int) -> Tuple[Callable, Any, Any, Any]:
        # adding _2 to the filename is just a hack to prevent overwriting the original SsdTensor data
        return (
            type(self).__unpickle__,
            (self.shape, self.filename + "_2", self.dtype,),
            None,
            iter(FileChunkingIterator(self.filename)),
        )

    def append(self, item: bytes) -> None:
        assert self.f
        self.f.write(item)

    def extend(self, items: List[bytes]) -> None:
        for i in items:
            self.append(i)


torch_saver = TorchSaver()

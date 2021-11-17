from __future__ import annotations

from functools import reduce
import io
import os
import pickle
from typing import IO, Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL as DEFAULT_PROTOCOL

try:
    from torch.utils._pytree import tree_map
except ImportError:
    # The PyTorch version(<1.9) we test with may not support the tree_map API.
    pass
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
    file_flags = "r+b" if os.path.exists(filename) else "wb"
    with open(filename, file_flags) as f:
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

class SsdTensorHandle(torch.Tensor):
    @staticmethod
    def __new__(
        cls: SsdTensorHandle, shape: Tuple[int, ...], dtype: torch.dtype, requires_grad: bool = False
    ) -> SsdTensorHandle:
        r = torch.Tensor._make_wrapper_subclass(cls, shape, dtype=dtype, requires_grad=requires_grad)  # type: ignore
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
        # assert self.tensor is None
        assert self._shape == tensor.shape
        assert self._dtype == tensor.dtype
        self.tensor = tensor

    # if resizing a handle that is part of an ssd buffer, care must be taken that the new size
    # doesn't conflict with adjacent handles!
    def point_to_resized_tensor(self, tensor: torch.Tensor) -> None:
        assert self._dtype == tensor.dtype
        self._shape = tensor.shape
        self.tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        if self.tensor is not None:
            return self.tensor
        else:
            result_tensor = torch.empty(size=self._shape, dtype=self._dtype, requires_grad=self.requires_grad)
            self.copy_into_tensor(result_tensor)
            self.tensor = result_tensor
            return self.tensor

    def to_file(self, permit_when_tensor_none: bool = False, release_tensor_after_write: bool = True) -> None:
        assert self.tensor is not None or permit_when_tensor_none

        if self.tensor is not None:
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

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        ssd_tensor_handles = []

        def unwrap(e: Any) -> torch.Tensor:
            if isinstance(e, SsdTensorHandle):
                t = e.to_tensor()
                ssd_tensor_handles.append((e, t._version))  # type: ignore
                return t
            else:
                return e

        # need to test if tensor is modified
        r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        for e, saved_version in ssd_tensor_handles:
            # TODO: version counter trick doesn't work
            # if saved_version != e.tensor._version:
            #    r.to_file()

            inplace_is_this_tensor = func.__name__[-1] == "_" and e is args[0]
            out_is_this_tensor = False if "out" not in kwargs else e is kwargs["out"]
            if inplace_is_this_tensor or out_is_this_tensor:
                e.to_file()
        return r


# Class supporting a single SSD file backing one or more tensors
class SsdBuffer:
    def __init__(self, num_elems: int, filename: str, dtype: torch.dtype = torch.float32) -> None:
        self.buffer: torch.Tensor = torch.empty((num_elems,), dtype=dtype)
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

    def from_disk(self, num_elems: int, dtype: torch.dtype = torch.float32) -> None:
        if num_elems < self.offset:
            raise RuntimeError(
                f"Attempted to load from file ssdbuffer of size: {self.offset} into a buffer that is of size: {num_elems}"
            )
        self.buffer = torch.empty((num_elems,), dtype=dtype)
        valid_data = self.buffer.narrow(0, 0, self.offset)
        read(valid_data, self.filename)

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


class SsdParameter(torch.nn.Parameter, SsdTensorHandle):
    @classmethod
    def from_tensor(cls: SsdParameter, tensor: SsdTensorHandle) -> SsdParameter:
        r = cls(tensor.shape, tensor.dtype, tensor.requires_grad)
        r.tensor = tensor
        return r

    @staticmethod
    def __new__(
        cls: SsdParameter, shape: Tuple[int, ...], dtype: torch.dtype, requires_grad: bool = True
    ) -> SsdParameter:
        r = SsdTensorHandle._make_wrapper_subclass(cls, shape, dtype=dtype, requires_grad=requires_grad)  # type: ignore

        return r

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, requires_grad: bool = True) -> None:
        super(SsdParameter, self).__init__(shape, dtype, requires_grad)  # type: ignore

class SsdTensorHandleView(SsdTensorHandle):
    @classmethod
    def from_tensor(cls: SsdTensorHandleView, tensor: SsdTensorHandle, view_callback = None) -> SsdTensorHandleView:
        r = super(cls, cls).from_tensor(tensor)
        r.view_callback = view_callback
        return r

    def __init__(self, shape: Tuple[int, ...], dtype: torch.dtype, requires_grad: bool = True) -> None:
        super(SsdTensorHandleView, self).__init__(shape, dtype, requires_grad)  # type: ignore
        self.view_callback = None

    def to_tensor(self) -> torch.Tensor:
        self.view_callback.to_tensor()
        return self.tensor

    def to_file(self, permit_when_tensor_none: bool = False, release_tensor_after_write: bool = True) -> None:
        # PJ TODO: Should we ignore to_file from views?
        pass

class SsdFlatParameter(torch.nn.Parameter, SsdTensorHandle):
    """ A parameter that is initialized from a list of parameters and can be
        turned into a list of views as needed.
    """

    def __new__(cls, params: Sequence[torch.nn.Parameter], filename: str, requires_grad: bool = True) -> "SsdFlatParameter":
        """ Make an object using the parent's __new__ function. """

        # A empty of non-list input doesn't make sense.
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError("An non-empty list or tuple argument is needed")

        # Normally, all items are Parameters. But during pickling, we will have a single
        # Tensor as the input and later in __init__, the correct _param_numels and _param_shapes
        # are set.
        if not all(isinstance(p, (torch.nn.Parameter, torch.Tensor)) for p in params):
            raise ValueError("List items need to be Parameter types")

        # Flattening involves (1) making a tensor flat (i.e. single dimensional) and (2) making a module
        # heirarchy flat (using a single tensor to replace a tree of tensors). Therefore,
        # adding back nesting and heirarchy is counter-productive. If nesting is encountered
        # in the future, the reasonable thing to do is likely for the top level SsdFlatParameter to
        # absorb the nested one and keep the result flat, free from hierarchy.
        if any(isinstance(p, SsdFlatParameter) for p in params):
            raise ValueError("Nesting SsdFlatParameter is not supported")

        dtype = params[0].dtype
        size = sum(p.numel() for p in params)
        r = SsdTensorHandle._make_wrapper_subclass(cls, (size,), dtype=dtype, requires_grad=requires_grad)  # type: ignore
        return r

    def to_tensor(self) -> torch.Tensor:
        if self.tensor is None:
            # Update Views to point to the new tensor
            result = super(SsdFlatParameter, self).to_tensor()
            for view in self.views:
                view.point_to_tensor(self.tensor.narrow(0, view.offset, view._numel).view(view._shape))
        return self.tensor

    def to_file(self, permit_when_tensor_none: bool = False, release_tensor_after_write: bool = True) -> None:
        SsdTensorHandle.to_file(self, permit_when_tensor_none=permit_when_tensor_none, release_tensor_after_write=release_tensor_after_write)
        if release_tensor_after_write:
            for view in self.views:
                view.point_to_file(self.filename, view.offset)

    def __init__(self, params: Sequence[torch.nn.Parameter], filename: str, requires_grad: bool = True):
        """ Initialize the _param_numels and _param_shapes lists. """
        self._param_numels = [p.numel() for p in params]
        total_numels = sum(self._param_numels)
        assert self.numel() <= total_numels, f"Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}"
        self._param_shapes = [p.size() for p in params]

        # These are set by FPW class below, not by this class itself.
        self._param_infos: List[Tuple[str, torch.nn.Module, str]] = []
        self._shared_param_infos: List[Tuple[str, str, torch.nn.Module, str, torch.nn.Module, str]] = []

        super(SsdFlatParameter, self).__init__(shape=(total_numels,), dtype=params[0].dtype, requires_grad=requires_grad)  # type: ignore

        tensor = torch.cat([p.detach().reshape(-1) if isinstance(p, torch.nn.Parameter) else p.reshape(-1) for p in params], 0)
        tensor.requires_grad = requires_grad
        self.set_file_params(filename, 0)
        self.point_to_tensor(tensor)
        self.views: List[SsdTensorHandleView] = []
        offset = 0
        for (size, shape) in zip(self._param_numels, self._param_shapes):
            view = SsdTensorHandleView.from_tensor(self.tensor.narrow(0, offset, size).view(shape), view_callback=self)
            view.set_file_params(filename, offset)
            self.views.append(view)
            offset += size

    def get_param_views(self, external_data: Optional[torch.Tensor] = None, as_tensor_handle_view: bool = False) -> Iterator[torch.Tensor]:
        """ Return a generator of views that map to the original parameters. """
        # Note, self.data could be sharded, so its numel is <= to the sum.
        '''
        assert self.data.numel() <= sum(
            self._param_numels
        ), f"Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}"
        '''
        if external_data:
            if external_data.numel() != sum(self._param_numels):
                raise ValueError(
                    f"Incorrect numel of supplied data: got {external_data.numel()} but expected {sum(self._param_numels)}"
                )
            return (t.view(s) for (t, s) in zip(external_data.split(self._param_numels), self._param_shapes))
        else:
            if as_tensor_handle_view:
                return (v for v in self.views)
            else:
                return (t.view(s) for (t, s) in zip(self.split(self._param_numels), self._param_shapes))

    def metadata(self) -> Tuple[List[str], List[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for this flat parameter."""
        names = [".".join([m, n]) if m else n for (m, _, n) in self._param_infos]
        return names, self._param_shapes, self._param_numels

    def __setstate__(self, state: Tuple[Any, Any, Any, Any]) -> None:
        """ Use by pickle to set the internal states. """
        (self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos) = state
        assert self.numel() <= sum(
            self._param_numels
        ), f"Incorrect pickling {self.numel()} vs. {sum(self._param_numels)}"

    def __reduce_ex__(self, proto: int) -> Tuple[Any, Any, Any]:
        """ Support pickling between ranks. """
        return (
            SsdFlatParameter,  # Callable
            # Args to the callable above
            ([self.data], self.filename, self.requires_grad),
            # Args to __setstate__
            (self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos),
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

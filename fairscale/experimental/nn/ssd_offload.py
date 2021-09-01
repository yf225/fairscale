from __future__ import annotations

import io
import os
import pickle
from typing import IO, Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL as DEFAULT_PROTOCOL

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


def write(t: torch.Tensor, filename: str) -> None:
    num_chunks = _get_num_chunks(t)
    with open(filename, "wb") as f:
        for i in range(num_chunks):
            f.write(_tensor_to_bytes_chunks(t, i))


def read(t: torch.Tensor, filename: str, num_padded: int = None) -> None:
    size_in_bytes = t.nelement() * t.element_size()
    chunk_size_bytes = DEFAULT_CHUNK_SIZE
    num_chunks = _get_num_chunks(t)
    t_np = t.detach().numpy()
    t_mv = memoryview(t_np.view(dtype=np.uint8).reshape(-1))
    fixed_mv = t_mv[0:chunk_size_bytes]
    with io.open(filename, "rb") as f:
        for i in range(num_chunks):
            chunk_start = i * chunk_size_bytes
            chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
            data_read = f.readinto(t_mv[chunk_start:chunk_end])
            assert data_read == chunk_end - chunk_start


# Class supporting a single SSD file backing (possibly) one or
# more tensors, loosely modeled after DeepSpeed's SwapBuffer
class SsdBuffer:
    def __init__(self, buffer: torch.Tensor, filename: str) -> None:
        self.buffer: Optional[torch.Tensor] = buffer
        self.filename = filename
        self.reset()

    def reset(self) -> None:
        self.offset = 0
        self.num_elem = 0
        # tensor including padding at the end for efficient disk read/write
        self.padded_tensors: Dict[int, Tuple[int, Optional[torch.Tensor]]] = {}
        self.tensors: Dict[int, Tuple[int, Optional[torch.Tensor]]] = {}

    def allocate(self, n: int, padded_n: int) -> Tuple[int, Optional[torch.Tensor]]:
        assert self.can_alloc(padded_n)
        assert padded_n >= n
        assert n > 0
        assert self.buffer

        padded_tensor = self.buffer.narrow(0, self.offset, padded_n)
        tensor = padded_tensor.narrow(0, 0, n)

        tensor_offset = self.offset
        self.padded_tensors[tensor_offset] = (padded_n, padded_tensor)
        self.tensors[tensor_offset] = (n, tensor)
        self.offset += padded_n
        self.num_elem += n

        return tensor_offset, tensor

    def insert(self, tensor: torch.Tensor, padded_n: int) -> Tuple[int, Optional[torch.Tensor]]:
        offset, dst_tensor = self.allocate(tensor.numel(), padded_n)
        assert dst_tensor
        dst_tensor.data.copy_(tensor.data)
        return offset, dst_tensor

    def can_alloc(self, n: int) -> bool:
        assert self.buffer
        return (self.offset + n) <= self.buffer.numel()

    def get_tensors(self) -> List[Tuple[int, Optional[torch.Tensor]]]:
        return [(offset, t) for (offset, (n, t)) in self.tensors.items()]

    def get_padded_tensors(self) -> List[Tuple[int, Optional[torch.Tensor]]]:
        return [(offset, t) for (offset, (n, t)) in self.padded_tensors.items()]

    def get_tensor(self, offset: int) -> Optional[torch.Tensor]:
        _, tensor = self.tensors.get(offset, (None, None))
        return tensor

    def get_padded_tensor(self, offset: int) -> Optional[torch.Tensor]:
        _, tensor = self.padded_tensors.get(offset, (None, None))
        return tensor

    def to_disk(self) -> None:
        assert self.buffer
        valid_data = self.buffer.narrow(0, 0, self.offset)
        write(valid_data, self.filename)

        # Remove all Tensor references
        self.tensors = {offset: (n, None) for offset, (n, t) in self.tensors.items()}
        self.padded_tensors = {offset: (n, None) for offset, (n, t) in self.padded_tensors.items()}

        self.buffer = None

    def from_disk(self, buffer: torch.Tensor) -> None:
        if buffer.numel() < self.offset:
            raise RuntimeError(
                f"Attempted to load from file ssdbuffer of size: {self.offset} into a buffer that is of size: {buffer.numel()}"
            )
        self.buffer = buffer
        valid_data = self.buffer.narrow(0, 0, self.offset)
        read(valid_data, self.filename)

        # Restore Tensor References
        self.tensors = {offset: (n, self.buffer.narrow(0, offset, n)) for offset, (n, t) in self.tensors.items()}
        self.padded_tensors = {
            offset: (n, self.buffer.narrow(0, offset, n)) for offset, (n, t) in self.padded_tensors.items()
        }


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

    def _init_loading(self) -> None:
        self.f = io.open(self.filename, "wb")

    def append(self, item: bytes) -> None:
        assert self.f
        self.f.write(item)

    def extend(self, items: List[bytes]) -> None:
        for i in items:
            self.append(i)


torch_saver = TorchSaver()

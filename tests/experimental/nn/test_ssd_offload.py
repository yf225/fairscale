# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Testing SSD Offload Module
"""

import filecmp
import os
import tempfile

import numpy as np
import torch

import fairscale.experimental.nn.ssd_offload as so


def _init():
    torch.manual_seed(0)
    np.random.seed(0)


def test_write_read():
    _init()

    with tempfile.NamedTemporaryFile() as f:
        ref_tensor = torch.rand((128), dtype=torch.float32)
        test_tensor = torch.zeros_like(ref_tensor)
        assert not torch.equal(ref_tensor, test_tensor)
        so.write(ref_tensor, f.name)
        so.read(test_tensor, f.name)
        assert torch.equal(ref_tensor, test_tensor)


def test_ssd_buffer_basic():
    _init()
    with tempfile.NamedTemporaryFile() as f:
        refa_tensor = torch.rand((128), dtype=torch.float32)
        refb_tensor = torch.rand((128), dtype=torch.float32)
        refc_tensor = torch.rand((128), dtype=torch.float32)
        buffer = torch.empty((1024), dtype=torch.float32)
        ssd_buf = so.SsdBuffer(buffer, f.name)

        a_off, buf_a = ssd_buf.insert(refa_tensor, refa_tensor.numel())
        b_off, buf_b = ssd_buf.insert(refb_tensor, refb_tensor.numel())
        c_off, buf_c = ssd_buf.insert(refc_tensor, refc_tensor.numel())

        assert torch.equal(refa_tensor, buf_a)
        assert torch.equal(refb_tensor, buf_b)
        assert torch.equal(refc_tensor, buf_c)

        assert ssd_buf.get_tensor(a_off) is buf_a
        assert ssd_buf.get_tensor(b_off) is buf_b
        assert ssd_buf.get_tensor(c_off) is buf_c

        assert a_off == 0
        assert b_off == 128
        assert c_off == 256

        # remove references so memory will be cleaned up
        buffer = None
        buf_a = None
        buf_b = None
        buf_c = None

        ssd_buf.to_disk()

        assert ssd_buf.get_tensor(a_off) is None
        assert ssd_buf.get_tensor(b_off) is None
        assert ssd_buf.get_tensor(c_off) is None

        buffer = torch.empty((384), dtype=torch.float32)
        ssd_buf.from_disk(buffer)

        new_buf_a = ssd_buf.get_tensor(a_off)
        new_buf_b = ssd_buf.get_tensor(b_off)
        new_buf_c = ssd_buf.get_tensor(c_off)

        assert torch.equal(refa_tensor, new_buf_a)
        assert torch.equal(refb_tensor, new_buf_b)
        assert torch.equal(refc_tensor, new_buf_c)


def test_torch_save_load():
    _init()
    orig_file = tempfile.NamedTemporaryFile()
    checkpoint_file = tempfile.NamedTemporaryFile()

    # TENSOR_SHAPE = (1024, 1024, 1024)
    # use smaller shape for unit tests
    TENSOR_SHAPE = (1024, 1024)
    ref_tensor = torch.rand(TENSOR_SHAPE, dtype=torch.float32)
    ref_ssd_tensor = so.SsdTensor.fromtensor(ref_tensor, orig_file.name)
    del ref_tensor
    # after deleting ref_tensor, memory usage should be very low
    # For save it shouldn't be more than 10x so.DEFAULT_CHUNK_SIZE
    so.torch_saver.save(ref_ssd_tensor, checkpoint_file.name)
    # below line saves file to orig_file.name+"_2"
    # Memory usage here should be O(1000 * so.DEFAULT_CHUNK_SIZE)
    # 1000x because that's how many elements the python unpickler
    # will buffer before passing to the SsdTensor
    test_ssd_tensor = torch.load(checkpoint_file)
    assert filecmp.cmp(orig_file.name, orig_file.name + "_2", shallow=False)
    os.unlink(orig_file.name + "_2")

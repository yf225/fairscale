import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# flake8: noqa
# type: ignore


@triton.jit
def kernel(out, **meta):
    pid = tl.program_id(axis=0)

    v = float(1.1)
    v = v.to(tl.float16)
    c = tl.zeros((128, 128), tl.float16)
    a = tl.zeros((128, 256), tl.float16) + v
    b = tl.zeros((256, 128), tl.float16) + v
    for i in range(0, 1):
        # c += v
        c += tl.dot(a, b).to(tl.float16)

    epilog = meta["EPILOG"]
    epilog(out, c)


@triton.jit
def epilog(out, c):
    x = c + tl.zeros((128, 128), tl.float16)
    x = tl.max(x, axis=1)

    out_ptrs = out + tl.arange(0, 128)
    tl.store(out_ptrs, x)


def test():

    out = torch.ones((128,), dtype=torch.float16, device="cuda")

    def grid(meta):
        return (1,)

    kernel[grid](out, EPILOG=epilog, num_stages=1, num_warps=1)
    print(out)


test()

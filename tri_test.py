import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# flake8: noqa
# type: ignore


@triton.jit
def kernel(out, **meta):
    pid = tl.program_id(axis=0)

    c = tl.zeros((5, 5), tl.float16)
    for i in range(0, 5):
        a = tl.zeros((5, 6), tl.float16)
        b = tl.zeros((6, 5), tl.float16)
        c += tl.dot(a, b)
        c += 1.1

    epilog = meta["EPILOG"]
    epilog(out, c)


@triton.jit
def epilog(out, c):
    x = c + tl.zeros((5, 5), tl.float16)
    x = tl.max(x, axis=1)

    out_ptrs = out + tl.arange(0, 5)
    tl.store(out_ptrs, x)


def test():

    out = torch.ones((5,), dtype=torch.float16, device="cuda")

    def grid(meta):
        return (1,)

    kernel[grid](out, EPILOG=epilog)
    print(out)


test()

import os
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo
from torch.utils.cpp_extension import load

lib = load(
    name="mat_transpose_lib",
    sources=[str(Path(__file__).resolve().with_name("mat_transpost.cu"))],
    extra_cuda_cflags=[        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",],
    extra_cflags=["-std=c++17"],
)

def run_benchmark(    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
    ):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)    
    else:
        for i in range(warmup):
            perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    real_t = f"{out.T.equal(x)}"
    out_val = out[:2, :2].flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(
        f"{out_info:>35}: {out_val}, validate {real_t:<5}, time:{mean_time:.8f}ms"
    )
    if show_all:
        print(out)
    return out, mean_time


Ms = [1024, 2048, 4096, 8192]
Ns = [1024, 2048, 4096, 8192]

MNs = [(M, N) for M in Ms for N in Ns]
copy_x = lambda x: x.clone()
for M, N in MNs:
    print("-" * 130)
    print(" " * 55 + f"M={M}, N={N}")
    x = torch.arange(0, M * N).reshape(M, N).cuda().float().contiguous()
    y = torch.randn((N, M)).cuda().float().contiguous()
    run_benchmark(partial(copy_x), x, "original")
    run_benchmark(lib.mat_transpose_f32_col2row, x, "f32_col2row", y)
    run_benchmark(lib.mat_transpose_f32_diagnose, x, "f32_diagnose", y)
    run_benchmark(lib.mat_transpose_f32_shared, x, "f32_shared", y)

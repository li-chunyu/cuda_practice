import time

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

lib = load(
    name="dot_product_lib",
    sources=["dot_product.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
):
    # torch.dot vs custom dot_prod kernel
    for i in range(warmup):
        out = perf_func(a, b)  # warmup
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.item()
    if tag.startswith("i8"):
        print(f"{out_info:>17}: {out_val:<15}, time:{mean_time:.8f}ms")
    else:
        print(f"{out_info:>17}: {out_val:<15.8f}, time:{mean_time:.8f}ms")
    return out, mean_time


Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
Ss = [2,4,8,32,64,128]
Ks = [2,4,8,32,64,128]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    a = torch.randn((S * K)).cuda().float()
    b = torch.randn((S * K)).cuda().float()
    print("ruth dot : ",a.shape, b.shape, a.dot(b), lib.dot_product_fp32x4(a,b))
    # print(a.shape)
    # run_benchmark(lib.dot_prdocut, a, b, "f32f32")


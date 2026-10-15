#include "cuda_runtime.h"
#include <torch/extension.h>

#define  WRAP_SIZE 32


__device__ __forceinline__ float wrap_reduce(float v) {
    for (int lanemask = 1; lanemask < WRAP_SIZE; lanemask = lanemask << 1) {
        v += __shfl_xor_sync(0xffffffff, v, lanemask);
    }
    return v;
}

__global__ void dot_product_fp32_kernel(float* a, float* b, float* y, int N) {
    int global_x = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = (global_x < N) ? a[global_x] * b[global_x] : 0.0;
    sum = wrap_reduce(sum);
    constexpr int WRAP_NUM = (1024 + WRAP_SIZE - 1) / WRAP_SIZE;
    __shared__ float smem_wrap_reduce[WRAP_NUM];
    int wrap = threadIdx.x / WRAP_SIZE;
    int lane = threadIdx.x % WRAP_SIZE;
    // if (blockIdx.x == 0) {
    //     printf("wrap = %d, lane = %d\n", wrap, lane);
    // }
    if (lane == 0) {
        smem_wrap_reduce[wrap] = sum;
    }
    __syncthreads();
    sum = (lane < WRAP_NUM) ? smem_wrap_reduce[lane] : 0.0;
    if (wrap == 0) {
        sum = wrap_reduce(sum);
    }
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}


__global__ void dot_product_fp32x4_kernel(float* a, float* b, float* y, int N) {
    int global_x = blockDim.x * blockIdx.x + threadIdx.x;

    float4* a4 = reinterpret_cast<float4*>(a);
    float4* b4 = reinterpret_cast<float4*>(b);
    float sum = 0.0;
    // N = 8, x=0 -> 0,  x = 1 -> 4 , x=2 -> 8
    if (global_x * 4 + 3 < N) {
        float4 a4_val = a4[global_x];
        float4 b4_val = b4[global_x];
        sum = a4_val.x * b4_val.x + a4_val.y * b4_val.y + a4_val.z * b4_val.z + a4_val.w * b4_val.w;
    }

    sum = wrap_reduce(sum);
    constexpr int WRAP_NUM = (256 + WRAP_SIZE - 1) / WRAP_SIZE;
    __shared__ float smem_wrap_reduce[WRAP_NUM];
    int wrap = threadIdx.x / WRAP_SIZE;
    int lane = threadIdx.x % WRAP_SIZE;

    if (lane == 0) {
        smem_wrap_reduce[wrap] = sum;
    }
    __syncthreads();
    sum = (lane < WRAP_NUM) ? smem_wrap_reduce[lane] : 0.0;
    if (wrap == 0) {
        sum = wrap_reduce(sum);
    }
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}

torch::Tensor dot_product_fp32x4(torch::Tensor a, torch::Tensor b) {
    int N = a.size(0);

    dim3 block(256);
    dim3 grid((N+(256*4) - 1) / (256*4));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    auto y = torch::zeros({1}, options);        
    
    dot_product_fp32x4_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}


torch::Tensor dot_product_fp32(torch::Tensor a, torch::Tensor b) {
    int N = a.size(0);
    dim3 block(1024);
    dim3 grid((N+1024 - 1) / 1024);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    auto y = torch::zeros({1}, options);        
    
    dot_product_fp32_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_prdocut", &dot_product_fp32,
          "reduce (CUDA)");
    m.def("dot_product_fp32x4", &dot_product_fp32x4, "");
}
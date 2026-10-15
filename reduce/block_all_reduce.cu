
#include <torch/extension.h>
#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "cuda_runtime.h"
#include "torch/types.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#define WRAP_SIZE 32

/*
lane=0, 0 xor 1 = 1
lane=1, 1 xor 1 = 0

lane 2, 2 xor 1 = 3
lane 3, 3 xor 1 = 2


lane=0, 00 xor 10 = 2

0 1
0 2
0 4
0 8
0 16
0 32 

*/
__device__ __forceinline__ float wrap_reduce(float val) {
    #pragma unroll
    for (int lanemask = 1; lanemask < WRAP_SIZE; lanemask = lanemask << 1) {
        val += __shfl_xor_sync(0xffffffff, val, lanemask);
    }
    return val;
}

// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 1024>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
    constexpr int num_wrap = (1024 + WRAP_SIZE - 1) / WRAP_SIZE;
    __shared__ float reduce_smem[num_wrap];
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = (global_tid < N) ? a[global_tid]: 0.0;
    sum = wrap_reduce(sum);
    int wrap_idx = threadIdx.x / WRAP_SIZE;
    int lane = threadIdx.x % WRAP_SIZE;
    if (lane == 0) {
        reduce_smem[wrap_idx] = sum;
    }
    __syncthreads();
    sum = (lane < num_wrap) ? reduce_smem[lane]: 0.0;
    if (wrap_idx == 0) {
        sum = wrap_reduce(sum);
    }
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}

torch::Tensor block_all_reduce_sum_f32_f32(torch::Tensor a) {

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    auto y = torch::zeros({1}, options);        
    
    const int S = a.size(0);
    const int M = a.size(1); 

    const int N = S*M;
    dim3 block(1024);
    dim3 grid((N+1024 -1 )/ 1024);
    block_all_reduce_sum_f32_f32_kernel<<<grid, block>>>
        (reinterpret_cast<float*>(a.data_ptr()), reinterpret_cast<float*>(y.data_ptr()), N);
    return y;
}

__global__ void block_all_reduce_f32x4_f32_sum(float *a, float* y, int N) {
    float4*   a4 = reinterpret_cast<float4*>(a);
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if (global_idx < N /4) {
        float4 a_val = a4[global_idx];
        sum = a_val.x + a_val.y + a_val.w + a_val.z;
    } else {
        sum = 0.0;
    }
    sum = wrap_reduce(sum);
    constexpr int wrap_num = (1024+WRAP_SIZE -1) / WRAP_SIZE;
    __shared__ float smem_sum_reduce[wrap_num];
    int wrap = threadIdx.x / WRAP_SIZE;
    int lane = threadIdx.x % WRAP_SIZE;
    if (lane == 0) {
        smem_sum_reduce[wrap] = sum;
    }
    __syncthreads();
    sum = (lane < wrap_num)? smem_sum_reduce[lane]:0.0;
    if (wrap == 0) {
        sum = wrap_reduce(sum);
    }
    if (threadIdx.x == 0) {
        atomicAdd(y, sum);
    }
}

torch::Tensor block_all_reduce_sum_f32x4_f32(torch::Tensor a) {

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
    auto y = torch::zeros({1}, options);        
    
    const int S = a.size(0);
    const int M = a.size(1); 

    const int N = S*M;
    dim3 block(1024);
    
    dim3 grid((N/4 + 1024 -1 )/ 1024);
    block_all_reduce_f32x4_f32_sum<<<grid, block>>>
        (reinterpret_cast<float*>(a.data_ptr()), reinterpret_cast<float*>(y.data_ptr()), N);
    return y;
}

torch::Tensor block_reduce(torch::Tensor a) {
    return block_all_reduce_sum_f32x4_f32(a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_all_reduce_sum_f32_f32", &block_reduce,
          "reduce (CUDA)");
}
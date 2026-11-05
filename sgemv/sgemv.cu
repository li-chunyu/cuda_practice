#include <cuda_runtime.h>
#include "../utils.h"
#include <torch/extension.h>

__device__ float warp_reduce_sum(float val) {
    for (int lane_mask = 1; lane_mask < WARP_SIZE; lane_mask = lane_mask << 1) {
        val += __shfl_xor_sync(0xffffffff, val, lane_mask);
    }
    return val;
}

__global__ void sgemv_f32_kernel(float* a, float* x, float* y, int M, int K) {
    int m = blockDim.y * blockIdx.x + threadIdx.y;
    int lane_idx = threadIdx.x % WARP_SIZE;
    if (m < M) {
        float sum = 0.0;
        for (int k = threadIdx.x; k < K; k += WARP_SIZE) {
            sum += a[m*K + k] * x[k];
        }
        sum = warp_reduce_sum(sum);
        if (lane_idx == 0) {
            y[m] = sum;
        }
    }
}

__global__ void sgemv_pack_f32_kernel(float* a, float* x, float* y, int M, int K) {
    int m = blockIdx.x * blockDim.x + threadIdx.y;
    int lane_idx = threadIdx.x % WARP_SIZE;
    if (m < M) {
        float sum = 0.0f;
        int tid = threadIdx.x;
        for (int k = tid * 4; k < K; k += (WARP_SIZE*4)) {
            float4 a_val = FLOAT4(a)[m*K + tid];
            float4 x_val = FLOAT4(x)[tid];
            sum += (a_val.x * x_val.x + a_val.y * x_val.y + a_val.z * x_val.z + a_val.w * x_val.w);
        }
        sum = warp_reduce_sum(sum);
        if (lane_idx == 0) {
            y[m] = sum;
        }
    }
}

void sgemv_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
    int M = a.size(0);
    int K = x.size(0);
    dim3 block(WARP_SIZE, 4);
    dim3 grid((M+4 - 1) / 4);
    sgemv_f32_kernel<<<grid, block>>>(FLOAT_DATA_PTR(a.data_ptr()), FLOAT_DATA_PTR(x.data_ptr()), FLOAT_DATA_PTR(y.data_ptr()), M, K);
}

void sgemv_pack_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
    int M = a.size(0);
    int K = x.size(0);
    dim3 block(WARP_SIZE, 4);
    dim3 grid((M+4 - 1) / 4);
    sgemv_pack_f32_kernel<<<grid, block>>>(FLOAT_DATA_PTR(a.data_ptr()), FLOAT_DATA_PTR(x.data_ptr()), FLOAT_DATA_PTR(y.data_ptr()), M, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemv_f32", &sgemv_f32, "");
}
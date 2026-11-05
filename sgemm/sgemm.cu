#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#include "../utils.h"

__global__ void sgemm_naive_f32_kernel(float* a, float* b, float* c, int M, int K, int N) {
    int c_x = blockDim.x * blockIdx.x + threadIdx.x;
    int c_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (c_y < M && c_x < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += a[c_y*K + k] * b[k * N + c_x];
        }
        c[N * c_y + c_x] = sum;
    }
}

void sgemm_naive_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    dim3 block(32, 32);
    dim3 grid((N+32 - 1)/ 32, (M+32 - 1)/ 32);

    sgemm_naive_f32_kernel<<<grid, block>>>(FLOAT(a.data_ptr()), FLOAT(b.data_ptr()), FLOAT(c.data_ptr()), M, K, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_naive_f32", &sgemm_naive_f32, "");
}
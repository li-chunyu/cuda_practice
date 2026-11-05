#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 256

__global__ void rope_f32_kernel(float *x, float *out, int seq_len, int N) {
    int global_n = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_n * 2 + 1 < seq_len * N * 2) {
        int p = global_n / N;
        int i = global_n % N;
        float x0 = x[global_n*2] ;
        float x1 = x[global_n*2 + 1];
        float theta = __fdividef(p, powf(10000.0f, 2.0f * i / (2*N)));
        float c = cosf(theta);
        float s = sinf(theta);
        float x0_r = x0 * c - x1 * s;
        float x1_r = x0 * s + x1 * c;
        out[global_n*2] = x0_r;
        out[global_n*2 + 1] = x1_r;
    }
}

void rope_fp32(torch::Tensor x, torch::Tensor out) {
  int seq_len = x.size(0);
  int hiddent_size = x.size(1);
  int N = hiddent_size / 2;
  dim3 block(BLOCK_SIZE);
  dim3 grid((seq_len * N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  rope_f32_kernel<<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),
                                   reinterpret_cast<float *>(out.data_ptr()), seq_len, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_fp32", &rope_fp32, "");
}
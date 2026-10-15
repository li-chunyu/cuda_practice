#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

__global__ void elementwise_add_f32_kernel(const float *a, const float *b,
                                         float *c, int64_t n) {
    const int64_t index = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

__global__ void elementwise_add_f32x4_kernel(const float *a, const float* b, float* c, int64_t N) {
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    const int64_t vector_index = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t index = vector_index * 4;
    if (index + 3 < N) {
        float4 x = a4[vector_index];
        float4 y = b4[vector_index];
        float4 z;

        z.x = x.x + y.x;
        z.y = x.y + y.y;
        z.z = x.z + y.z;
        z.w = x.w + y.w;
        c4[vector_index] = z;
    } else if (index < N) {
        for (int i = 0; i < N-index; i++) {
            c[index+i] = a[index+i] + b[index+i];
        }
    }
}

static void check_inputs(const torch::Tensor& a, const torch::Tensor& b,
                         const torch::Tensor& c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
                "a, b and c must be CUDA tensors");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
                "a, b and c must be on the same CUDA device");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                b.scalar_type() == torch::kFloat32 &&
                c.scalar_type() == torch::kFloat32,
                "a, b and c must have dtype float32");
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(),
                "a, b and c must have the same shape");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "a, b and c must be contiguous");
}

void elementwise_add_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    check_inputs(a, b, c);
    const int64_t n = a.numel();
    if (n == 0) {
        return;
    }
    const c10::cuda::CUDAGuard device_guard(a.device());
    constexpr int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    elementwise_add_f32_kernel<<<blocks, threads, 0,
                                c10::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void elementwise_add_f32x4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    check_inputs(a, b, c);
    const int64_t n = a.numel();
    if (n == 0) {
        return;
    }

    // Contiguous views may have storage offsets that are not float4-aligned.
    if (reinterpret_cast<std::uintptr_t>(a.data_ptr<float>()) % alignof(float4) != 0 ||
        reinterpret_cast<std::uintptr_t>(b.data_ptr<float>()) % alignof(float4) != 0 ||
        reinterpret_cast<std::uintptr_t>(c.data_ptr<float>()) % alignof(float4) != 0) {
        elementwise_add_f32(a, b, c);
        return;
    }

    const c10::cuda::CUDAGuard device_guard(a.device());
    constexpr int threads = 256;
    const int64_t blocks = (n + threads * 4 - 1) / (threads * 4);
    elementwise_add_f32x4_kernel<<<blocks, threads, 0,
                                  c10::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_add_f32", &elementwise_add_f32,
          "Elementwise FP32 addition into an output tensor (CUDA)");
    m.def("elementwise_add_f32x4", &elementwise_add_f32x4,
          "Vectorized elementwise FP32 addition into an output tensor (CUDA)");
}

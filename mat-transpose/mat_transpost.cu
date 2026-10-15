#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>

#define WARP_SIZE_S 32

// FP32
// col2row means read x[row][col] and write y[col][row]
__global__ void mat_transpose_f32_col2row_kernel(float *x, float *y,
                                                 const int row, const int col) {
    int global_x = blockDim.x * blockIdx.x + threadIdx.x;
    int global_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (global_x < col && global_y < row) {
        y[global_x * row + global_y] = x[global_y * col + global_x];
    }
}

__global__ void mat_transpose_f32_diagnose_kernel(float* x, float* y, const int row, const int col) {
    int tile_y = blockIdx.x;
    int tile_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    int global_x = tile_x * blockDim.x + threadIdx.x;
    int global_y = tile_y * blockDim.y + threadIdx.y;

    if (global_x < col && global_y < row) {
        y[global_x * row + global_y] = x[global_y * col + global_x];
    }
}

// must row == col
__global__ void mat_transpose_f32_shared(float* x, float* y, const int row, const int col) {
    int tile_y = blockIdx.x;
    int tile_x = (blockIdx.x + blockIdx.y) % gridDim.x;

    int global_x = tile_x * blockDim.x + threadIdx.x;
    int global_y = tile_y * blockDim.y + threadIdx.y;

    __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S+1];
    if (global_x < col && global_y < row) {
        // y[global_x * row + global_y] = x[global_y * col + global_x];
        tile[threadIdx.y][threadIdx.x] = x[global_y * col + global_x];
    }

    // 所有线程必须经过这里
    __syncthreads();

    // 2. 转置后的输出坐标
    int out_y =
        tile_x * blockDim.x + threadIdx.y;

    int out_x =
        tile_y * blockDim.y + threadIdx.x;

    // 3. Store 使用自己的边界判断
    if (out_y < col && out_x < row) {
        y[out_y * row + out_x] =
            tile[threadIdx.x][threadIdx.y];
    }
}

// TODO：写一个可以处理 col 不能被4整除的 kernel.
__global__ void mat_transpose_f32x4(float* x, float* y, const int row, const int col) {
    // dim3 block(32,32);
    // dim3 grid((n/4+1))
    const float4* x4 = reinterpret_cast<float4*>(x);
    const float4* y4 = reinterpret_cast<float4*>(y);
    int global_x = (blockDim.x * blockIdx.x + threadIdx.x);
    int global_y = (blockDim.y * blockIdx.y + threadIdx.y);
    int col4 = (col / 4) + (col % 4 > 0? 1: 0);
// 当前实现按每行从第 0 列开始，每 4 个元素用 float4 读取，
// 因此要求 x 起始地址按 16 字节对齐，且 col 能被 4 整除。
// 否则需要额外处理对齐和尾部；简单做法是在 launch 时选择标量 kernel。
    if (global_x*4 + 3 < col && global_y < row) {
        float4 x_val = x4[global_y * col / 4 + global_x];
        float4 y_val;
        y[(global_x * 4)*row + global_y] = x_val.x;
        y[(global_x * 4 + 1)*row + global_y] = x_val.y;
        y[(global_x * 4 + 2)*row + global_y] = x_val.z;
        y[(global_x * 4 + 3)*row + global_y] = x_val.w;
    }
}

__global__ void mat_transpose_f32x4_shared(float* x, float* y, const int row, const int col) {
    float4* x4 = reinterpret_cast<float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);
    int global_x = blockDim.x * blockIdx.x + threadIdx.x;
    int global_y = blockDim.y * blockIdx.y + threadIdx.y;

    int tile_x = blockDim.x * blockIdx.x;
    int tile_y = blockDim.y * blockIdx.y;
    __shared__ float xtile[WARP_SIZE_S][WARP_SIZE_S*4];
    if (4  * global_x + 3 < col && global_y < row) {
        float4 x_val = x4[col / 4 * global_y + global_x];
        xtile[threadIdx.y][4*threadIdx.x] = x_val.x;
        xtile[threadIdx.y][4*threadIdx.x + 1] = x_val.y;
        xtile[threadIdx.y][4*threadIdx.x + 2] = x_val.z;
        xtile[threadIdx.y][4*threadIdx.x + 3] = x_val.w;
    }
    __syncthreads();

    // X[tile_x + (threadIdx.y/4)*4 + k][4*tile_x + 4*threadIdx.x + threadIdx.y % 4];
    int out_y = 4*tile_x + 4*threadIdx.x + threadIdx.y % 4;
    int out_x = tile_x + (threadIdx.y/4)*4; // + 0 1 2 3

    if (out_y < col && out_x + 3 < row) {
        float4 smemval;
        smemval.x = xtile[(threadIdx.y/4)*4 + 0][4*threadIdx.x + threadIdx.y % 4];
        smemval.y = xtile[(threadIdx.y/4)*4 + 1][4*threadIdx.x + threadIdx.y % 4];
        smemval.z = xtile[(threadIdx.y/4)*4 + 2][4*threadIdx.x + threadIdx.y % 4];
        smemval.w = xtile[(threadIdx.y/4)*4 + 3][4*threadIdx.x + threadIdx.y % 4];
        y4[out_y*(row/4) + out_x/4] = smemval;
    }

}

void mat_transpose_f32_col2row(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");
    TORCH_CHECK(x.device() == y.device(), "x and y must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                y.scalar_type() == torch::kFloat32,
                "x and y must have dtype float32");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "x and y must be 2D tensors");
    TORCH_CHECK(y.size(0) == x.size(1) && y.size(1) == x.size(0),
                "y must have shape (x.size(1), x.size(0))");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                "x and y must be contiguous");
    if (x.numel() == 0) {
        return;
    }
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(),
                "matrix is too large for 32-bit kernel indexing");
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(WARP_SIZE_S, WARP_SIZE_S);
    dim3 grid((N - 1) / WARP_SIZE_S + 1, (M - 1) / WARP_SIZE_S + 1);
    mat_transpose_f32_col2row_kernel<<<grid, block, 0,
                                     c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void mat_transpose_f32_diagnose(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");
    TORCH_CHECK(x.device() == y.device(), "x and y must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                y.scalar_type() == torch::kFloat32,
                "x and y must have dtype float32");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "x and y must be 2D tensors");
    TORCH_CHECK(y.size(0) == x.size(1) && y.size(1) == x.size(0),
                "y must have shape (x.size(1), x.size(0))");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                "x and y must be contiguous");
    if (x.numel() == 0) {
        return;
    }
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(),
                "matrix is too large for 32-bit kernel indexing");
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(WARP_SIZE_S, WARP_SIZE_S);
    dim3 grid((N - 1) / WARP_SIZE_S + 1, (M - 1) / WARP_SIZE_S + 1);
    mat_transpose_f32_diagnose_kernel<<<grid, block, 0,
                                     c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mat_transpose_f32_shared_launcher(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "x and y must be CUDA tensors");
    TORCH_CHECK(x.device() == y.device(), "x and y must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                y.scalar_type() == torch::kFloat32,
                "x and y must have dtype float32");
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2, "x and y must be 2D tensors");
    TORCH_CHECK(y.size(0) == x.size(1) && y.size(1) == x.size(0),
                "y must have shape (x.size(1), x.size(0))");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                "x and y must be contiguous");
    if (x.numel() == 0) {
        return;
    }
    TORCH_CHECK(x.numel() <= std::numeric_limits<int>::max(),
                "matrix is too large for 32-bit kernel indexing");
    const c10::cuda::CUDAGuard device_guard(x.device());
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(WARP_SIZE_S, WARP_SIZE_S);
    dim3 grid((N - 1) / WARP_SIZE_S + 1, (M - 1) / WARP_SIZE_S + 1);
    mat_transpose_f32_shared<<<grid, block, 0,
                             c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), M, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mat_transpose_f32_col2row", &mat_transpose_f32_col2row,
          "Transpose a float32 CUDA matrix into an output tensor");
    m.def("mat_transpose_f32_diagnose", &mat_transpose_f32_diagnose,
          "Transpose a float32 CUDA matrix into an output tensor");
    m.def("mat_transpose_f32_shared", &mat_transpose_f32_shared_launcher,
          "Transpose a float32 CUDA matrix using shared memory into an output tensor");
}

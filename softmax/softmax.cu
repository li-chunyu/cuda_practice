#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <math_constants.h>

// block i proc x[i, :]
// 
// npart = (col +tnum -1) / tnum , use a stride loop
// 
/*
1. max reduce
2. sum reduce
3. e(xi - max) / esum
*/

#define WRAP_SIZE 32

__device__ float wrap_reduce_max(float v) {
    #pragma unroll
    for (int lanemask = 1; lanemask < WRAP_SIZE; lanemask = lanemask << 1) {
        v = max(__shfl_xor_sync(0xffffffff, v, lanemask), v);
    }
    return v;
}

__device__ float block_all_reduce_max(float v) {
    constexpr int NUM_WRAPS = (256 + WRAP_SIZE - 1) / WRAP_SIZE;    
    v = wrap_reduce_max(v);
    __shared__ float smem_wrap[NUM_WRAPS];
    __shared__ float smem_ret;
    int wrap = threadIdx.x / WRAP_SIZE;
    int lane = threadIdx.x % WRAP_SIZE;
    if (lane == 0) {
        smem_wrap[wrap] = v;
    }
    __syncthreads();
    if (wrap == 0) {
        v = lane < NUM_WRAPS ? smem_wrap[lane] : -CUDART_INF_F;
    }
    if (wrap == 0 ) {
        v = wrap_reduce_max(v);
        if (lane == 0) {
            smem_ret = v;
        }
    }
    __syncthreads();
    return smem_ret;
}

__device__ float wrap_reduce_sum(float v) {
    #pragma unroll
    for (int lanemask = 1; lanemask < WRAP_SIZE; lanemask = lanemask << 1) {
        v += __shfl_xor_sync(0xffffffff, v, lanemask);
    }
    return v;
}

__device__ float block_all_reduce_sum(float v) {
    constexpr int NUM_WRAPS = (256 + WRAP_SIZE - 1) / WRAP_SIZE;    
    v = wrap_reduce_sum(v);
    __shared__ float smem_wrap[NUM_WRAPS];
    __shared__ float smem_ret;
    int wrap = threadIdx.x / WRAP_SIZE;
    int lane = threadIdx.x % WRAP_SIZE;
    if (lane == 0) {
        smem_wrap[wrap] = v;
    }
    __syncthreads();
    if (wrap == 0) {
        v = lane < NUM_WRAPS ? smem_wrap[lane] : 0.0;
    }
    if (wrap == 0 ) {
        v = wrap_reduce_sum(v);
        if (lane == 0) {
            smem_ret = v;
        }
    }
    __syncthreads();
    return smem_ret;
}

__global__ void safe_softmax_fp32_kernel(float* x, float* out, int col, int row) {
    float max_of_row = -CUDART_INF_F;

    for (int i = threadIdx.x; i < col; i += blockDim.x) {
        float val = x[blockIdx.x * col + i];
        max_of_row = std::max(max_of_row, val);
    }
    // __syncthreads();

    max_of_row = block_all_reduce_max(max_of_row);

    float sumexp_of_row = 0.0;
    for (int i = threadIdx.x; i < col; i += blockDim.x) {
        float val = x[blockIdx.x * col + i];
        sumexp_of_row += std::exp(val - max_of_row);
    }
    // __syncthreads();
    sumexp_of_row = block_all_reduce_sum(sumexp_of_row);

    for (int i = threadIdx.x; i < col; i += blockDim.x) {
        int idx = blockIdx.x * col + i;
        out[idx] = std::exp(x[idx] - max_of_row) / sumexp_of_row;
    }
}

struct MD {
    float m;
    float d;
};

__device__ MD wrap_reduce_md_op(MD val) {
    #pragma unroll
    for (int lanemask = 1; lanemask < WRAP_SIZE; lanemask = lanemask << 1) {
        MD other ;
        other.m = __shfl_xor_sync(0xffffffff, val.m, lanemask);
        other.d = __shfl_xor_sync(0xffffffff, val.d, lanemask);
        bool value_bigger = val.m > other.m;
        MD smaller_md = value_bigger ? other : val;
        MD bigger_md = value_bigger ? val : other;
        val.m = bigger_md.m;
        val.d = bigger_md.d + __expf(smaller_md.m - bigger_md.m) * smaller_md.d;
    }
    return val;
};

template<int NUM_THREAD=1024>
__global__ void online_safe_softmax_per_token_kernel(float* x, float* out, int col, int row) {
    constexpr int NUM_WRAP = (NUM_THREAD + WRAP_SIZE - 1) / WRAP_SIZE;
    __shared__ MD smem_wrap_reduce_md[NUM_WRAP];
    int wrap = threadIdx.x / WRAP_SIZE;
    int lane = threadIdx.x % WRAP_SIZE;
    float val = x[blockIdx.x * col + threadIdx.x];
    MD md;
    md.m = val;
    md.d = 1;
    md = wrap_reduce_md_op(md);
    if (lane == 0) {
        smem_wrap_reduce_md[wrap] = md;
    }
    __syncthreads();
    if (wrap == 0) {
        md = lane < NUM_WRAP ? smem_wrap_reduce_md[lane] : MD{.m = -FLT_MAX, .d = 0};
        md = wrap_reduce_md_op(md);
        if (lane == 0) {
            smem_wrap_reduce_md[0] = md;
        }
    }
    __syncthreads();
    MD final_md = smem_wrap_reduce_md[0];
    float final_d_reverse = __fdividef(1.0, final_md.d);
    out[blockIdx.x * col + threadIdx.x] = (__expf(x[blockIdx.x * col + threadIdx.x] - final_md.m)) * final_d_reverse;
}

void online_safe_softmax_per_token(torch::Tensor x, torch::Tensor out) {
    int row = x.size(0); // token
    int col = x.size(1); // head

    dim3 grid(row);
    dim3 block(col);
    online_safe_softmax_per_token_kernel<<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()),
        reinterpret_cast<float*>(out.data_ptr()), col, row);
}

void safe_softmax_fp32(torch::Tensor x, torch::Tensor out) {
    int row = x.size(0);
    int col = x.size(1);
    dim3 grid(row);
    dim3 block(256);
    safe_softmax_fp32_kernel<<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()),
        reinterpret_cast<float*>(out.data_ptr()), col, row);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("safe_softmax", &safe_softmax_fp32, "");
    m.def("online_safe_softmax_fp32", 
        &online_safe_softmax_per_token, "");
}
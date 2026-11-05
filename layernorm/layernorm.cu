#include <cuda_runtime.h>
#include <torch/extension.h>

#define WRAP_SIZE 32

__device__ float wrap_reduce_sum(float val) {
    for (int mask = 1 ; mask < WRAP_SIZE; mask = mask << 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <int NUM_THREAD = 512>
__global__ void layer_norm_f32_kernel(float *x, float *out, float g, float b) {
    float val = x[blockDim.x * blockIdx.x + threadIdx.x];
    constexpr int NUM_WRAP = (NUM_THREAD + WRAP_SIZE - 1) / WRAP_SIZE;
    int lane_id = threadIdx.x % WRAP_SIZE;
    int wrap_id = threadIdx.x / WRAP_SIZE;
    __shared__ float smem_wrap_sum[NUM_WRAP];

    val = wrap_reduce_sum(val);
    if (wrap_id == 0) {
        // printf("first val = %lf\n", val);
    }

    if (lane_id == 0) {
        smem_wrap_sum[wrap_id] = val;
    }
    __syncthreads();
    if (wrap_id == 0) {
        val = lane_id < NUM_WRAP ? smem_wrap_sum[lane_id] : 0.0;
        val = wrap_reduce_sum(val);
        if (lane_id == 0) {
            smem_wrap_sum[0] = val;
            // printf("smem_wrap_sum[0]=%lf\n",  smem_wrap_sum[0]);
        }
    }
    __syncthreads();

    float sum = smem_wrap_sum[0];
    float avg = sum * __fdividef(1, NUM_THREAD);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // printf("avg=%lf, sum=%lf\n", avg, sum);
    }

    float var;
    val = x[blockDim.x * blockIdx.x + threadIdx.x];
    float divid_pow_sum = (val - avg) * (val - avg);
    divid_pow_sum = wrap_reduce_sum(divid_pow_sum);
    if (lane_id == 0 ) {
        smem_wrap_sum[wrap_id] = divid_pow_sum;
    }
    __syncthreads();
    if (wrap_id == 0) {
        divid_pow_sum = lane_id < NUM_WRAP ? smem_wrap_sum[lane_id] : 0.0;
        divid_pow_sum = wrap_reduce_sum(divid_pow_sum);
        if (lane_id == 0) {
            smem_wrap_sum[0] = divid_pow_sum;
        }
    }
    __syncthreads();

    var = __fdividef(smem_wrap_sum[0], NUM_THREAD);

    out[blockIdx.x * blockDim.x + threadIdx.x] = (val - avg) * rsqrtf(var + 1e-5f) * g + b;
}

template<int NUM_THREAD = 512>
__global__ void rms_norm_f32_kernel(float* x, float* out, float gamma) {
    constexpr int NUM_WRAP = (NUM_THREAD + WRAP_SIZE - 1) / WRAP_SIZE;
    float val = x[blockDim.x * blockIdx.x + threadIdx.x];
    int lane_id = threadIdx.x % WRAP_SIZE;
    int wrap_id = threadIdx.x / WRAP_SIZE;
    __shared__ float smem_reduce_sum[NUM_WRAP];
    val = val * val;
    val = wrap_reduce_sum(val);
    if (lane_id == 0) {
        smem_reduce_sum[wrap_id] = val;
    }
    __syncthreads();
    if (wrap_id == 0) {
        val = lane_id < NUM_WRAP ? smem_reduce_sum[lane_id]: 0.0f;
        val = wrap_reduce_sum(val);
        if (lane_id == 0) {
            smem_reduce_sum[0] = val;
        }
    }
    __syncthreads();
    float rms_reverse = rsqrtf(__fdividef(smem_reduce_sum[0], NUM_THREAD));
    out[blockDim.x * blockIdx.x + threadIdx.x] = x[blockDim.x * blockIdx.x + threadIdx.x] * rms_reverse;
}

void layer_norm_f32(torch::Tensor x, torch::Tensor out, float g, float b) {
  int row = x.size(0);
  int col = x.size(1);
  dim3 grid(row);
  dim3 block(col);
  layer_norm_f32_kernel<<<grid, block>>>(
      reinterpret_cast<float *>(x.data_ptr()),
      reinterpret_cast<float *>(out.data_ptr()), g, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layer_norm_f32", &layer_norm_f32, "");
}
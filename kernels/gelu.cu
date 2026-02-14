/*
 * Vectorized GELU activation kernel for GB10 Blackwell (sm_121)
 *
 * Implements the approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Supports: BF16, FP16, FP32 with vectorized memory access
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int BLOCK_SIZE = 256;

// Type conversion helpers
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// GELU approximate constants
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float GELU_COEFF = 0.044715f;

__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x)));
}

// ============================================================================
// Scalar fallback
// ============================================================================
template <typename scalar_t>
__global__ void gelu_kernel_scalar(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = to_float(input[idx]);
        output[idx] = from_float(gelu_approx(val), (scalar_t*)nullptr);
    }
}

// ============================================================================
// Vectorized BF16 kernel
// ============================================================================
__global__ void gelu_kernel_bf16_vec(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const int vec_elements  // total_elements / 2
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_elements) {
        const __nv_bfloat162* vec_in = reinterpret_cast<const __nv_bfloat162*>(input);
        __nv_bfloat162* vec_out = reinterpret_cast<__nv_bfloat162*>(output);

        __nv_bfloat162 v = vec_in[idx];
        float v0 = __bfloat162float(v.x);
        float v1 = __bfloat162float(v.y);

        __nv_bfloat162 result;
        result.x = __float2bfloat16(gelu_approx(v0));
        result.y = __float2bfloat16(gelu_approx(v1));
        vec_out[idx] = result;
    }
}

// ============================================================================
// Vectorized FP16 kernel
// ============================================================================
__global__ void gelu_kernel_fp16_vec(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const int vec_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_elements) {
        const __half2* vec_in = reinterpret_cast<const __half2*>(input);
        __half2* vec_out = reinterpret_cast<__half2*>(output);

        __half2 v = vec_in[idx];
        float v0 = __half2float(v.x);
        float v1 = __half2float(v.y);

        __half2 result;
        result.x = __float2half(gelu_approx(v0));
        result.y = __float2half(gelu_approx(v1));
        vec_out[idx] = result;
    }
}

// ============================================================================
// Vectorized FP32 kernel (float4)
// ============================================================================
__global__ void gelu_kernel_fp32_vec(
    float* __restrict__ output,
    const float* __restrict__ input,
    const int vec_elements  // total_elements / 4
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_elements) {
        const float4* vec_in = reinterpret_cast<const float4*>(input);
        float4* vec_out = reinterpret_cast<float4*>(output);

        float4 v = vec_in[idx];
        float4 result;
        result.x = gelu_approx(v.x);
        result.y = gelu_approx(v.y);
        result.z = gelu_approx(v.z);
        result.w = gelu_approx(v.w);
        vec_out[idx] = result;
    }
}

// ============================================================================
// C++ entry points
// ============================================================================
extern "C" {

void gelu_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int total_elements,
    cudaStream_t stream
) {
    if (total_elements % 2 == 0) {
        int vec_elements = total_elements / 2;
        int num_blocks = (vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_kernel_bf16_vec<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, input, vec_elements
        );
    } else {
        int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_kernel_scalar<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, input, total_elements
        );
    }
}

void gelu_forward_fp16(
    __half* output,
    const __half* input,
    int total_elements,
    cudaStream_t stream
) {
    if (total_elements % 2 == 0) {
        int vec_elements = total_elements / 2;
        int num_blocks = (vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_kernel_fp16_vec<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, input, vec_elements
        );
    } else {
        int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_kernel_scalar<__half><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, input, total_elements
        );
    }
}

void gelu_forward_fp32(
    float* output,
    const float* input,
    int total_elements,
    cudaStream_t stream
) {
    if (total_elements % 4 == 0) {
        int vec_elements = total_elements / 4;
        int num_blocks = (vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_kernel_fp32_vec<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, input, vec_elements
        );
    } else {
        int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        gelu_kernel_scalar<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, input, total_elements
        );
    }
}

} // extern "C"

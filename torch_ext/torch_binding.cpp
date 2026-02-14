/*
 * PyTorch C++ bindings for GB10-optimized CUDA kernels
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// RMSNorm entry points
extern "C" {
void rmsnorm_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int, int, float, cudaStream_t);
void rmsnorm_forward_fp16(__half*, const __half*, const __half*, int, int, float, cudaStream_t);
void rmsnorm_forward_fp32(float*, const float*, const float*, int, int, float, cudaStream_t);
}

// GELU entry points
extern "C" {
void gelu_forward_bf16(__nv_bfloat16*, const __nv_bfloat16*, int, cudaStream_t);
void gelu_forward_fp16(__half*, const __half*, int, cudaStream_t);
void gelu_forward_fp32(float*, const float*, int, cudaStream_t);
}

// ============================================================================
// RMSNorm binding
// ============================================================================
void rmsnorm_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    const int hidden_size = input.size(-1);
    const int num_rows = input.numel() / hidden_size;

    TORCH_CHECK(weight.numel() == hidden_size,
        "weight size (", weight.numel(), ") must match hidden_size (", hidden_size, ")");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.scalar_type() == at::kBFloat16) {
        rmsnorm_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
            num_rows, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kHalf) {
        rmsnorm_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            num_rows, hidden_size, eps, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        rmsnorm_forward_fp32(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(input.data_ptr()),
            static_cast<const float*>(weight.data_ptr()),
            num_rows, hidden_size, eps, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", input.scalar_type());
    }
}

// ============================================================================
// GELU binding
// ============================================================================
void gelu_forward(
    torch::Tensor& output,
    const torch::Tensor& input
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const int total_elements = input.numel();

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (input.scalar_type() == at::kBFloat16) {
        gelu_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            total_elements, stream
        );
    } else if (input.scalar_type() == at::kHalf) {
        gelu_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(input.data_ptr()),
            total_elements, stream
        );
    } else if (input.scalar_type() == at::kFloat) {
        gelu_forward_fp32(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(input.data_ptr()),
            total_elements, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", input.scalar_type());
    }
}

// ============================================================================
// Module registration
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
    m.def("gelu_forward", &gelu_forward, "GELU approximate forward (CUDA)");
}

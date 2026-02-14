#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rmsnorm(Tensor! out, Tensor input, Tensor weight, float eps) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("rmsnorm", torch::kCUDA, &rmsnorm);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

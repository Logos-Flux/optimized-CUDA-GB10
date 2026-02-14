#pragma once

#include <torch/torch.h>

void rmsnorm(torch::Tensor& out,
             torch::Tensor const& input,
             torch::Tensor const& weight,
             double eps);

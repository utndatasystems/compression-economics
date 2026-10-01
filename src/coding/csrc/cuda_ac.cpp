#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> encode_intervals_cuda(
    torch::Tensor lows,
    torch::Tensor highs,
    torch::Tensor totals,
    torch::Tensor counts,
    int64_t nominal_total,
    int64_t state_bits,
    int64_t workspace_bytes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "encode_intervals",
      &encode_intervals_cuda,
      "Encode independent target-interval arithmetic streams (CUDA)");
}

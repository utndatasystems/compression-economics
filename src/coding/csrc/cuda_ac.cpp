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

std::vector<torch::Tensor> quantize_target_intervals_cuda(
    torch::Tensor probabilities,
    torch::Tensor probability_sums,
    torch::Tensor targets,
    int64_t nominal_total);

void decode_cdfs_cuda(
    torch::Tensor cdfs,
    torch::Tensor active,
    torch::Tensor payload,
    torch::Tensor bit_counts,
    torch::Tensor counts,
    torch::Tensor lows,
    torch::Tensor highs,
    torch::Tensor codes,
    torch::Tensor positions,
    torch::Tensor decoded,
    torch::Tensor symbols,
    torch::Tensor errors,
    int64_t nominal_total,
    int64_t state_bits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "encode_intervals",
      &encode_intervals_cuda,
      "Encode independent target-interval arithmetic streams (CUDA)");
  module.def(
      "quantize_target_intervals",
      &quantize_target_intervals_cuda,
      "Fuse floor-count quantization and target interval extraction (CUDA)");
  module.def(
      "decode_cdfs",
      &decode_cdfs_cuda,
      "Decode one symbol per active arithmetic stream from supplied CDFs (CUDA)");
}

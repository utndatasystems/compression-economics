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

void encode_intervals_step_cuda(
    torch::Tensor interval_lows,
    torch::Tensor interval_highs,
    torch::Tensor interval_totals,
    torch::Tensor active,
    torch::Tensor lows,
    torch::Tensor highs,
    torch::Tensor underflows,
    torch::Tensor workspace,
    torch::Tensor bit_counts,
    torch::Tensor byte_counts,
    torch::Tensor current_bytes,
    torch::Tensor bit_positions,
    torch::Tensor errors,
    int64_t nominal_total,
    int64_t state_bits);

void finish_intervals_cuda(
    torch::Tensor workspace,
    torch::Tensor bit_counts,
    torch::Tensor byte_counts,
    torch::Tensor current_bytes,
    torch::Tensor bit_positions,
    torch::Tensor errors);

std::vector<torch::Tensor> quantize_target_intervals_cuda(
    torch::Tensor probabilities,
    torch::Tensor probability_sums,
    torch::Tensor targets,
    int64_t nominal_total);

std::vector<torch::Tensor> quantize_target_intervals_compact_cuda(
    torch::Tensor probabilities,
    torch::Tensor probability_sums,
    torch::Tensor targets,
    int64_t nominal_total);

std::vector<torch::Tensor> quantize_target_intervals_exact_cuda(
    torch::Tensor probabilities,
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
      "encode_intervals_step",
      &encode_intervals_step_cuda,
      "Update persistent multistream arithmetic encoder state (CUDA v4)");
  module.def(
      "finish_intervals",
      &finish_intervals_cuda,
      "Finalize persistent multistream arithmetic encoder state (CUDA v4)");
  module.def(
      "quantize_target_intervals",
      &quantize_target_intervals_cuda,
      "Fuse floor-count quantization and target interval extraction (CUDA)");
  module.def(
      "quantize_target_intervals_compact",
      &quantize_target_intervals_compact_cuda,
      "Fuse source-dtype conversion, floor-count quantization, and target "
      "interval extraction (CUDA v3)");
  module.def(
      "quantize_target_intervals_exact",
      &quantize_target_intervals_exact_cuda,
      "Fuse normalization, exact fixed-total residual allocation, stable "
      "selection, and target interval extraction (CUDA v4)");
  module.def(
      "decode_cdfs",
      &decode_cdfs_cuda,
      "Decode one symbol per active arithmetic stream from supplied CDFs (CUDA)");
}

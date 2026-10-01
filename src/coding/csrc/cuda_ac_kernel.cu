#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

enum ErrorCode : int32_t {
  kSuccess = 0,
  kInvalidCount = 1,
  kInvalidInterval = 2,
  kWorkspaceOverflow = 3,
  kInvalidState = 4,
};

struct BitWriter {
  uint8_t* output;
  int64_t capacity;
  int64_t byte_count;
  int64_t bit_count;
  uint8_t current_byte;
  int32_t bit_position;
  int32_t error;

  __device__ void flush(uint8_t value) {
    if (byte_count < capacity) {
      output[byte_count] = value;
    } else if (error == kSuccess) {
      error = kWorkspaceOverflow;
    }
    ++byte_count;
  }

  __device__ void write(int64_t bit) {
    current_byte = static_cast<uint8_t>((current_byte << 1) | (bit & 1));
    ++bit_position;
    ++bit_count;
    if (bit_position == 8) {
      flush(current_byte);
      current_byte = 0;
      bit_position = 0;
    }
  }

  __device__ void finish() {
    write(1);
    if (bit_position != 0) {
      flush(static_cast<uint8_t>(current_byte << (8 - bit_position)));
    }
  }
};

__global__ void encode_intervals_kernel(
    const int64_t* __restrict__ lows,
    const int64_t* __restrict__ highs,
    const int64_t* __restrict__ totals,
    const int64_t* __restrict__ counts,
    int64_t steps,
    int64_t streams,
    int64_t nominal_total,
    int64_t state_bits,
    int64_t workspace_bytes,
    uint8_t* __restrict__ workspace,
    int64_t* __restrict__ bit_counts,
    int64_t* __restrict__ byte_counts,
    int32_t* __restrict__ errors) {
  const int64_t stream = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream >= streams) {
    return;
  }

  const int64_t count = counts[stream];
  if (count < 0 || count > steps) {
    errors[stream] = kInvalidCount;
    return;
  }

  const int64_t max_range = int64_t{1} << state_bits;
  const int64_t mask = max_range - 1;
  const int64_t top_mask = max_range >> 1;
  const int64_t second_mask = top_mask >> 1;
  int64_t low = 0;
  int64_t high = mask;
  int64_t underflow = 0;
  BitWriter writer{
      workspace + stream * workspace_bytes,
      workspace_bytes,
      0,
      0,
      0,
      0,
      kSuccess,
  };

  for (int64_t step = 0; step < count; ++step) {
    const int64_t offset = step * streams + stream;
    const int64_t interval_low = lows[offset];
    const int64_t interval_high = highs[offset];
    const int64_t symbol_total = totals[offset];
    if (interval_low < 0 || interval_low >= interval_high ||
        interval_high > symbol_total || symbol_total > nominal_total) {
      writer.error = kInvalidInterval;
      break;
    }

    const int64_t current_range = high - low + 1;
    const int64_t base_low = low;
    low = base_low + interval_low * current_range / symbol_total;
    high = base_low + interval_high * current_range / symbol_total - 1;
    if (low < 0 || high < low || high > mask) {
      writer.error = kInvalidState;
      break;
    }

    while (((low ^ high) & top_mask) == 0) {
      const int64_t bit = low >> (state_bits - 1);
      writer.write(bit);
      const int64_t inverse = bit ^ 1;
      for (int64_t pending = 0; pending < underflow; ++pending) {
        writer.write(inverse);
      }
      underflow = 0;
      low = (low << 1) & mask;
      high = ((high << 1) & mask) | 1;
    }

    while ((low & ~high & second_mask) != 0) {
      ++underflow;
      low = (low << 1) & (mask >> 1);
      high = ((high << 1) & (mask >> 1)) | top_mask | 1;
    }
  }

  if (writer.error == kSuccess || writer.error == kWorkspaceOverflow) {
    writer.finish();
  }
  bit_counts[stream] = writer.bit_count;
  byte_counts[stream] = writer.byte_count;
  errors[stream] = writer.error;
}

void check_input(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must have dtype int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> encode_intervals_cuda(
    torch::Tensor lows,
    torch::Tensor highs,
    torch::Tensor totals,
    torch::Tensor counts,
    int64_t nominal_total,
    int64_t state_bits,
    int64_t workspace_bytes) {
  check_input(lows, "lows");
  check_input(highs, "highs");
  check_input(totals, "totals");
  check_input(counts, "counts");
  TORCH_CHECK(lows.dim() == 2, "interval tensors must be two-dimensional");
  TORCH_CHECK(highs.sizes() == lows.sizes() && totals.sizes() == lows.sizes(),
              "low, high, and total tensors must have the same shape");
  TORCH_CHECK(counts.dim() == 1 && counts.size(0) == lows.size(1),
              "counts must contain one entry per stream");
  TORCH_CHECK(lows.device() == highs.device() && lows.device() == totals.device() &&
                  lows.device() == counts.device(),
              "all inputs must use the same CUDA device");
  TORCH_CHECK(state_bits >= 16 && state_bits <= 56,
              "state_bits must be between 16 and 56");
  TORCH_CHECK(nominal_total >= 1, "nominal_total must be positive");
  TORCH_CHECK(workspace_bytes >= 1, "workspace_bytes must be positive");

  const c10::cuda::CUDAGuard device_guard(lows.device());
  const int64_t streams = lows.size(1);
  auto byte_options = lows.options().dtype(torch::kUInt8);
  auto int_options = lows.options().dtype(torch::kInt64);
  auto error_options = lows.options().dtype(torch::kInt32);
  auto workspace = torch::zeros({streams, workspace_bytes}, byte_options);
  auto bit_counts = torch::zeros({streams}, int_options);
  auto byte_counts = torch::zeros({streams}, int_options);
  auto errors = torch::zeros({streams}, error_options);

  constexpr int threads = 128;
  const int blocks = static_cast<int>((streams + threads - 1) / threads);
  encode_intervals_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      lows.data_ptr<int64_t>(),
      highs.data_ptr<int64_t>(),
      totals.data_ptr<int64_t>(),
      counts.data_ptr<int64_t>(),
      lows.size(0),
      streams,
      nominal_total,
      state_bits,
      workspace_bytes,
      workspace.data_ptr<uint8_t>(),
      bit_counts.data_ptr<int64_t>(),
      byte_counts.data_ptr<int64_t>(),
      errors.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {workspace, bit_counts, byte_counts, errors};
}

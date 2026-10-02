#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
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

// The canonical MSAC v2 quantizer first widens probabilities to float64 and
// computes each row sum with PyTorch. This kernel deliberately receives both
// tensors: the ablation changes only frequency materialization/CDF extraction,
// not normalization semantics or reduction order.
__global__ void quantize_target_intervals_kernel(
    const double* __restrict__ probabilities,
    const double* __restrict__ probability_sums,
    const int64_t* __restrict__ targets,
    int64_t rows,
    int64_t columns,
    int64_t nominal_total,
    int64_t* __restrict__ lows,
    int64_t* __restrict__ highs,
    int64_t* __restrict__ totals,
    double* __restrict__ target_probabilities) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t target = targets[row];
  const double mass = probability_sums[row];
  if (target < 0 || target >= columns || !isfinite(mass) || mass <= 0.0) {
    if (threadIdx.x == 0) {
      lows[row] = 0;
      highs[row] = 0;
      totals[row] = 0;
      target_probabilities[row] = 0.0;
    }
    return;
  }

  int64_t local_low = 0;
  int64_t local_total = 0;
  int64_t local_target_frequency = 0;
  const double scale = static_cast<double>(nominal_total - columns);
  const int64_t row_offset = row * columns;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    // Explicit round-to-nearest operations match the two separate PyTorch
    // elementwise kernels used by target_intervals_from_probs_tensor.
    const double normalized = __ddiv_rn(probabilities[row_offset + column], mass);
    const double scaled = __dmul_rn(normalized, scale);
    const int64_t frequency = static_cast<int64_t>(floor(scaled)) + 1;
    local_total += frequency;
    if (column < target) {
      local_low += frequency;
    } else if (column == target) {
      local_target_frequency = frequency;
    }
  }

  __shared__ int64_t total_reduction[256];
  __shared__ int64_t low_reduction[256];
  __shared__ int64_t target_reduction[256];
  total_reduction[threadIdx.x] = local_total;
  low_reduction[threadIdx.x] = local_low;
  target_reduction[threadIdx.x] = local_target_frequency;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      total_reduction[threadIdx.x] += total_reduction[threadIdx.x + stride];
      low_reduction[threadIdx.x] += low_reduction[threadIdx.x + stride];
      target_reduction[threadIdx.x] += target_reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int64_t low = low_reduction[0];
    lows[row] = low;
    highs[row] = low + target_reduction[0];
    totals[row] = total_reduction[0];
    target_probabilities[row] =
        __ddiv_rn(probabilities[row_offset + target], mass);
  }
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

std::vector<torch::Tensor> quantize_target_intervals_cuda(
    torch::Tensor probabilities,
    torch::Tensor probability_sums,
    torch::Tensor targets,
    int64_t nominal_total) {
  TORCH_CHECK(probabilities.is_cuda(), "probabilities must be a CUDA tensor");
  TORCH_CHECK(probabilities.scalar_type() == torch::kFloat64,
              "probabilities must have dtype float64");
  TORCH_CHECK(probabilities.is_contiguous(), "probabilities must be contiguous");
  TORCH_CHECK(probabilities.dim() == 2, "probabilities must be two-dimensional");
  TORCH_CHECK(probability_sums.is_cuda() &&
                  probability_sums.scalar_type() == torch::kFloat64 &&
                  probability_sums.is_contiguous(),
              "probability_sums must be contiguous CUDA float64");
  TORCH_CHECK(probability_sums.dim() == 1 &&
                  probability_sums.size(0) == probabilities.size(0),
              "probability_sums must contain one value per row");
  check_input(targets, "targets");
  TORCH_CHECK(targets.dim() == 1 && targets.size(0) == probabilities.size(0),
              "targets must contain one index per row");
  TORCH_CHECK(probabilities.device() == probability_sums.device() &&
                  probabilities.device() == targets.device(),
              "all quantizer inputs must use the same CUDA device");
  TORCH_CHECK(probabilities.size(1) >= 1 &&
                  probabilities.size(1) < nominal_total,
              "alphabet must be nonempty and smaller than nominal_total");

  const c10::cuda::CUDAGuard device_guard(probabilities.device());
  const int64_t rows = probabilities.size(0);
  auto int_options = probabilities.options().dtype(torch::kInt64);
  auto lows = torch::empty({rows}, int_options);
  auto highs = torch::empty({rows}, int_options);
  auto totals = torch::empty({rows}, int_options);
  auto target_probabilities = torch::empty({rows}, probabilities.options());
  constexpr int threads = 256;
  quantize_target_intervals_kernel<<<
      static_cast<int>(rows), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      probabilities.data_ptr<double>(),
      probability_sums.data_ptr<double>(),
      targets.data_ptr<int64_t>(),
      rows,
      probabilities.size(1),
      nominal_total,
      lows.data_ptr<int64_t>(),
      highs.data_ptr<int64_t>(),
      totals.data_ptr<int64_t>(),
      target_probabilities.data_ptr<double>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {lows, highs, totals, target_probabilities};
}

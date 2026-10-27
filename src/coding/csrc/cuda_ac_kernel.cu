#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cub/block/block_radix_sort.cuh>

#include <cmath>
#include <climits>
#include <cstdint>
#include <vector>

namespace {

enum ErrorCode : int32_t {
  kSuccess = 0,
  kInvalidCount = 1,
  kInvalidInterval = 2,
  kWorkspaceOverflow = 3,
  kInvalidState = 4,
  kDecodeExhausted = 5,
  kInvalidCdf = 6,
  kInvalidSymbol = 7,
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

// IEEE -0.0 compares equal to +0.0 and the fixed-total contract uses a stable
// column-order tie break. Raw bits would rank -0.0 above positive finite values.
__device__ __forceinline__ uint32_t nonnegative_float_key(float value) {
  return value == 0.0f ? 0u : __float_as_uint(value);
}

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

__global__ void encode_intervals_step_kernel(
    const int64_t* __restrict__ interval_lows,
    const int64_t* __restrict__ interval_highs,
    const int64_t* __restrict__ interval_totals,
    const bool* __restrict__ active,
    int64_t streams,
    int64_t nominal_total,
    int64_t state_bits,
    int64_t workspace_bytes,
    int64_t* __restrict__ lows,
    int64_t* __restrict__ highs,
    int64_t* __restrict__ underflows,
    uint8_t* __restrict__ workspace,
    int64_t* __restrict__ bit_counts,
    int64_t* __restrict__ byte_counts,
    uint8_t* __restrict__ current_bytes,
    int32_t* __restrict__ bit_positions,
    int32_t* __restrict__ errors) {
  const int64_t stream = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream >= streams || !active[stream] || errors[stream] != kSuccess) {
    return;
  }
  const int64_t interval_low = interval_lows[stream];
  const int64_t interval_high = interval_highs[stream];
  const int64_t symbol_total = interval_totals[stream];
  if (interval_low < 0 || interval_low >= interval_high ||
      interval_high > symbol_total || symbol_total > nominal_total) {
    errors[stream] = kInvalidInterval;
    return;
  }

  int64_t low = lows[stream];
  int64_t high = highs[stream];
  int64_t underflow = underflows[stream];
  BitWriter writer{
      workspace + stream * workspace_bytes,
      workspace_bytes,
      byte_counts[stream],
      bit_counts[stream],
      current_bytes[stream],
      bit_positions[stream],
      errors[stream],
  };
  const int64_t current_range = high - low + 1;
  const int64_t base_low = low;
  low = base_low + interval_low * current_range / symbol_total;
  high = base_low + interval_high * current_range / symbol_total - 1;
  const int64_t max_range = int64_t{1} << state_bits;
  const int64_t mask = max_range - 1;
  const int64_t top_mask = max_range >> 1;
  const int64_t second_mask = top_mask >> 1;
  if (low < 0 || high < low || high > mask) {
    writer.error = kInvalidState;
  } else {
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

  lows[stream] = low;
  highs[stream] = high;
  underflows[stream] = underflow;
  bit_counts[stream] = writer.bit_count;
  byte_counts[stream] = writer.byte_count;
  current_bytes[stream] = writer.current_byte;
  bit_positions[stream] = writer.bit_position;
  errors[stream] = writer.error;
}

__global__ void finish_intervals_kernel(
    int64_t streams,
    int64_t workspace_bytes,
    uint8_t* __restrict__ workspace,
    int64_t* __restrict__ bit_counts,
    int64_t* __restrict__ byte_counts,
    uint8_t* __restrict__ current_bytes,
    int32_t* __restrict__ bit_positions,
    int32_t* __restrict__ errors) {
  const int64_t stream = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream >= streams) {
    return;
  }
  BitWriter writer{
      workspace + stream * workspace_bytes,
      workspace_bytes,
      byte_counts[stream],
      bit_counts[stream],
      current_bytes[stream],
      bit_positions[stream],
      errors[stream],
  };
  if (writer.error == kSuccess || writer.error == kWorkspaceOverflow) {
    writer.finish();
  }
  bit_counts[stream] = writer.bit_count;
  byte_counts[stream] = writer.byte_count;
  current_bytes[stream] = writer.current_byte;
  bit_positions[stream] = writer.bit_position;
  errors[stream] = writer.error;
}

// The canonical MSAC v2 quantizer first widens probabilities to float64 and
// computes each row sum with PyTorch. This kernel deliberately receives both
// tensors: the ablation changes only frequency materialization/CDF extraction,
// not normalization semantics or reduction order.
template <typename scalar_t>
__global__ void quantize_target_intervals_kernel(
    const scalar_t* __restrict__ probabilities,
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
    const double probability =
        static_cast<double>(probabilities[row_offset + column]);
    const double normalized = __ddiv_rn(probability, mass);
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
        __ddiv_rn(
            static_cast<double>(probabilities[row_offset + target]), mass);
  }
}

// CUDA v4 performs the complete fixed-total target-interval operation in one
// block per probability row.  It deliberately emits no vocabulary-sized
// frequency or CDF tensor.  Stable descending selection is implemented by a
// four-pass radix selection of the residual cutoff; equal values are awarded
// residual counts in increasing column order.
template <typename scalar_t>
__global__ void quantize_target_intervals_exact_kernel(
    const scalar_t* __restrict__ probabilities,
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
  const int64_t row_offset = row * columns;
  const int64_t target = targets[row];

  __shared__ double mass;
  __shared__ int32_t valid;
  __shared__ int64_t total_reduction[256];
  __shared__ int64_t low_reduction[256];
  __shared__ int64_t target_reduction[256];
  __shared__ uint32_t histogram[256];
  __shared__ uint32_t radix_prefix;
  __shared__ uint32_t cutoff_key;
  __shared__ int64_t cutoff_column;
  __shared__ int64_t residual;
  __shared__ int64_t cutoff_tie_count;
  using SmallRowSort = cub::BlockRadixSort<uint64_t, 256, 4>;
  __shared__ typename SmallRowSort::TempStorage small_sort_storage;

  if (threadIdx.x == 0) {
    // A canonical column-order float64 accumulation makes normalization
    // independent of block scheduling and source probability dtype.
    double row_mass = 0.0;
    int32_t row_valid = target >= 0 && target < columns;
    for (int64_t column = 0; column < columns; ++column) {
      const float value = static_cast<float>(probabilities[row_offset + column]);
      row_valid = row_valid && isfinite(value) && value >= 0.0f;
      row_mass += static_cast<double>(value);
    }
    mass = row_mass;
    valid = row_valid && isfinite(row_mass) && row_mass > 0.0;
  }
  __syncthreads();

  if (!valid) {
    if (threadIdx.x == 0) {
      lows[row] = 0;
      highs[row] = 0;
      totals[row] = 0;
      target_probabilities[row] = 0.0;
    }
    return;
  }

  int64_t local_total = 0;
  int64_t local_low = 0;
  int64_t local_target = 0;
  const double scale = static_cast<double>(nominal_total - columns);
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    const float value = static_cast<float>(probabilities[row_offset + column]);
    const int64_t frequency =
        static_cast<int64_t>(floor(static_cast<double>(value) / mass * scale)) + 1;
    local_total += frequency;
    if (column < target) {
      local_low += frequency;
    } else if (column == target) {
      local_target = frequency;
    }
  }
  total_reduction[threadIdx.x] = local_total;
  low_reduction[threadIdx.x] = local_low;
  target_reduction[threadIdx.x] = local_target;
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
    residual = nominal_total - total_reduction[0];
    radix_prefix = 0;
    cutoff_key = 0;
    cutoff_column = -1;
    cutoff_tie_count = 0;
    valid = residual >= 0 && residual <= columns;
  }
  __syncthreads();
  if (!valid) {
    if (threadIdx.x == 0) {
      lows[row] = 0;
      highs[row] = 0;
      totals[row] = 0;
      target_probabilities[row] = 0.0;
    }
    return;
  }

  int64_t local_low_bonus = 0;
  int64_t local_target_bonus = 0;
  if (residual > 0 && columns <= 1024) {
    uint64_t keys[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int64_t column = static_cast<int64_t>(threadIdx.x) * 4 + item;
      if (column < columns) {
        const float value = static_cast<float>(probabilities[row_offset + column]);
        keys[item] = (static_cast<uint64_t>(nonnegative_float_key(value)) << 32) |
            static_cast<uint32_t>(0xffffffffu - static_cast<uint32_t>(column));
      } else {
        keys[item] = 0;
      }
    }
    SmallRowSort(small_sort_storage).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int64_t rank = static_cast<int64_t>(threadIdx.x) * 4 + item;
      if (rank < residual) {
        const int64_t column = static_cast<int64_t>(
            0xffffffffu - static_cast<uint32_t>(keys[item]));
        if (column < target) {
          ++local_low_bonus;
        } else if (column == target) {
          local_target_bonus = 1;
        }
      }
    }
  } else if (residual > 0) {
    // Select the residual-th largest nonnegative float key without producing a
    // sorted row.  The residual is strictly smaller than the alphabet for a
    // normalized floor allocation, so only a partial final tie is possible.
    if (threadIdx.x == 0) {
      cutoff_tie_count = residual - 1;  // zero-based rank within current prefix
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
      histogram[threadIdx.x] = 0;
      __syncthreads();
      const uint32_t prefix_mask =
          shift == 24 ? 0u : ~((uint32_t{1} << (shift + 8)) - 1u);
      const uint32_t prefix = radix_prefix;
      for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
        const float value = static_cast<float>(probabilities[row_offset + column]);
        const uint32_t key = nonnegative_float_key(value);
        if ((key & prefix_mask) == prefix) {
          atomicAdd(&histogram[(key >> shift) & 0xffu], 1u);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        int64_t rank = cutoff_tie_count;
        uint32_t bucket = 255;
        for (;; --bucket) {
          const int64_t count = histogram[bucket];
          if (rank < count) {
            break;
          }
          rank -= count;
        }
        radix_prefix |= bucket << shift;
        cutoff_tie_count = rank;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      cutoff_key = radix_prefix;
      ++cutoff_tie_count;  // number of equal keys awarded in stable order
      int64_t seen = 0;
      for (int64_t column = 0; column < columns; ++column) {
        const float value = static_cast<float>(probabilities[row_offset + column]);
        if (nonnegative_float_key(value) == cutoff_key &&
            ++seen == cutoff_tie_count) {
          cutoff_column = column;
          break;
        }
      }
    }
    __syncthreads();
  }

  if (residual > 0 && columns > 1024) {
    for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
      const float value = static_cast<float>(probabilities[row_offset + column]);
      const uint32_t key = nonnegative_float_key(value);
      const bool selected = key > cutoff_key ||
          (key == cutoff_key && column <= cutoff_column);
      if (selected && column < target) {
        ++local_low_bonus;
      } else if (selected && column == target) {
        local_target_bonus = 1;
      }
    }
  }
  low_reduction[threadIdx.x] = local_low_bonus;
  target_reduction[threadIdx.x] = local_target_bonus;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      low_reduction[threadIdx.x] += low_reduction[threadIdx.x + stride];
      target_reduction[threadIdx.x] += target_reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // Reconstruct the base prefix because the reduction scratch was reused.
    int64_t base_low = 0;
    for (int64_t column = 0; column < target; ++column) {
      const float value = static_cast<float>(probabilities[row_offset + column]);
      base_low += static_cast<int64_t>(
          floor(static_cast<double>(value) / mass * scale)) + 1;
    }
    const float target_value =
        static_cast<float>(probabilities[row_offset + target]);
    const int64_t target_frequency = static_cast<int64_t>(
        floor(static_cast<double>(target_value) / mass * scale)) + 1;
    const int64_t final_low = base_low + low_reduction[0];
    lows[row] = final_low;
    highs[row] = final_low + target_frequency + target_reduction[0];
    totals[row] = nominal_total;
    target_probabilities[row] = static_cast<double>(target_value) / mass;
  }
}

__device__ int64_t read_payload_bit(
    const int64_t* __restrict__ payload,
    int64_t payload_width,
    int64_t stream,
    int64_t bit_count,
    int64_t* position) {
  const int64_t current = *position;
  ++(*position);
  if (current >= bit_count) {
    return 0;
  }
  const int64_t byte = payload[stream * payload_width + current / 8];
  return (byte >> (7 - current % 8)) & 1;
}

__global__ void decode_cdfs_kernel(
    const int64_t* __restrict__ cdfs,
    const bool* __restrict__ active,
    const int64_t* __restrict__ payload,
    const int64_t* __restrict__ bit_counts,
    const int64_t* __restrict__ counts,
    int64_t streams,
    int64_t cdf_columns,
    int64_t payload_width,
    int64_t nominal_total,
    int64_t state_bits,
    int64_t* __restrict__ lows,
    int64_t* __restrict__ highs,
    int64_t* __restrict__ codes,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ decoded,
    int64_t* __restrict__ symbols,
    int32_t* __restrict__ errors) {
  const int64_t stream = blockIdx.x * blockDim.x + threadIdx.x;
  if (stream >= streams || !active[stream] || errors[stream] != kSuccess) {
    return;
  }
  if (decoded[stream] >= counts[stream]) {
    errors[stream] = kDecodeExhausted;
    return;
  }

  const int64_t* cdf = cdfs + stream * cdf_columns;
  const int64_t alphabet = cdf_columns - 1;
  const int64_t total = cdf[alphabet];
  if (cdf[0] != 0 || total <= 0 || total > nominal_total) {
    errors[stream] = kInvalidCdf;
    return;
  }
  int64_t low = lows[stream];
  int64_t high = highs[stream];
  int64_t code = codes[stream];
  int64_t position = positions[stream];
  const int64_t current_range = high - low + 1;
  if (current_range <= 0 || code < low || code > high) {
    errors[stream] = kInvalidState;
    return;
  }
  const int64_t value = ((code - low + 1) * total - 1) / current_range;

  // upper_bound(cdf, value) - 1. Every positive floor-count frequency
  // makes the CDF strictly increasing for valid archives.
  int64_t left = 0;
  int64_t right = cdf_columns;
  while (left < right) {
    const int64_t middle = left + (right - left) / 2;
    if (cdf[middle] <= value) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }
  const int64_t symbol = left - 1;
  if (symbol < 0 || symbol >= alphabet) {
    errors[stream] = kInvalidSymbol;
    return;
  }
  const int64_t interval_low = cdf[symbol];
  const int64_t interval_high = cdf[symbol + 1];
  if (interval_low < 0 || interval_low >= interval_high ||
      interval_high > total) {
    errors[stream] = kInvalidCdf;
    return;
  }

  const int64_t base_low = low;
  low = base_low + interval_low * current_range / total;
  high = base_low + interval_high * current_range / total - 1;
  const int64_t max_range = int64_t{1} << state_bits;
  const int64_t mask = max_range - 1;
  const int64_t top_mask = max_range >> 1;
  const int64_t second_mask = top_mask >> 1;
  if (low < 0 || high < low || high > mask) {
    errors[stream] = kInvalidState;
    return;
  }

  while (((low ^ high) & top_mask) == 0) {
    const int64_t bit = read_payload_bit(
        payload, payload_width, stream, bit_counts[stream], &position);
    code = ((code << 1) & mask) | bit;
    low = (low << 1) & mask;
    high = ((high << 1) & mask) | 1;
  }
  while ((low & ~high & second_mask) != 0) {
    const int64_t bit = read_payload_bit(
        payload, payload_width, stream, bit_counts[stream], &position);
    code = (code & top_mask) | ((code << 1) & (mask >> 1)) | bit;
    low = (low << 1) & (mask >> 1);
    high = ((high << 1) & (mask >> 1)) | top_mask | 1;
  }

  lows[stream] = low;
  highs[stream] = high;
  codes[stream] = code;
  positions[stream] = position;
  ++decoded[stream];
  symbols[stream] = symbol;
}

void check_input(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must have dtype int64");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_coder_settings(int64_t nominal_total, int64_t state_bits) {
  TORCH_CHECK(state_bits >= 16 && state_bits <= 56,
              "state_bits must be between 16 and 56");
  TORCH_CHECK(nominal_total >= 1, "nominal_total must be positive");
  const int64_t max_range = int64_t{1} << state_bits;
  TORCH_CHECK(nominal_total <= (max_range >> 2) + 2,
              "nominal_total is outside the arithmetic coder's range");
  TORCH_CHECK(nominal_total <= INT64_MAX / max_range,
              "arithmetic interval product exceeds signed 64-bit arithmetic");
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
  TORCH_CHECK(lows.size(1) > 0,
              "interval tensors must contain at least one stream");
  TORCH_CHECK(lows.device() == highs.device() && lows.device() == totals.device() &&
                  lows.device() == counts.device(),
              "all inputs must use the same CUDA device");
  check_coder_settings(nominal_total, state_bits);
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
    int64_t state_bits) {
  check_input(interval_lows, "interval_lows");
  check_input(interval_highs, "interval_highs");
  check_input(interval_totals, "interval_totals");
  check_input(lows, "lows");
  check_input(highs, "highs");
  check_input(underflows, "underflows");
  check_input(bit_counts, "bit_counts");
  check_input(byte_counts, "byte_counts");
  TORCH_CHECK(active.is_cuda() && active.scalar_type() == torch::kBool &&
                  active.is_contiguous(), "active must be contiguous CUDA bool");
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kUInt8 &&
                  workspace.is_contiguous() && workspace.dim() == 2,
              "workspace must be contiguous 2D CUDA uint8");
  TORCH_CHECK(current_bytes.is_cuda() &&
                  current_bytes.scalar_type() == torch::kUInt8 &&
                  current_bytes.is_contiguous(),
              "current_bytes must be contiguous CUDA uint8");
  TORCH_CHECK(bit_positions.is_cuda() &&
                  bit_positions.scalar_type() == torch::kInt32 &&
                  bit_positions.is_contiguous(),
              "bit_positions must be contiguous CUDA int32");
  TORCH_CHECK(errors.is_cuda() && errors.scalar_type() == torch::kInt32 &&
                  errors.is_contiguous(), "errors must be contiguous CUDA int32");
  const int64_t streams = interval_lows.numel();
  TORCH_CHECK(streams > 0 && interval_lows.dim() == 1,
              "interval inputs must be nonempty vectors");
  for (const auto& tensor : {interval_highs, interval_totals, lows, highs,
                             underflows, bit_counts, byte_counts}) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == streams,
                "persistent encoder vectors must match stream count");
    TORCH_CHECK(tensor.device() == interval_lows.device(),
                "persistent encoder tensors must share one CUDA device");
  }
  for (const auto& tensor : {active, current_bytes, bit_positions, errors}) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == streams &&
                    tensor.device() == interval_lows.device(),
                "persistent encoder state must match stream count and device");
  }
  TORCH_CHECK(workspace.size(0) == streams &&
                  workspace.size(1) > 0 &&
                  workspace.device() == interval_lows.device(),
              "workspace must contain one row per stream on the same device");
  check_coder_settings(nominal_total, state_bits);

  const c10::cuda::CUDAGuard device_guard(interval_lows.device());
  constexpr int threads = 128;
  const int blocks = static_cast<int>((streams + threads - 1) / threads);
  encode_intervals_step_kernel<<<
      blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      interval_lows.data_ptr<int64_t>(), interval_highs.data_ptr<int64_t>(),
      interval_totals.data_ptr<int64_t>(), active.data_ptr<bool>(), streams,
      nominal_total, state_bits, workspace.size(1), lows.data_ptr<int64_t>(),
      highs.data_ptr<int64_t>(), underflows.data_ptr<int64_t>(),
      workspace.data_ptr<uint8_t>(), bit_counts.data_ptr<int64_t>(),
      byte_counts.data_ptr<int64_t>(), current_bytes.data_ptr<uint8_t>(),
      bit_positions.data_ptr<int32_t>(), errors.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finish_intervals_cuda(
    torch::Tensor workspace,
    torch::Tensor bit_counts,
    torch::Tensor byte_counts,
    torch::Tensor current_bytes,
    torch::Tensor bit_positions,
    torch::Tensor errors) {
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type() == torch::kUInt8 &&
                  workspace.is_contiguous() && workspace.dim() == 2,
              "workspace must be contiguous 2D CUDA uint8");
  const int64_t streams = workspace.size(0);
  check_input(bit_counts, "bit_counts");
  check_input(byte_counts, "byte_counts");
  TORCH_CHECK(current_bytes.is_cuda() &&
                  current_bytes.scalar_type() == torch::kUInt8 &&
                  current_bytes.is_contiguous(),
              "current_bytes must be contiguous CUDA uint8");
  TORCH_CHECK(bit_positions.is_cuda() &&
                  bit_positions.scalar_type() == torch::kInt32 &&
                  bit_positions.is_contiguous(),
              "bit_positions must be contiguous CUDA int32");
  TORCH_CHECK(errors.is_cuda() && errors.scalar_type() == torch::kInt32 &&
                  errors.is_contiguous(), "errors must be contiguous CUDA int32");
  for (const auto& tensor : {bit_counts, byte_counts, current_bytes,
                             bit_positions, errors}) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == streams &&
                    tensor.device() == workspace.device(),
                "final encoder state must match workspace streams and device");
  }
  const c10::cuda::CUDAGuard device_guard(workspace.device());
  constexpr int threads = 128;
  const int blocks = static_cast<int>((streams + threads - 1) / threads);
  finish_intervals_kernel<<<
      blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      streams, workspace.size(1), workspace.data_ptr<uint8_t>(),
      bit_counts.data_ptr<int64_t>(), byte_counts.data_ptr<int64_t>(),
      current_bytes.data_ptr<uint8_t>(), bit_positions.data_ptr<int32_t>(),
      errors.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  if (rows == 0) {
    return {lows, highs, totals, target_probabilities};
  }
  TORCH_CHECK(rows <= INT_MAX, "too many probability rows for a CUDA grid");
  constexpr int threads = 256;
  quantize_target_intervals_kernel<double><<<
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

std::vector<torch::Tensor> quantize_target_intervals_compact_cuda(
    torch::Tensor probabilities,
    torch::Tensor probability_sums,
    torch::Tensor targets,
    int64_t nominal_total) {
  TORCH_CHECK(probabilities.is_cuda(), "probabilities must be a CUDA tensor");
  TORCH_CHECK(probabilities.is_floating_point(),
              "probabilities must have a floating-point dtype");
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
  auto double_options = probabilities.options().dtype(torch::kFloat64);
  auto lows = torch::empty({rows}, int_options);
  auto highs = torch::empty({rows}, int_options);
  auto totals = torch::empty({rows}, int_options);
  auto target_probabilities = torch::empty({rows}, double_options);
  if (rows == 0) {
    return {lows, highs, totals, target_probabilities};
  }
  TORCH_CHECK(rows <= INT_MAX, "too many probability rows for a CUDA grid");
  constexpr int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      probabilities.scalar_type(),
      "quantize_target_intervals_compact_cuda",
      [&] {
        quantize_target_intervals_kernel<scalar_t><<<
            static_cast<int>(rows),
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            probabilities.data_ptr<scalar_t>(),
            probability_sums.data_ptr<double>(),
            targets.data_ptr<int64_t>(),
            rows,
            probabilities.size(1),
            nominal_total,
            lows.data_ptr<int64_t>(),
            highs.data_ptr<int64_t>(),
            totals.data_ptr<int64_t>(),
            target_probabilities.data_ptr<double>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {lows, highs, totals, target_probabilities};
}

std::vector<torch::Tensor> quantize_target_intervals_exact_cuda(
    torch::Tensor probabilities,
    torch::Tensor targets,
    int64_t nominal_total) {
  TORCH_CHECK(probabilities.is_cuda(), "probabilities must be a CUDA tensor");
  TORCH_CHECK(probabilities.is_floating_point(),
              "probabilities must have a floating-point dtype");
  TORCH_CHECK(probabilities.is_contiguous(), "probabilities must be contiguous");
  TORCH_CHECK(probabilities.dim() == 2, "probabilities must be two-dimensional");
  check_input(targets, "targets");
  TORCH_CHECK(targets.dim() == 1 && targets.size(0) == probabilities.size(0),
              "targets must contain one index per row");
  TORCH_CHECK(probabilities.device() == targets.device(),
              "all quantizer inputs must use the same CUDA device");
  TORCH_CHECK(probabilities.size(1) > 1 &&
                  probabilities.size(1) < nominal_total,
              "alphabet must contain at least two entries and be smaller than nominal_total");

  const c10::cuda::CUDAGuard device_guard(probabilities.device());
  const int64_t rows = probabilities.size(0);
  auto int_options = probabilities.options().dtype(torch::kInt64);
  auto double_options = probabilities.options().dtype(torch::kFloat64);
  auto lows = torch::empty({rows}, int_options);
  auto highs = torch::empty({rows}, int_options);
  auto totals = torch::empty({rows}, int_options);
  auto target_probabilities = torch::empty({rows}, double_options);
  if (rows == 0) {
    return {lows, highs, totals, target_probabilities};
  }
  TORCH_CHECK(rows <= INT_MAX, "too many probability rows for a CUDA grid");
  constexpr int threads = 256;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf,
      torch::kBFloat16,
      probabilities.scalar_type(),
      "quantize_target_intervals_exact_cuda",
      [&] {
        quantize_target_intervals_exact_kernel<scalar_t><<<
            static_cast<int>(rows),
            threads,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            probabilities.data_ptr<scalar_t>(),
            targets.data_ptr<int64_t>(),
            rows,
            probabilities.size(1),
            nominal_total,
            lows.data_ptr<int64_t>(),
            highs.data_ptr<int64_t>(),
            totals.data_ptr<int64_t>(),
            target_probabilities.data_ptr<double>());
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {lows, highs, totals, target_probabilities};
}

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
    int64_t state_bits) {
  check_input(cdfs, "cdfs");
  check_input(payload, "payload");
  check_input(bit_counts, "bit_counts");
  check_input(counts, "counts");
  check_input(lows, "lows");
  check_input(highs, "highs");
  check_input(codes, "codes");
  check_input(positions, "positions");
  check_input(decoded, "decoded");
  check_input(symbols, "symbols");
  TORCH_CHECK(active.is_cuda() && active.scalar_type() == torch::kBool &&
                  active.is_contiguous(),
              "active must be contiguous CUDA bool");
  TORCH_CHECK(errors.is_cuda() && errors.scalar_type() == torch::kInt32 &&
                  errors.is_contiguous(),
              "errors must be contiguous CUDA int32");
  TORCH_CHECK(cdfs.dim() == 2 && cdfs.size(1) >= 3,
              "cdfs must have shape [streams, alphabet + 1]");
  const int64_t streams = cdfs.size(0);
  TORCH_CHECK(streams > 0,
              "decoder tensors must contain at least one stream");
  TORCH_CHECK(payload.dim() == 2 && payload.size(0) == streams,
              "payload must contain one row per stream");
  for (const auto& tensor : {active, bit_counts, counts, lows, highs, codes,
                             positions, decoded, symbols, errors}) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == streams,
                "decoder state tensors must contain one value per stream");
    TORCH_CHECK(tensor.device() == cdfs.device(),
                "all decoder tensors must use the same CUDA device");
  }
  check_coder_settings(nominal_total, state_bits);
  TORCH_CHECK(nominal_total > cdfs.size(1) - 1,
              "nominal_total must exceed the alphabet size");

  const c10::cuda::CUDAGuard device_guard(cdfs.device());
  constexpr int threads = 128;
  const int blocks = static_cast<int>((streams + threads - 1) / threads);
  decode_cdfs_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      cdfs.data_ptr<int64_t>(),
      active.data_ptr<bool>(),
      payload.data_ptr<int64_t>(),
      bit_counts.data_ptr<int64_t>(),
      counts.data_ptr<int64_t>(),
      streams,
      cdfs.size(1),
      payload.size(1),
      nominal_total,
      state_bits,
      lows.data_ptr<int64_t>(),
      highs.data_ptr<int64_t>(),
      codes.data_ptr<int64_t>(),
      positions.data_ptr<int64_t>(),
      decoded.data_ptr<int64_t>(),
      symbols.data_ptr<int64_t>(),
      errors.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

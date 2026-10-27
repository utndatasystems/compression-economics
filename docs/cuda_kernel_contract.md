# CUDA arithmetic-coding kernel contract

This document records the supported inputs of `src/coding/cuda_ac.py` and its
extension in `src/coding/csrc/cuda_ac_kernel.cu`. It is a contract boundary,
not a claim that arbitrary tensors accepted by the private pybind module are
valid.

## Build and execution

The extension is JIT-built with host and device optimization (`-O3`) and CUDA
device line information (`-lineinfo`). Every launch uses the guarded input
device and PyTorch's current stream. Launch configuration errors are checked
immediately. Public archive-producing operations synchronize before consuming
device results; functions whose names end in `_raw`, and the quantizers, remain
asynchronous until the caller performs a normal PyTorch dependency or stream
synchronization.

All tensors used by one kernel invocation must be on one NVIDIA CUDA device.
PyTorch owns their allocations and supplies naturally aligned pointers for each
declared scalar dtype; the kernels make scalar accesses and require no stronger
alignment. Inputs must remain alive and unmodified until their consuming stream
has completed. Outputs are fresh allocations except for the explicitly
persistent encoder/decoder state. Input/output aliasing and overlap between
mutable state tensors are unsupported. The high-level classes allocate
non-overlapping state.

## Buffered interval encoder

`encode_intervals_cuda` accepts contiguous CUDA `int64` `lows`, `highs`, and
`totals` with shape `[steps, streams]`, plus contiguous CUDA `int64` `counts`
with shape `[streams]`, all on one device. `steps` may be zero. `streams` is in
`[1, 2**32 - 1]`, subject to allocation and CUDA grid limits. Each
`counts[s]` is in `[0, steps]`; only that prefix of column `s` is read. Each
read interval satisfies `0 <= low < high <= total <= nominal_total`.

`state_bits` is 16 through 56. `nominal_total` must satisfy the arithmetic
coder range rule and `2**state_bits * nominal_total <= INT64_MAX`.
`workspace_bytes` is positive. An undersized workspace is supported: no write
is made outside it, the required byte count is returned, and the high-level
wrapper retries once at that exact size. Every stream, including a zero-symbol
stream, receives the terminating bit and therefore a nonempty payload.

One CUDA thread owns one stream and all of its output bytes. No global-memory
location is intentionally shared between threads or blocks.

## Persistent encoder

`CudaPipelinedACEncoder` has the same stream-count, arithmetic-range, and
interval-value rules. `max_steps` is nonnegative and determines a conservative
workspace capacity. Each `encode_step` input is a contiguous one-dimensional
CUDA tensor of length `streams`: `int64` for low/high/total and `bool` for the
active mask. State tensors are private, contiguous, and have dtype `int64`,
`uint8`, or `int32` as declared in the source. The caller must not mutate a
submitted input before the encoder's dedicated stream has consumed it.

An event makes every step wait for its producer stream. All updates and final
output execute serially on one dedicated stream; `finish` synchronizes its end
event before host access. A lane is updated by exactly one thread, so there is
no cross-block dependency or global-memory race under the no-alias contract.

## Target-interval quantizers

All variants accept a two-dimensional CUDA probability tensor with shape
`[rows, alphabet]`; `rows` may be zero. V2 and v3 require
`1 <= alphabet < total`; v4 requires `2 <= alphabet < total`. Targets contain
one integer column per row. Non-contiguous probability and target inputs are
supported by making contiguous tensors before launch.

- V2 widens the source to contiguous `float64` before the kernel.
- V3 supports `float16`, `bfloat16`, `float32`, and `float64`, preserves that
  contiguous source dtype, and supplies a separately reduced `float64` mass.
- V4 supports those same floating dtypes and defines its archive semantics by
  first converting each value to `float32`, then accumulating mass in
  left-to-right column order as `float64`. Values must remain finite through
  that conversion.

With validation enabled (the default), values are finite and nonnegative, each
row has positive mass, and targets are in range. NaNs, infinities (including
finite `float64` values that overflow on v4's `float32` conversion), negative
values, zero-mass rows, and invalid targets are rejected. `validate=False` is
an internal fast path: such malformed numerical inputs are unsupported.

V2/v3 implement variable-total floor counts. V4 implements the versioned exact
fixed-total rule with stable descending residual allocation; equal values,
including `-0.0` and `+0.0`, use increasing column order. V2/v3 share-memory
reductions use one 256-thread block per row. V4 additionally uses block radix
sort through alphabet 1024 and four-pass shared-memory radix selection above
1024. All barriers are reached uniformly by the block. Each block owns one
output row; shared histogram atomics do not cross blocks.

Integer intervals and archive bytes are exact contract values: tolerance is
zero. Returned target probabilities are `float64` diagnostic values computed
from the variant's declared normalization; tests compare deterministic cases
exactly. NaN payload propagation is not part of the validated contract.

## Supplied-CDF decoder

The high-level decoder constructs contiguous CUDA `int64` CDFs of shape
`[streams, alphabet + 1]`, with `streams >= 1`, `alphabet >= 2`, leading zero,
strictly positive frequencies, and a final total no larger than
`nominal_total`. Payload is contiguous CUDA `int64` `[streams, payload_width]`;
bit counts and all range/code/position/count/symbol state are contiguous
`int64` vectors of length `streams`; active is contiguous `bool`; errors is
contiguous `int32`. Archive validation guarantees each bit count fits its
payload row. Mutable state and inputs must not alias.

Arithmetic settings obey the same state-bit, coder-range, and signed-product
bounds as encoding. One thread exclusively owns each stream's state and output
symbol. Inactive lanes are untouched. Kernel error flags are intentionally
reported at `assert_complete`, which synchronizes through the device-to-host
copy; `_raw` callers must synchronize and inspect errors themselves.

## Sanitizer reproduction

On a machine with an NVIDIA GPU, a compatible driver, and CUDA toolkit, force a
fresh optimized line-info build and run the focused suite with:

```bash
export TORCH_EXTENSIONS_DIR=/tmp/compression-economics-cuda-audit
CUDA_AC_VERBOSE_BUILD=1 CUDA_LAUNCH_BLOCKING=1 \
  python -m pytest -q tests/test_cuda_ac_safety.py tests/test_device_ac.py
for tool in memcheck racecheck initcheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode=99 \
    python -m pytest -q tests/test_cuda_ac_safety.py tests/test_device_ac.py
done
```

`racecheck` covers shared-memory hazards only. Global-memory safety additionally
depends on the per-row/per-stream ownership and no-alias rules documented
above.

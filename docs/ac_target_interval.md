# Target interval arithmetic coding (O3)

The global-mask compression CLI now accepts `--encoding AC_TARGET_INTERVAL`.
Its predictor computes the true next token's interval where the model scores
reside, then transfers a 32-byte record per batch row: low, high, actual frequency
total, and normalized target probability. Encoding does not receive full host
probability vectors. There is still vocabulary-sized work on the inference device.

## Transformers

```bash
python main.py --mode compress --engine transformer \
  --encoding AC_TARGET_INTERVAL --input_path data/text8 \
  --output_path artifacts/runs/o3.bin --first_n_tokens 10000 --batch_size 32
python main.py --mode decompress --input_path artifacts/runs/o3.bin \
  --output_path artifacts/runs/o3.txt
```

Use `--ac-backend numba_parallel --ac-threads 4` to select the optional compiled
coder (`pip install -e '.[parallel]'`). Python is the default. Numba buffers the
captured intervals and encodes independent streams during finalization; this
port does not overlap coding with the next model call.

CPU and CUDA are supported by the float64 quantizer. The O3 path explicitly
rejects MPS because it cannot run this float64 policy. Existing non-O3 paths
retain their current device support.

## Optional vLLM

Install `pip install -e '.[vllm,parallel]'` in a CUDA environment and replace
`--engine transformer` with `--engine vllm` in the compression command.
vLLM uses its v1 custom logits-processor API:
https://docs.vllm.ai/en/v0.11.0/features/custom_logitsprocs.html

The worker computes intervals before forcing the known token and writes only
32 bytes per row into compact shared memory. Captured rows are restored to caller
order. Both compression and decompression use the archive vocabulary, excluding
model padding IDs. vLLM owns prefix-cache management.

Options: `--gpu-memory-utilization 0.8`, `--tensor-parallel-size 1`.
This integration currently supports standard decoder-only inference without LoRA,
teacher-forced windows, or speculative decompression. Parameter counts reported
by vLLM are explicitly marked as configuration estimates.

The current environment has no CUDA. CPU Transformer round trips and simulated
vLLM worker scheduling/capture are tested; live vLLM/CUDA behavior and performance
remain to be validated. No GPU speedup is claimed.

## Quantization and archives

O3 uses normalized float64 probabilities, with integer frequencies
`floor(p * (nominal_total - alphabet_size)) + 1`. The actual total is the sum of
these frequencies and may vary per token. This differs from the existing AC/MSAC
fixed-total count redistribution. The fixed-total `--frequency-quantizer` options
do not apply to O3; a nondefault option is rejected.

O3 payloads use MSAC version 2, flag 1, inside the existing checksummed GMMS text
archive. The text header identifies `floor_float64_v1`. Existing MSAC version 1
archives remain readable. These portable archives are not the old fast-ac branch's
dictionary payloads; this port does not promise compatibility with those files.

Decompression still needs the full probability distribution to identify the
unknown next token. It uses the same floor-count rule; O3 saves transfer only on
the compression path. Use matching model, tokenizer, device/inference configuration,
vocabulary, and context policies for lossless round trips.

The bigram adapter also accepts O3 for local correctness experiments, though its
CPU output offers no GPU-to-host transfer saving.

## Measurement and follow-up work

Compression stats include `interval_transfer_bytes` (record payload bytes,
including inactive tail rows) and `interval_conversion_seconds` for the local
predictor. CPU record bytes are logical payload size, not physical device traffic.
GPU timing is currently based on host timers and is not CUDA-event profiling.
vLLM worker softmax/quantization is included in inference time; its interval
conversion timing is not separately instrumented.

The frozen-trace `evaluate_coders.py` benchmark continues to use its matched
fixed-total coders. It has no live inference and cannot measure O3 transfer savings.
Compare O3 via live compression runs, and report its different quantization rule
when comparing compression size.

Design choices and possible ablations are tracked in
[ac_target_interval_todos.txt](ac_target_interval_todos.txt).

# Coding modules

- `encoding.py`: arithmetic, ANS, Huffman, bit-packed rank, and PMATIC coders.
- `encoding_utils.py`: shared probability quantization and PMATIC helpers. The `reference` and `vectorized_exact` methods produce the same integer CDF at fixed frequency total.
- `multistream_ac.py`: portable MSAC archive and independent stream encoder/decoder.
- `paired_ac.py`: MSAP reference layout with forward/backward byte packing and compatible terminal-byte sharing. Select it with `--encoding AC_MULTISTREAM --ac-layout paired` (also works with `AC_TARGET_INTERVAL`). The decoder currently reconstructs ordinary MSAC streams before decoding; the arithmetic coder still uses its fixed final bit.
- `target_interval.py`: device-side float64 floor counts and compact target intervals (MSAC v2).
- `parallel_ac.py`: optional Numba range-encoding kernel used by MSAC.

Import these modules through `src.coding.*`. The frozen-trace comparison
lives in `src/coding/trace_benchmark.py`; its runnable experiments are in
`research/papers/cidr_2027/experiments/`.

# Coding modules

- `encoding.py`: arithmetic, ANS, Huffman, bit-packed rank, and PMATIC coders.
- `encoding_utils.py`: shared probability quantization and PMATIC helpers.
- `multistream_ac.py`: portable MSAC archive and independent stream encoder/decoder.
- `parallel_ac.py`: optional Numba range-encoding kernel used by MSAC.

Use `src.coding.*` for new imports. The matching modules directly under `src/`
forward old imports used by existing scripts and notebooks. The frozen-trace
comparison remains in `src/coder_benchmark.py`; its runnable experiments are in
`research/papers/cidr_2027/experiments/`.

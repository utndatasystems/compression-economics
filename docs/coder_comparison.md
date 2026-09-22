# Matched coder comparison

The coder benchmark operates on one immutable probability trace rather than
rerunning a predictor for every coder. This holds target symbols, floating-point
probabilities, and the integer frequency total constant while excluding model
inference from coder timings.

Each result reports ideal and quantized-distribution cross-entropy, actual
payload and archive size, framing/codebook/helper costs, encode/decode speed,
Python peak traced memory, exact round-trip status, and decoder-only numerical
perturbation outcomes. ANS block size and lane count and PMATIC `delta`/`r` are
recorded parameters.

Huffman-rank and fixed-width rank streams now have serialized, end-to-end
decoders. They are labeled `rank_transform`: their sizes are not evidence that
they code the predictor probability distribution as efficiently as AC or ANS.
The Huffman archive contains a canonical serialized codebook, so its codebook
charge is measured rather than estimated.

For a self-contained smoke benchmark:

```bash
.venv/bin/python scripts/plain_text_compression/evaluate_coders.py \
  --synthetic-symbols 10000 --alphabet-size 256 \
  --ans-block-size 256 --ans-lanes 4 \
  --perturbation-scale 1e-8 --perturbation-scale 1e-6 \
  --perturbation-scale 1e-4
```

For model results, provide an NPZ trace containing exactly:

- `probabilities`: a float matrix shaped `[symbol_count, alphabet_size]`.
- `symbols`: the corresponding integer target IDs shaped `[symbol_count]`.

```bash
.venv/bin/python scripts/plain_text_compression/evaluate_coders.py \
  --trace artifacts/traces/qwen-text8.npz
```

All current coders are marked `python_reference` and
`throughput_claim_eligible=false`. These measurements are useful for correctness,
accounting, and profiling, but paper-level throughput claims should wait for
matched native implementations. “ANS is best of both” is recorded only as a
hypothesis to test.

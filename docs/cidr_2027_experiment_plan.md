# CIDR 2027 experiment roadmap

## Research question

When, why, and under what systems constraints does predictive language-model
compression beat conventional database compression?

This document is the working roadmap. The first implementation slice isolates
physical layout (row-major versus column-major) while holding serialization,
tokenization, blocking, prediction, and entropy coding explicit and measurable.

## Repository inventory

The repository already provides:

- Hugging Face tokenization and causal predictors, including a working default
  for `Qwen/Qwen2.5-0.5B`, in `src/prediction.py`.
- Arithmetic coding, PMATIC, and rank-based bit-packed/Huffman paths in
  `src/encoding.py` and `src/global_mask_compressor.py`.
- Contiguous multistream token processing, first-token decoder seeds, KV-cache
  support, compression/decompression timing, and a serialized global-mask
  format.
- Exact token- and text-level round-trip tests for existing compression paths.
- Matched zstd and Brotli byte baselines in `src/adversarial/compression.py`.
- Deterministic adversarial generation and fixed-sequence scoring.
- Text corpora and extracted database text columns under the ignored `data/`
  tree, plus paper-specific experiment/evaluation conventions established by
  `papers/neurips_2026/`.

## CPU foundation and remaining work

The implemented CPU foundation now provides:

- Typed relational tables and byte-exact reversible row-major and column-major
  serializers, including schema, nulls, types, and lengths.
- A deterministic synthetic relational generator plus a checksum-pinned IMDb
  `title.basics` loader with stable, disjoint tuning/evaluation samples.
- A strict declarative runner with stable condition and run identifiers.
- Identity and zstd pipelines using independently checksummed blocks and a
  validated persisted index.
- Plain fixed-width token IDs and token IDs followed by zstd, using a pinned
  locally loaded Qwen tokenizer and a reversible byte mapping.
- Exact component accounting, shared-model and self-contained-archive labels,
  mandatory round trips, aggregate records, and per-block records.

The broader sweep still needs:

- Adapters for prediction, probability quantization, arithmetic coding, and
  restart state.
- LZ4 in the common runner; it is not currently a project dependency.
- Additional versioned real-table loaders beyond IMDb.
- Random-access measurements, memory instrumentation, and tail latency.
- Reproducible analysis commands for confidence intervals, ECDFs, Pareto plots,
  overhead breakdowns, and generated findings.

## Reuse boundaries

The arithmetic/PMATIC implementations, model loading, tokenizer access,
adversarial tooling, and existing classical baseline helpers should be wrapped
and reused. Existing `main.py` behavior and its binary format remain unchanged.
The relational benchmark will use a separate runner and result schema until its
contracts are stable enough to generalize.

The current rank-plus-Huffman and rank bit-packing paths are approximation
baselines, not substitutes for arithmetic coding from the same quantized
distribution. Existing batched compression behavior must not be described as
equivalent to decoder-side teacher forcing.

## Accounting contract

Every result will record the following non-overlapping byte counts:

```text
total_stored_bytes =
    payload_bytes
  + framing_bytes
  + index_bytes
  + seed_bytes
  + tokenizer_bytes
  + adapter_bytes
  + model_bytes_amortized
```

`source_bytes` is the canonical serialized input before compression.
`bits_per_source_byte` is `8 * total_stored_bytes / source_bytes`, and
`compression_factor` is `source_bytes / total_stored_bytes`. Both shared-model
and self-contained-archive accounting modes must be emitted. A result is valid
only after decoding and byte-for-byte comparison with the source.

## Staged implementation

1. **Scaffold (complete).** Establish the paper package, version-controlled
   sweep definitions, analysis boundary, artifact root, and manifest.
2. **Contracts (raw-byte slice complete).** Typed configuration, explicit
   pipelines, result records, canonical IDs, environment capture, and exact
   component accounting are implemented for identity and zstd.
3. **Relational bytes (complete).** A deterministic synthetic table generator,
   checksum-pinned IMDb loader, stable tuning/evaluation splits, and reversible
   row/column serialization include schema, types, nulls, and variable-length
   values.
4. **Block runner (raw-byte slice complete).** Independent checksummed blocks,
   a validated persisted index, identity/zstd codecs, mandatory round trips,
   and aggregate/block JSONL are runnable on CPU.
5. **Tokenizer baselines (complete).** Pinned tokenizer metadata, reversible
   byte tokenization, fixed-width token IDs, token-ID zstd, phase timings, and
   tokenizer storage charges are integrated into the CPU runner.
6. **Predictive adapters.** Reuse the smallest local causal model and arithmetic
   coder through explicit predictor/CDF interfaces; keep encoder and decoder
   timings separate.
7. **Analysis.** Produce aggregate statistics, bootstrap intervals, plots, and
   a generated Markdown findings report solely from raw results.
8. **Scale-out.** Add more representative real tables, LZ4 if available, GPU
   runs, tail latency, random access, and lifecycle-cost experiments.

## First pilot

The first runnable pilot uses a deterministic synthetic mixed-type table. Its
smoke configuration compares both layouts with identity framing and zstd level 3
at approximately 64 KiB per block on CPU. It now includes raw bytes, plain Qwen
token IDs, and both representations followed by zstd. The full configuration
adds approximately 1 MiB blocks and more rows. Predictive pipelines remain a
later stage. A parallel IMDb pilot uses the same matrix over a pinned official
`title.basics` snapshot, preserving source order after deterministic split
selection.

Committed sweep definitions live in
`papers/cidr_2027/experiments/configs/`. Generated data and results live below
`artifacts/papers/cidr-2027/` and are referenced from its tracked manifest.

## Correctness and comparability risks

- Schema/framing bytes can reverse close layout comparisons if omitted.
- Tokenizer round trips are not automatically byte-preserving for arbitrary
  input, so canonical bytes must remain the validation authority.
- Block size must refer to source-byte boundaries, with restart/index costs
  charged explicitly.
- Compression and decompression are distinct workloads; decoder timing cannot
  inherit encoder-only knowledge of target tokens.
- Probability quantization, model/tokenizer revisions, preprocessing, and coder
  versions are part of the lossless format contract.
- Averages may hide pathological blocks, so distributions and tail summaries
  are required before drawing conclusions.

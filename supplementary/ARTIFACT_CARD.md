# Anonymous supplementary artifact

This artifact accompanies the submission *When Adversarial Prediction Still
Compresses: Tokenization and Adversarial Robustness in LLM-Based Lossless
Compression*. It contains the implementation, experiment entry points, locked
software environment, evaluation code, and tests needed to reproduce the
paper's reported results. It intentionally contains no author or affiliation
information for review.

## Quick start

Start with `papers/neurips_2026/README.md`. It provides the exact commands to:

1. create the Python 3.12 environment from `uv.lock`;
2. download the pinned Qwen2.5-0.5B snapshot;
3. download and checksum the text8 input;
4. run the natural-text, adversarial, and tokenizer-fertility experiments;
5. build the pinned FSST baseline;
6. finalize and independently decode the Qwen arithmetic streams; and
7. rebuild the paper's tables and figures.

The machine-readable mapping from paper conditions to raw and finalized
artifacts is `artifacts/papers/neurips-2026/manifest.json`.

## Contents and scope

- `src/`: compression, prediction, adversarial-generation, and stream code.
- `scripts/`: paper attack generation and scoring entry points.
- `experiments/tokenizer_fertility.py`: tokenizer-fertility experiment.
- `papers/neurips_2026/experiments/`: reproducible paper stages.
- `papers/neurips_2026/evaluation/`: validation, plotting, and focused tests.
- `tests/`: unit tests for the included implementation.
- `artifacts/papers/neurips-2026/`: artifact manifest and the small,
  rebuildable quantization-sensitivity result included in the repository.

The archive does not bundle model weights, text8, generated attack sequences,
or large finalized streams. The README gives pinned sources and checksums for
external inputs, and all omitted generated artifacts are recreated by the
provided entry points. No new dataset or pretrained model is released.

## Hardware and expected cost

The primary experiments ran CPU-only on Ubuntu 24.04.1 LTS using one Intel Xeon
Gold 5318Y processor (24 physical cores, 48 hardware threads) with 125 GiB RAM.
The full-vocabulary MaxSurprisal/Byte attack is intentionally limited to 1,024
tokens because it performs a context-sensitive decode for each vocabulary item
at every generation step. The scripts checkpoint long-running generation.

## Limitations

- Full reproduction requires downloading the public Qwen model and text8 data.
- Wall-clock time is hardware-dependent and the original result files did not
  record reliable per-run elapsed times.
- Floating-point execution can vary across CPU and GPU platforms; the paper
  reports the observed cross-hardware discrepancies.
- The release reproduces the paper's Qwen2.5-0.5B evaluation and does not claim
  identical behavior for other models, tokenizers, languages, or domains.
- The archive omits exploratory notebooks and experiments unrelated to the
  submitted paper.

## Licensing and third-party assets

The code in this archive is released under the MIT License; see `LICENSE`.
Third-party assets are not redistributed:

- `Qwen/Qwen2.5-0.5B`, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`: Apache License 2.0.
- FSST, commit `e638d4cf8c26129d73c242a4127b42b975de5b63`: MIT License.
- Brotli: MIT License.
- text8: downloaded from the Large Text Compression Benchmark source listed in
  the reproduction README; it is not repackaged in this archive.

The review-stage copyright notice uses "Anonymous Authors" solely to preserve
anonymous review. It should be replaced with the correct holder information in
the public camera-ready release.

## Human subjects and consent

This artifact introduces no human-subject study, crowdsourcing activity, or new
human-derived dataset. Consequently, participant consent and compensation are
not applicable to the released code artifact.

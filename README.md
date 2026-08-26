# LLM Compression
LLM-guided text compression experiments with global token masks.

This repo explores how a language model can act as a probabilistic oracle to compress
text. A single bitmap (global mask) reduces the effective vocabulary, and the model's
next-token probabilities are encoded using arithmetic coding or rank-based schemes.

## How it works (high level)
- Tokenize input text with the chosen LLM tokenizer.
- Optionally reduce the vocabulary to only tokens seen in the input (global mask).
- Run batched, one-step LLM inference to predict next-token distributions.
- Encode each next token via:
  - `AC` (arithmetic coding), or
  - `bitpacked` / `huffman` (rank-based coding).

## Project layout

- `main.py`: compression and decompression CLI.
- `src/`: maintained compression, prediction, encoding, and training code.
- `scripts/`: standalone training, quantization, and data-generation CLIs.
- `experiments/`: version-controlled sweep definitions and run configurations.
- `evaluation/`: result loaders, baselines, plots, notebooks, and reference data.
- `tests/`: automated tests for maintained code.
- `data/`: local datasets (ignored).
- `artifacts/`: generated runs, figures, model weights, and logs (ignored).

See `experiments/README.md` and `evaluation/README.md` for the boundary between
running experiments and analyzing their output.

## Setup
1. Clone the repo:
```
git clone https://github.com/utndatasystems/summer-offsite.git
cd summer-offsite
```
2. Run the setup script to download the text8 dataset, create a virtual environment, and install dependencies:
```
./setup.sh
source .venv/bin/activate
```

## Basic usage
### Compression
```
python main.py \
  --mode compress
```

Compression using an existing adaptor:
```
python main.py \
    --mode compress \
    --input_path ./data/text8 \
    --first_n_tokens 100000 \
    --batch_size 16 \
    --lora_path ./artifacts/models/adapters/vera/text8/r4_lr0.0005_lsconstant_bs64_ep4_gas2/
```

### Decompression
```
python main.py \
  --mode decompress \
  --input_path artifacts/runs/current/compression_data.bin
```

Decompression reads all required settings from the binary header, so only the
compressed file path is required.

### Adaptor Training
```
python scripts/train_adapter.py \
    --adapter_type lora \
    --lr 0.0005 \
    --batch_size 64 \
    --r 4 \
    --epoch 4
```

## Key options
- `--input_path`: Text file to compress (compress mode) or `.bin` to decompress.
- `--output_path`: Override default output file path.
- `--model_name`: HuggingFace model name (tokenizer + LLM). Default: `Qwen/Qwen2.5-0.5B`.
- `--context_length`: Max context length for inference. Default: 1000.
- `--retain_tokens`: Context tail length when trimming. Default: 100.
- `--first_n_tokens`: Limit number of tokens processed. Default: 1000.
- `--batch_size`: Number of parallel sequences per step. Default: 1.
- `--use_kv_cache`: Enable KV cache for faster incremental inference. Default: enabled.
- `--reduce_tokens/--no_reduce_tokens`: Toggle global vocabulary reduction. Default: enabled.
- `--encoding`: `AC`, `bitpacked`, or `huffman`. Default: `AC`.
- `--print_results`: Print detailed stats to stdout. Default: disabled.

[ToDo: update key options with new training arguments]

## Outputs
- `artifacts/runs/current/compression_results.json`: aggregated experiment metrics.
- `artifacts/runs/current/compression_data.bin`: binary bitstream artifact.
- `artifacts/runs/current/text_results.txt`: reconstructed text.

## Adversarial worst-case inputs

`scripts/generate_adversarial.py` creates several equal-token-length worst-case runs
by repeatedly selecting the full-vocabulary token with the lowest finite next-token
logit. Since softmax preserves ordering, this is exactly the lowest-probability token
without numerical underflow from materializing tiny probabilities.

The full-vocabulary sequence is generated exactly once. Candidate tokens are checked
from lowest model probability upward, accepting the first extension for which
`encode(decode(token_ids)) == token_ids`. This prefix invariant guarantees that the
completed adversarial token sequence is a lossless, canonical representation of its
decoded UTF-8 text. For the occurring-token variant, those token IDs are replayed
unchanged while the logits are masked and renormalized over the tokens occurring in
the completed sequence. This is a post-hoc rescore: the mask never changes how the
worst-case sequence is constructed.

```bash
python scripts/generate_adversarial.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --start-text "The" \
  --start-text "A" \
  --start-text "In" \
  --total-length 1000 \
  --candidate-mode full \
  --candidate-mode occurring
```

To construct the lossless expansion stress test, use the canonical one-byte ASCII
alphabet:

```bash
python scripts/generate_adversarial.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --start-text "A" \
  --total-length 1000 \
  --generation-alphabet ascii-bytes \
  --candidate-mode full \
  --output-dir artifacts/runs/adversarial/qwen_05b_ascii_bytes_n1000
```

For the checked Qwen run, 1,000 raw bytes produce a 1,204.125-byte arithmetic
payload (120.4%) and a 1,239.125-byte payload-plus-bitmap total (123.9%). The full
serialized file is 1,552 bytes (155.2%). Arithmetic decoding reproduces all 1,000
token IDs and source bytes exactly. The 105.3% model-entropy floor already exceeds
100%, so expansion is not caused solely by fixed file overhead. The gap from 105.3%
to 120.4% comes mainly from the current integer frequency table: it reserves one
count for every vocabulary symbol before distributing the remaining precision.

Results are stored below `artifacts/runs/adversarial/<variant>/`. Each run has decoded
text and an authoritative `.tokens.json` file. The combined `results.json`
records the token IDs, text round-trip status, per-step log probabilities, post-hoc
mask metadata, and full-vocabulary versus mask-renormalized surprisal. The occurring
result also records that its token IDs came from the full-vocabulary run. The fixed
total length includes the starting token or starting text.

### Compression-oriented attacks

The minimum-probability attack maximizes one token's surprisal, not the compression
ratio of the decoded source. `src/compression_attacks.py` implements the corresponding
byte-level objective,

```text
surprisal_per_byte(v | prefix)
    = -log2 p(v | prefix) / added_utf8_bytes(prefix, v).
```

The byte count is measured by decoding the complete prefix before and after an
extension. This matters for tokenizers whose standalone token decoding is
context-dependent. Candidates that add no source bytes are not valid for a
byte-normalized objective.

`scripts/run_compression_attacks.py` runs a matched experiment containing:

- a seeded uniform random-token control;
- the original minimum-probability attack;
- greedy maximum surprisal per decoded byte;
- beam search over the sequence-level entropy ratio;
- beam search over the realized arithmetic-code ratio;
- ordinary text and random printable UTF-8 controls when requested;
- gzip level 9, Zstandard level 22, and Brotli quality 11 on every exact decoded
  byte sequence.

```bash
python scripts/run_compression_attacks.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --start-text "A" \
  --total-length 1000 \
  --generation-alphabet ascii-bytes \
  --beam-width 8 \
  --branch-factor 16 \
  --ordinary-text data/text8 \
  --random-utf8-bytes 1000 \
  --output-dir artifacts/runs/compression-attacks/qwen_05b_n1000
```

The actual-ratio beam owns an independent clone of the arithmetic coder for each
branch. A branch is ranked by its finalized payload size divided by decoded UTF-8
bits. Pass a measured, branch-invariant header plus bitmap cost through
`--fixed-overhead-bits` to optimize the complete serialized ratio. Keeping this
overhead explicit avoids pretending that a format-dependent header is intrinsic to
the model entropy.

Beam search is approximate unless `--branch-factor` covers the entire candidate
alphabet. It disables the KV cache because pruning changes beam ancestry. The output
records beam parameters, all token IDs, entropy-floor ratios, realized payload and
serialized ratios, raw byte sizes, and matched classical-compressor results in one
`results.json`.

### Resumable paper evaluation

The paper suite separates scalable long-sequence experiments from expensive beam
search. Its default long length is 10,000 tokens: large enough to move beyond the
1,000-token pilot without committing to the slow 100,000-token run. Beam search
defaults to 512 tokens because it keeps multiple model and arithmetic-coder states
per step.

```bash
# Full suite: long runs followed by bounded beam searches.
bash experiments/sweeps/paper_evaluation.sh all

# Run only the long greedy/replay experiments.
bash experiments/sweeps/paper_evaluation.sh core

# Override either budget independently.
PAPER_LENGTH=20000 PAPER_SEARCH_LENGTH=1000 \
  bash experiments/sweeps/paper_evaluation.sh all
```

The long suite includes the full, printable-ASCII, and canonical one-byte ASCII
alphabets, three starts, matched text8/random controls, and both full-vocabulary
and occurring-token arithmetic payloads. The attack runner writes an atomic
checkpoint after every completed condition; the payload scorer checkpoints after
every sequence and dictionary policy, while adversarial generation checkpoints
every 250 tokens. Re-running the same command skips or resumes completed work.
Configuration mismatches fail rather than silently combining incompatible rows;
use a new output directory or explicitly pass `--force` to the underlying runner.

To fill only the currently missing arithmetic payloads for an existing generated
adversarial directory:

```bash
python scripts/score_adversarial_payloads.py \
  --input-dir artifacts/runs/adversarial/qwen_05b_n1000
```

# Anonymous supplementary code

Minimal code artifact for *When Adversarial Prediction Still Compresses:
Tokenization and Adversarial Robustness in LLM-Based Lossless Compression*.
It contains only the code and configuration needed to rerun the reported
Qwen2.5-0.5B experiments. Tests, notebooks, manuscript sources, plotting code,
model weights, datasets, and generated outputs are deliberately excluded.

## Setup

The reported runs used Python 3.12.3. Install the exact locked environment:

```bash
uv sync --frozen
```

Download the pinned model snapshot into the cache expected by the code:

```bash
.venv/bin/hf download Qwen/Qwen2.5-0.5B \
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --cache-dir .cache
mkdir -p .cache/models--Qwen--Qwen2.5-0.5B/refs
printf '%s' 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  > .cache/models--Qwen--Qwen2.5-0.5B/refs/main
```

Download and verify the 100,000,000-byte text8 input:

```bash
wget -O /tmp/text8.zip http://mattmahoney.net/dc/text8.zip
mkdir -p data
unzip -o /tmp/text8.zip -d data
sha256sum data/text8
```

Expected SHA-256:
`6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7`.

## Run

Run both reported stages, or select `fertility` or `attacks`:

```bash
HF_HUB_OFFLINE=1 bash research/papers/neurips_2026/experiments/run_all.sh all
```

Set `DRY_RUN=1` to print every expanded command without running the model.
Outputs are written below `artifacts/papers/neurips-2026/`. The manifest at
`artifacts/papers/neurips-2026/manifest.json` maps paper conditions to their
generated paths. Long generation stages checkpoint completed outputs.

The main attack budget is 10,000 tokens with context length 1,000 and 100
retained tokens. Full-vocabulary MaxSurprisal/Byte uses a separate 1,024-token
budget because it decodes every vocabulary candidate at every step. These
values can be overridden through the environment variables documented directly
in `run_all.sh`.

## Contents

- `src/`: only the runtime modules imported by the experiments.
- `research/papers/neurips_2026/experiments/`: adversarial generation,
  compression, payload scoring, tokenizer fertility, and shell entry points.
- `pyproject.toml`, `uv.lock`: minimal CPU-only locked runtime.
- `artifacts/papers/neurips-2026/manifest.json`: condition-to-output mapping.

## Platform and scope

The primary CPU-only runs used Ubuntu 24.04.1 LTS, one Intel Xeon Gold 5318Y
processor (24 physical cores, 48 hardware threads), and 125 GiB RAM. Runtime is
hardware-dependent; original result files did not record reliable wall-clock
times. Floating-point execution may vary across CPU and GPU platforms.

This artifact does not redistribute or introduce a dataset or pretrained model.
Qwen2.5-0.5B is Apache-2.0 licensed. The FSST reference implementation at
commit `e638d4cf8c26129d73c242a4127b42b975de5b63` and Brotli are MIT licensed.
The upstream text8 page does not state an explicit dataset license, so text8 is
downloaded by the reviewer and is not redistributed. The included code is MIT
licensed. No human-subject study, crowdsourcing, or participant data is
involved.

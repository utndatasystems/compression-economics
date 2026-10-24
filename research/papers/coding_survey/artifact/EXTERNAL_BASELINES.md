# External LLM baselines

The main comparison includes adapters around pinned upstream implementations.
Clone each project outside this artifact so its license and environment remain
separate:

```bash
mkdir -p third_party
git clone https://github.com/fazalmittu/FineZip.git third_party/FineZip
git -C third_party/FineZip checkout 26cbc7c1
git clone https://github.com/vcskaushik/LLMzip.git third_party/LLMzip
git -C third_party/LLMzip checkout 39b4b7a9
git clone https://github.com/robtacconelli/Nacrith-GPU.git third_party/Nacrith-GPU
git -C third_party/Nacrith-GPU checkout ff29c42e
```

Run FineZip and LLMZip with the artifact environment:

```bash
.venv/bin/python research/papers/coding_survey/experiments/benchmark_finezip.py \
  --finezip-repo third_party/FineZip --dtype bfloat16 \
  --batch-size 1024 --context-size 512 \
  --model Qwen/Qwen2.5-0.5B \
  --model-revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --output results/external/finezip.json

.venv/bin/python research/papers/coding_survey/experiments/benchmark_llmzip.py \
  --llmzip-repo third_party/LLMzip --dtype bfloat16 --memory 511 \
  --model Qwen/Qwen2.5-0.5B \
  --model-revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --output results/external/llmzip.json
```

Nacrith requires its upstream llama.cpp environment, a BF16 GGUF converted
from the same pinned Qwen snapshot, and that snapshot's `tokenizer.json` and
`tokenizer_config.json`:

```bash
/path/to/nacrith-venv/bin/python research/papers/coding_survey/experiments/benchmark_nacrith.py \
  --nacrith-repo third_party/Nacrith-GPU --workers 8 \
  --model-name Qwen/Qwen2.5-0.5B \
  --gguf /path/to/Qwen2.5-0.5B-BF16.gguf \
  --tokenizer-dir /path/to/Qwen2.5-0.5B-tokenizer \
  --output results/external/nacrith.json
```

Combine the three result records:

```bash
.venv/bin/python research/papers/coding_survey/experiments/summarize_llm_baselines.py \
  results/external/finezip.json results/external/llmzip.json \
  results/external/nacrith.json results/external/all.json
```

Regenerate the main plot after the CUDA and traditional runs:

```bash
.venv/bin/python research/papers/coding_survey/experiments/plot_optimization_operating_points.py \
  --encoding-inputs results/cuda/text8-b1024-c512-r256.json \
                    results/cuda/text8-b8192-c128-r64.json \
                    results/cuda/text8-b16384-c128-r64.json \
  --inputs results/cuda/text8-b1024-c512-r256.json \
           results/cuda/text8-b8192-c128-r64.json \
           results/cuda/text8-b16384-c128-r64.json \
  --baselines results/traditional/text8-first-10MiB.json \
  --external-baselines results/external/all.json \
  --pdf results/optimization-operating-points.pdf \
  --zoomed-pdf results/optimization-operating-points-llm-zoom.pdf
```

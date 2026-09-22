# Pretrained model ladder

`research/papers/cidr_2027/experiments/configs/model_ladder.json` is the versioned catalog for pretrained causal
models. It pins model and tokenizer commits and records architecture family,
checkpoint dtype, and total and active parameter counts. The catalog contains
dense GPT-2 and Qwen2.5 ladders, an SSM Mamba ladder, and an optional Nemotron 3
Nano hybrid-MoE endpoint. Larger checkpoints are only selected explicitly or
with `--include-optional`.

The evaluator supplies the exact same UTF-8 input-byte prefix to every tokenizer.
Token counts can differ because tokenization is part of the condition. Schema-v1
rows report both input-symbol throughput and model-input-token throughput.

```bash
.venv/bin/python research/papers/cidr_2027/experiments/evaluate_model_ladder.py \
  --model gpt2-124m --model mamba-129m --model qwen2.5-494m \
  --input-bytes 10000 \
  --context-length 32 --context-length 128 \
  --batch-size 8 --batch-size 32 \
  --warmups 1 --repetitions 3
```

Add `--local-files-only` for an offline run. Selecting
`nemotron3-nano-30b-a3b` is always explicit because it needs an
accelerator-scale memory budget.

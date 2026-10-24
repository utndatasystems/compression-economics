# Experiment reference

## Pretrained model ladder

`research/papers/cidr_2027/experiments/configs/model_ladder.json` is the
versioned model catalog. It pins model and tokenizer revisions and records
architecture, checkpoint dtype, and total and active parameter counts. Optional
large models must be selected explicitly or with `--include-optional`.

The evaluator gives every tokenizer the same UTF-8 byte prefix. Because token
counts can differ, result rows report both source-symbol and model-token
throughput.

```bash
.venv/bin/python research/papers/cidr_2027/experiments/evaluate_model_ladder.py \
  --model gpt2-124m --model mamba-129m --model qwen2.5-494m \
  --input-bytes 10000 \
  --context-length 32 --context-length 128 \
  --batch-size 8 --batch-size 32 \
  --warmups 1 --repetitions 3
```

Use `--local-files-only` for offline runs. The
`nemotron3-nano-30b-a3b` endpoint always requires explicit selection.

## Plain-text result schema

Publishable observations use `plain-text-compression-result` version 1 from
`src/result_schema.py`. Records contain:

- source identity and extent;
- tokenizer and predictor identities and revisions;
- coder and execution settings;
- symbol and model-work counts;
- payload, framing, bitmap, seed, and dependency sizes;
- training, encoding, decoding, and optional phase timings;
- artifact paths and explicitly named experiment metadata.

`condition_id` hashes the dataset, tokenizer, predictor, coder, and execution
settings, excluding measurements and `repetition`. Repeated observations of one
condition therefore share an identifier.

### Accounting

```text
stream_bytes = payload_bytes + framing_bytes + bitmap_bytes + seed_bytes
```

Self-contained size additionally includes tokenizer, model, and adapter state.
If a dependency size is unknown, self-contained derived values are `null`, not
zero. Compression factor and bits per source byte are derived centrally.

`input_symbols` counts source tokenizer output, while `encoded_symbols` excludes
literal restart or seed symbols. Optional `model_input_tokens` counts all tokens
presented to predictor calls, including replayed context. Source progress and
model-work throughput remain separate metrics.

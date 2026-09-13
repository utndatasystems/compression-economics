# Plain-text compression result schema

Every publishable plain-text compression observation uses
`plain-text-compression-result`, version 1, implemented in
`src/result_schema.py`.

The record is divided into the following non-overlapping sections:

- `dataset`: exact source-byte identity, region, and size;
- `tokenizer`: tokenizer identity, revision, vocabulary, and stored state;
- `predictor`: architecture, context, parameters, adaptation, and checkpoint size;
- `coder`: entropy coder and bitstream-affecting settings;
- `execution`: device, backend, batch size, seed, hardware, and software;
- `counts`: input symbols, entropy-coded symbols, and model input tokens;
- `sizes`: payload, framing, bitmap, seeds, and external dependency bytes;
- `timings`: training, encoding, decoding, and optional phase timings;
- `artifacts`: paths to the stream, checkpoint, and related outputs;
- `notes`: explicitly named experiment-specific metadata.

`condition_id` is a stable hash of dataset, tokenizer, predictor, coder, and
execution settings. It deliberately excludes measurements so repetitions of
the same condition share an identifier.

## Accounting

The shared-model stream size is:

```text
stream_bytes = payload_bytes + framing_bytes + bitmap_bytes + seed_bytes
```

The self-contained size additionally includes tokenizer, model, and adapter
state. If any dependency size is unknown, all self-contained derived values are
`null`; the schema never silently treats an unknown dependency as zero.

Compression factor and bits per source byte are derived centrally from these
components. Producers should not calculate competing versions of these fields.

## Counter and throughput semantics

- `input_symbols` counts tokenizer output symbols in the source.
- `encoded_symbols` excludes literal restart/seed symbols.
- `model_input_tokens` counts tokens actually presented to predictor forward
  passes, including repeated context tokens. It is optional until predictor
  instrumentation is enabled.

The schema reports input-symbol throughput and model-input-token throughput as
separate derived metrics. The former measures useful source progress; the latter
measures model work and may grow when contexts are replayed.

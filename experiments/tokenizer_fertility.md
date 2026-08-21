# Tokenizer fertility on text8

[`tokenizer_fertility.py`](tokenizer_fertility.py) compares three lossless
tokenizations on held-out text8:

- one token per ASCII byte (a character-level baseline);
- a 32,000-entry byte-level BPE trained on text8;
- the pretrained `Qwen/Qwen2.5-0.5B` tokenizer.

By default, the BPE is trained on bytes `[0, 90M)` and every tokenizer is evaluated
on the final 5 MB. These are the conventional text8 train and test regions, so the
BPE does not see evaluation data. Fertility is defined as the number of corpus
tokens divided by the number of whitespace-delimited words. The results also
contain token/byte and token/character rates, inverse rates, vocabulary use,
timings, region hashes, and exact decode checks.

Run the full experiment from the repository root:

```bash
python experiments/tokenizer_fertility.py
```

The script writes `results.json`, `results.csv`, and the trained BPE tokenizer to
`artifacts/tokenizer-fertility/`. It downloads only the Qwen tokenizer files, not
the language model weights.

For a quick smoke run on smaller, explicitly disjoint regions:

```bash
python experiments/tokenizer_fertility.py \
  --train-bytes 1000000 \
  --eval-bytes 100000 \
  --eval-offset 1000000 \
  --bpe-vocab-size 8000 \
  --output-dir artifacts/tokenizer-fertility-smoke
```

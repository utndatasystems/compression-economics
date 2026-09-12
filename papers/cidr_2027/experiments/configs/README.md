# Sweep configuration contract

These TOML files are version-controlled experiment specifications. Values in
`matrix` are orthogonal sweep axes; each `[[pipelines]]` table is an explicit,
valid representation/codec combination. This avoids meaningless Cartesian
products between raw bytes, tokenizers, predictors, and entropy coders.

The runner treats byte block sizes as approximate targets because an encoded
value is never split. Model weights and external datasets must be prepared
separately and must never be downloaded implicitly by a smoke test.

`imdb_smoke.toml` and `imdb_full.toml` use the official IMDb `title.basics`
table. They pin the compressed source by SHA-256 and select disjoint tuning and
evaluation partitions by hashing `tconst`; a deterministic streaming reservoir
samples across the full split, then restores source order.
Prepare the ignored local snapshot with
`.venv/bin/python -m scripts.prepare_cidr_imdb`. Because IMDb refreshes its files
daily, pass `--overwrite` only when intentionally updating the pinned snapshot
and its configuration checksum. The data is limited to personal and
non-commercial use under IMDb's published terms.

Token pipelines must pin both `tokenizer_name` and `tokenizer_revision`. The
runner loads these assets locally and fails with a preparation command when they
are absent. `token_ids` uses the smallest fixed whole-byte width for the pinned
vocabulary; the optional `zstd` codec operates on those packed IDs.

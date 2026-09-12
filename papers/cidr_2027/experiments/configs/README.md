# Sweep configuration contract

These TOML files are version-controlled experiment specifications. Values in
`matrix` are orthogonal sweep axes; each `[[pipelines]]` table is an explicit,
valid representation/codec combination. This avoids meaningless Cartesian
products between raw bytes, tokenizers, predictors, and entropy coders.

The runner treats byte block sizes as approximate targets because an encoded
value is never split. Model weights and external datasets must be prepared
separately and must never be downloaded implicitly by a smoke test.

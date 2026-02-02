#!/usr/bin/env bash

for i in $(seq -w 1 50); do
  python3 main.py \
    --mode compress \
    --input_path data/output3.txt \
    --batch_size 128 \
    --engine transformer \
    --encoding AC \
    --output_path "output/output3/AC/output${i}.bin"
  rm compression_results_rank.json
done
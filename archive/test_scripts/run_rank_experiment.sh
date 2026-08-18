#!/usr/bin/env bash

python3 main.py \
  --mode compress \
  --input_path data/text8 \
  --first_n_tokens 500000 \
  --batch_size 128 \
  --engine transformer \
  --encoding AC \
  --model Qwen/Qwen2.5-0.5B \

python3 main.py \
  --mode compress \
  --input_path data/text8 \
  --first_n_tokens 500000 \
  --batch_size 128 \
  --engine transformer \
  --encoding bitpacked \
  --model Qwen/Qwen2.5-0.5B \

python3 main.py \
  --mode compress \
  --input_path data/text8 \
  --first_n_tokens 500000 \
  --batch_size 128 \
  --engine transformer \
  --encoding huffman \
  --model Qwen/Qwen2.5-0.5B \

python3 main.py \
  --mode compress \
  --input_path data/text8 \
  --first_n_tokens 500000 \
  --batch_size 128 \
  --engine transformer \
  --encoding AC \
  --model Qwen/Qwen3-8B \

python3 main.py \
  --mode compress \
  --input_path data/text8 \
  --first_n_tokens 500000 \
  --batch_size 128 \
  --engine transformer \
  --encoding bitpacked \
  --model Qwen/Qwen3-8B \

python3 main.py \
  --mode compress \
  --input_path data/text8 \
  --first_n_tokens 500000 \
  --batch_size 128 \
  --engine transformer \
  --encoding huffman \
  --model Qwen/Qwen3-8B \
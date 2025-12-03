# summer-offsite
Why Compress When You Have an Oracle? LLM-Powered Text Compression

## Enviroment Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Download
```
python3 datasets_download.py
```

## Basic Usage
### Compression
```
python main.py \
  --mode compress \
  --input_path data/text8 \
  --model_name Qwen/Qwen3-0.6B \
  --first_n_tokens 500000 \
  --batch_size 128
```
Arguments:
- --mode
Must be compress or decompress. Use compress to encode a text file.
- --input_path
Path to the input text file to be compressed.
- --model_name
Name of the LLM used for compression, e.g. Qwen/Qwen3-0.6B or Qwen/Qwen3-8B.
- --first_n_tokens
Only compress the first N tokens of the input. Useful to limit experiment size.
- --batch_size
Batch size for LLM inference during compression.

### Deompression
```
python3 main.py --mode decompress --input_path compression_data.bin
```
In decompression mode you only need to give the compressed file, the other information are stored in the header
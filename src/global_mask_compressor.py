from src.llm_compressor import LLMCompressor, LLMDecompressor
from src.prediction import TokenDataPreparer, TokenPredictor
from itertools import chain
import numpy as np
import time
import torch

def run_global_mask_compression(args):
    """
    Runs the compression experiment using a global token mask.

    This function compresses a sequence of tokens using an arithmetic coder guided by
    probabilities from a language model. A global bitmap of possible tokens is used
    to reduce the prediction space.

    Args:
        input_path (str): Path to the data file.
        model_name (str): Name of the language model to use.
        context_length (int): The maximum number of tokens to consider as context.
        first_n_tokens (int): The number of tokens from the dataset to process.
        use_kv_cache (bool): Whether to use the model's KV cache for faster inference.
        retain_tokens (int): Number of tokens to retain when the context length is exceeded (only with KV cache).

    Returns:
        tuple: A tuple containing:
            - str: The compressed bit string.
            - bytes: The serialized bitmap data.
            - dict: A dictionary with compression statistics.
    """
    # TODO: A better summary of the experiment's parameters
    print(f"\n----- Running Compression: Global Token Mask (tokens={args.first_n_tokens}, kv_cache={args.use_kv_cache}) -----")

    t0_tokenize = time.perf_counter()
    input_token_cnt = 0

    # Initialize the token predictor to get tokens and probabilities from the model.
    token_data_preparer = TokenDataPreparer(args)
    data_tokens = token_data_preparer.get_data_tokens()
    args = token_data_preparer.get_args()
    if args.first_n_tokens is None:
        args.first_n_tokens = len(data_tokens)


    # chunk_length = math.ceil(len(data_tokens) / args.batch_size)
    chunk_length = len(data_tokens) // args.batch_size      # minimum tokens per batch
    extra = len(data_tokens) % args.batch_size           # remainder tokens

    batches = []
    start = 0
    for i in range(args.batch_size):
        size = chunk_length + (1 if i < extra else 0)
        end = start + size
        batches.append(data_tokens[start:end])
        start = end
    first_tokens = [batch[0] for batch in batches if batch]
    batches_length = [len(batch) for batch in batches]

    # Get the compressed bitmap of the vocabulary and its size.
    bitmask_data = token_data_preparer.get_bitmap()
    total_bitmap_size = len(bitmask_data) * 8
    tokenize_time = time.perf_counter() - t0_tokenize

    llm_compressor = LLMCompressor()
    token_predictor = TokenPredictor(args, bitmap_data=bitmask_data)

    prompts = [[] for _ in range(args.batch_size)]
    compression_time = time.perf_counter()
    inference_time = 0
    ac_time = 0
    data_copy_time = 0
    softmax_time = 0
    entropy = 0.0
    rank_list = []
    probs_list = []
    # Process each token in the dataset to compress it.
    for token_idx in range(chunk_length):
        print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

        for i in range(args.batch_size):
            prompts[i].append(batches[i][token_idx])

        if len(prompts[0]) >= args.context_length:
            prompts = [prompt[-args.retain_tokens:] for prompt in prompts]
        
        input_token_cnt += args.batch_size * len(prompts[0])

        # Run LLM inference
        t0_inference = time.perf_counter()
        token_ids, probs_values, _data_copy_time, _softmax_time = token_predictor.run_batched_inference(prompts, args.use_kv_cache)
        data_copy_time += _data_copy_time
        softmax_time += _softmax_time
        inference_time += time.perf_counter() - t0_inference

        actual_next_tokens = []
        valid_mask = [] 

        for idx in range(args.batch_size):
            if token_idx + 1 < batches_length[idx]:
                token = batches[idx][token_idx + 1]
                actual_next_tokens.append(token_ids.index(token))
                valid_mask.append(True)
            else:
                actual_next_tokens.append(0)
                valid_mask.append(False)
        if args.engine == "transformer":
            if args.encoding == "AC":
                t0_ac = time.perf_counter()
                probs_cpu = probs_values.to(torch.float32).numpy()  # [B, V]

                for idx, probs in enumerate(probs_cpu):
                    if not valid_mask[idx]:
                        continue
                    target_idx = actual_next_tokens[idx]
                    llm_compressor.next_token(target_idx, probs)
                    entropy += -np.log2(probs[target_idx])
                    probs_list.append(probs[target_idx])

                ac_time += time.perf_counter() - t0_ac

            elif args.encoding in ("bitpacked", "huffman"):
                logits = probs_values.to(torch.float32)
                device = logits.device
                B, V = logits.shape

                target_idx = torch.tensor(actual_next_tokens, device=device, dtype=torch.long)  # [B]
                batch_idx = torch.arange(B, device=device)  # [B]

                target_logits = logits[batch_idx, target_idx].unsqueeze(1)

                ranks_0 = (logits > target_logits).sum(dim=1)  # [B]
                ranks = ranks_0.cpu().tolist()

                for idx in range(args.batch_size):
                    if not valid_mask[idx]:
                        continue
                    rank = ranks[idx]
                    rank_list.append(rank)
                    # llm_compressor.next_token(rank)

            else:
                raise NotImplementedError(f"Encoding method '{args.encoding}' is not implemented.")
        elif args.engine == "vllm":
            rank_list = []
            for idx, probs in enumerate(probs_values):
                if not valid_mask[idx]:
                    continue
                target_idx = actual_next_tokens[idx]
                if target_idx in probs:
                    rank_list.append(probs[target_idx].rank)
                else:
                    rank_list.append(target_idx)
        else:
            raise ValueError(f"Unsupported engine: {args.engine}")

    if args.engine == "transformer":
        if args.encoding == "AC":
            bit_string = llm_compressor.compress(encoding="AC")
        elif args.encoding == "bitpacked":
            print(f"len of rank list: {len(rank_list)}")
            print(f"max rank: {max(rank_list)}")
            bit_string = llm_compressor.compress(encoding="bitpacked", rank_list=rank_list)
        elif args.encoding == "huffman":
            print(f"len of rank list: {len(rank_list)}")
            print(f"first 10 in rank list: {rank_list[:10]}")
            bit_string, codebook = llm_compressor.compress(encoding="huffman", rank_list=rank_list)
        else:
            raise NotImplementedError(f"Encoding method '{args.encoding}' is not implemented.")
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")
    
    compression_time = time.perf_counter() - compression_time
    
    total_compression_time = time.perf_counter() - t0_tokenize
    total_arithmetic_code_size = len(bit_string)
    if args.encoding == "huffman":
        # Estimate the size of the codebook in bits
        codebook_size = sum(len(code) for code in codebook.values())
        total_arithmetic_code_size += codebook_size
        print(f"Estimated codebook size (bits): {codebook_size}")

    # Calculate final size and compression ratio.
    final_size = total_arithmetic_code_size + total_bitmap_size
    original_size_bytes = len(token_predictor.detokenize(data_tokens))

    return first_tokens, bit_string, bitmask_data, {
        "args": args.__dict__,
        "chunk_length": chunk_length,
        "chunk_size": -1, # -1 indicates global mask, not chunking
        "original_size_bytes": original_size_bytes,
        "arithmetic_code_size_bytes": total_arithmetic_code_size / 8,
        "bitmap_size_bytes": total_bitmap_size / 8,
        "final_size_bytes": final_size / 8,
        "pure_compression_factor": original_size_bytes / (total_arithmetic_code_size / 8),
        "compression_factor": original_size_bytes / (final_size / 8),
        "input_tokens_count": input_token_cnt,
        "entropy": float(entropy),
        # Timings
        "total_compression_time": total_compression_time,
        "tokenize_time": tokenize_time,
        "compression_time": compression_time,
        "inference_time": inference_time,
        "ac_time": ac_time,
        "data_copy_time": data_copy_time,
        "softmax_time": softmax_time,
        # Throughput
        "throughput_tokens_per_sec": input_token_cnt / total_compression_time,
        "throughput_kibibytes_per_sec": original_size_bytes / 1024 / total_compression_time,
        "inference_throughput_tokens_per_sec": input_token_cnt / inference_time,
        "inference_throughput_kibibytes_per_sec": original_size_bytes / 1024 / inference_time,
    }, args

def run_global_mask_decompression(
    args,
    first_tokens,
    bit_string,
    bitmap
):
    """
    Runs the decompression for a bit string compressed with a global token mask.

    This function reconstructs the original sequence of tokens by using the language
    model to predict probabilities and the arithmetic decompressor to decode the next token.

    Args:
        bit_string (str): The compressed bit string.
        input_path (str): Path to the original data file (for the token predictor).
        model_name (str): Name of the language model to use.
        context_length (int): The maximum number of tokens to consider as context.
        first_n_tokens (int): The number of tokens to decompress.
        use_kv_cache (bool): Whether to use the model's KV cache.
        retain_tokens (int): Number of tokens to retain in the context (with KV cache).

    Returns:
        list: The list of reconstructed token IDs.
    """
    print(f"\n----- Running Decompression: Global Token Mask (first_n_tokens={args.first_n_tokens}, kv_cache={args.use_kv_cache}) -----")

    # Start the decompression timer.
    t0_decompress = time.perf_counter()

    # Initialize the token predictor.
    token_predictor = TokenPredictor(
        args,
        bitmap_data=bitmap
    )

    # Get the original tokens to know the starting token and the total length.
    
    decompressor = LLMDecompressor(bit_string)
    # Start decompression with the first token from the original data.
    prompts = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]

    chunk_length = args.first_n_tokens // args.batch_size
    extra = args.first_n_tokens % args.batch_size

    batches_length = [
        chunk_length + (1 if i < extra else 0)
        for i in range(args.batch_size)
    ]
    input_tokens_cnt = 0
    inference_time = 0
    ac_time = 0
    data_copy_time = 0
    softmax_time = 0


    for token_idx in range(chunk_length):

        print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

        if len(prompts[0]) >= args.context_length:
            prompts = [prompt[-args.retain_tokens:] for prompt in prompts]

        input_tokens_cnt += args.batch_size * len(prompts[0])
        # Run LLM inference
        t0_inference = time.perf_counter()
        _, probs_values, _data_copy_time, _softmax_time = token_predictor.run_batched_inference(prompts, enable_kv_cache=args.use_kv_cache)
        data_copy_time += _data_copy_time
        softmax_time += _softmax_time
        inference_time += time.perf_counter() - t0_inference

        t0_ac = time.perf_counter()
        # Provide the actual token's indexes and the probability distributions to the compressor.
        for idx, probs in enumerate(probs_values.to(torch.float32).numpy()):
            if token_idx + 1 < batches_length[idx]:
                # Decompress the next token's index from the bit string.
                next_token_idx = decompressor.decompress(probs)
                next_token = token_predictor.get_token_by_id(next_token_idx)
                
                # Append the decompressed token to the context for the next step.
                prompts[idx].append(next_token)
                reconstructed_tokens[idx].append(next_token)
        ac_time += time.perf_counter() - t0_ac
    
    reconstructed_tokens = list(chain.from_iterable(reconstructed_tokens))
    t0_detokenize = time.perf_counter()
    detoken_string = token_predictor.detokenize(reconstructed_tokens)
    detokenize_time = time.perf_counter() - t0_detokenize

    decompression_time = time.perf_counter() - t0_decompress

    return reconstructed_tokens, detoken_string, {
        "args": args.__dict__,
        "decompression_time_sec": decompression_time,
        "input_tokens_cnt": input_tokens_cnt,
        # Timings
        "total_decompression_time": decompression_time,
        "detokenize_time": detokenize_time,
        "inference_time": inference_time,
        "ac_time": ac_time,
        "data_copy_time": data_copy_time,
        "softmax_time": softmax_time,
        # Throughput
        "throughput_kibibytes_per_sec": len(detoken_string) / 1024 / decompression_time,
        "inference_throughput_kibibytes_per_sec": len(detoken_string) / 1024 / inference_time,
    }
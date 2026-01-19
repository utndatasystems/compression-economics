from llm_testing.llm_compressor import LLMCompressor, LLMDecompressor
from llm_testing.prediction import TokenDataPreparer, TokenPredictor
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

        # Provide the actual token's indexes and the probability distributions to the compressor.
        # actual_next_tokens = [batches[idx][token_idx+1] for idx in range(args.batch_size) if token_idx + 1 < batches_length[idx]]
        # actual_next_tokens = [token_ids.index(token) for token in actual_next_tokens]
        # t0_ac = time.perf_counter()
        # for idx, probs in enumerate(probs_values.to(torch.float32).numpy()):
        #     if token_idx + 1 < batches_length[idx]:
        #         # llm_compressor.next_token(actual_next_tokens[idx], probs)
        #         entropy += -(np.log2(probs[actual_next_tokens[idx]]))
        # ac_time += time.perf_counter() - t0_ac

    compression_time = time.perf_counter() - compression_time

    # Finalize the compression to get the bit string.
    bit_string = llm_compressor.compress()
    total_compression_time = time.perf_counter() - t0_tokenize
    total_arithmetic_code_size = len(bit_string)


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


    for token_idx in range(chunk_length):

        print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

        if len(prompts[0]) >= args.context_length:
            prompts = [prompt[-args.retain_tokens:] for prompt in prompts]

        input_tokens_cnt += args.batch_size * len(prompts[0])
        # Run LLM inference
        _, probs_values, _, _ = token_predictor.run_batched_inference(prompts)

        # Provide the actual token's indexes and the probability distributions to the compressor.
        for idx, probs in enumerate(probs_values.numpy()):
            if token_idx + 1 < batches_length[idx]:
                # Decompress the next token's index from the bit string.
                next_token_idx = decompressor.decompress(probs)
                next_token = token_predictor.get_token_by_id(next_token_idx)
                
                # Append the decompressed token to the context for the next step.
                prompts[idx].append(next_token)
                reconstructed_tokens[idx].append(next_token)
    
    reconstructed_tokens = list(chain.from_iterable(reconstructed_tokens))

    detoken_string = token_predictor.detokenize(reconstructed_tokens)

    decompression_time = time.perf_counter() - t0_decompress

    return reconstructed_tokens, detoken_string, {
        "decompression_time_sec": decompression_time,
        "input_tokens_cnt": input_tokens_cnt,
    }

def run_global_mask_compression_decompression_test(
    input_path,
    model_name="Qwen/Qwen2.5-0.5B",
    context_length=1000,
    first_n_tokens=1000,
    use_kv_cache=True,
    retain_tokens=100,
    batch_size=1
):
    """
    Runs a full compression and decompression cycle and verifies the result.

    This function is a test utility to ensure that the compression is lossless by
    comparing the original tokens with the reconstructed tokens.

    Args:
        input_path (str): Path to the data file.
        model_name (str): Name of the language model.
        context_length (int): Maximum context length.
        first_n_tokens (int): Number of tokens to process.
        use_kv_cache (bool): Whether to use the KV cache.
        retain_tokens (int): Number of tokens to retain in the context.
        batch_size (int): The batch size for inference.

    Returns:
        dict: A dictionary containing the compression results and a verification status.
    """
    print(f"\n===== Running Compression + Decompression Test (kv_cache={use_kv_cache}) =====")
    
    # Create a mock args object to pass to the functions.
    class Args:
        pass
    args = Args()
    args.input_path = input_path
    args.text_input = None
    args.model_name = model_name
    args.context_length = context_length
    args.first_n_tokens = first_n_tokens
    args.use_kv_cache = use_kv_cache
    args.retain_tokens = retain_tokens
    args.batch_size = batch_size
    args.reduce_tokens = True
    
    # Step 1: Compress the data.
    first_token, bit_string, bitmask_data, compression_result, args = run_global_mask_compression(args)
    
    print("\nCompression complete.")
    print(f"Compression Ratio: {compression_result['compression_ratio_percent']:.2f}%")
    print(f"Compression Time: {compression_result['inference_time_sec']:.2f} sec")

    # Step 2: Decompress the data.
    reconstructed_tokens, _, _ = run_global_mask_decompression(
        args,
        first_token,
        bit_string,
        bitmask_data
    )
    
    print("\nDecompression complete.")

    # Step 3: Verify that the reconstructed tokens match the original ones.
    token_data_preparer = TokenDataPreparer(args)
    original_tokens = token_data_preparer.get_data_tokens()
    
    match = reconstructed_tokens == original_tokens
    print(f"Verification: {'✅ PASS' if match else '❌ FAIL'}")
    
    return {
        "compression_result": compression_result,
        "verification_passed": match
    }

if __name__ == "__main__":
    run_global_mask_compression_decompression_test(
        input_path="../data/text8",
        first_n_tokens=100000,
        context_length=1000,
        use_kv_cache=True,
        retain_tokens=100,
        batch_size=32
    )
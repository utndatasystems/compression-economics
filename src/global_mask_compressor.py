"""
Global-mask LLM compression/decompression.

This module implements a compression experiment that uses a single global bitmap of
allowed tokens to reduce the LLM prediction space. Tokens are processed in batches
to amortize inference, and the resulting probability distributions are encoded via
arithmetic coding ("AC") or rank-based schemes ("bitpacked", "huffman") for the
transformer engine. Decompression mirrors compression using the same model and
bitmap to reconstruct tokens from the bitstream.
"""

from tqdm import tqdm
from wandb.plot import line

from src.encoding import LLMCompressor, LLMDecompressor
from src.prediction import TokenDataPreparer, TokenPredictor
from itertools import chain
import numpy as np
import time
import torch

def run_global_mask_compression(args):
    """
    Run compression using a global token mask and batched LLM inference.

    The dataset is tokenized, split into `batch_size` contiguous batches, and then
    processed step-wise. At each step the LLM predicts probabilities for the next
    token given the current prompt; the true next token is then encoded using the
    chosen encoding scheme. A single bitmap describing the full vocabulary mask is
    stored alongside the compressed bitstream.

    Args:
        args (argparse.Namespace): Experiment configuration. Expected fields include:
            input_path (str): Path to the data file.
            model_name (str): Name of the language model to use.
            context_length (int): Maximum number of tokens to use as context.
            first_n_tokens (int | None): Number of tokens to process (defaults to all).
            batch_size (int): Number of parallel sequences to process.
            use_kv_cache (bool): Whether to enable KV cache for inference.
            retain_tokens (int): Tokens to retain when truncating context.
            engine (str): "transformer".
            encoding (str): "AC", "bitpacked", or "huffman".

    Returns:
        tuple: (first_tokens, bit_string, bitmask_data, stats, args)
            - first_tokens (list[int]): First token from each batch (seed for decoding).
            - bit_string (str): The compressed bit string.
            - bitmask_data (bytes): Serialized bitmap describing allowed vocabulary.
            - stats (dict): Compression statistics (sizes, timings, throughput).
            - args (argparse.Namespace): Possibly updated args from token preparation.
    """
    print(f"\n----- Running Compression: Global Token Mask (tokens={args.first_n_tokens}, kv_cache={args.use_kv_cache}) -----")

    t0_tokenize = time.perf_counter()
    input_token_cnt = 0

    # Initialize the token predictor to get tokens and probabilities from the model.
    token_data_preparer = TokenDataPreparer(args)
    data_tokens = token_data_preparer.get_data_tokens()
    args = token_data_preparer.get_args()
    if args.first_n_tokens is None:
        args.first_n_tokens = len(data_tokens)


    # Split tokens into contiguous batches of (roughly) equal length.
    # chunk_length is the minimum tokens per batch; extras are distributed one per batch.
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
    # Capture the first token from each batch for seeding decompression.
    first_tokens = [batch[0] for batch in batches if batch]
    batches_length = [len(batch) for batch in batches]

    # Get the compressed bitmap of the vocabulary and its size.
    bitmask_data = token_data_preparer.get_bitmap()
    total_bitmap_size = len(bitmask_data) * 8
    tokenize_time = time.perf_counter() - t0_tokenize

    llm_compressor = LLMCompressor()
    token_predictor = TokenPredictor(args, bitmap_data=bitmask_data)

    # Per-batch prompt buffers that grow token-by-token.
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
    for token_idx in tqdm(range(chunk_length)):
        print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

        # Append the current token from each batch to its prompt context.
        for i in range(args.batch_size):
            prompts[i].append(batches[i][token_idx])

        # Trim context to keep inference cost bounded.
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

        # Build a mask for batches that still have a "next token" at this step.
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

                # Encode each valid batch's next token using arithmetic coding.
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

                # Compute rank of the true token within the model's logits.
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
    # Bitstream length (in bits) plus bitmap size yields the final compressed size.
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
    bitmap,
):
    """
    Decompress a bit string produced by the global-mask compressor.

    The LLM predicts next-token probabilities for each batch step, and the arithmetic
    decompressor uses those probabilities to recover the encoded token indices. The
    resulting batches are concatenated to recover the original token order.

    Args:
        args (argparse.Namespace): Same configuration used for compression.
        first_tokens (list[int]): First token from each batch.
        bit_string (str): The compressed bit string.
        bitmap (bytes): Serialized bitmap describing allowed vocabulary.

    Returns:
        tuple: (reconstructed_tokens, detoken_string, stats)
            - reconstructed_tokens (list[int]): Recovered token IDs (flattened).
            - detoken_string (str): Detokenized text.
            - stats (dict): Decompression timings and throughput.
    """
    print(f"\n----- Running Decompression: Global Token Mask (first_n_tokens={args.first_n_tokens}, kv_cache={args.use_kv_cache}) -----")

    # Start the decompression timer.
    t0_decompress = time.perf_counter()

    # Initialize the token predictor.
    token_predictor = TokenPredictor(
        args,
        bitmap_data=bitmap)

    # Get the original tokens to know the starting token and the total length.
    decompressor = LLMDecompressor(bit_string)

    # Start decompression with the first token from the original data.
    prompts = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]

    chunk_length = args.first_n_tokens // args.batch_size # minimum tokens per batch
    extra = args.first_n_tokens % args.batch_size

    batches_length = [
        chunk_length + (1 if i < extra else 0)
        for i in range(args.batch_size)]
    
    input_tokens_cnt = 0
    inference_time = 0
    ac_time = 0
    data_copy_time = 0
    softmax_time = 0

    # Determine max tokens to decode, set to first_n_tokens if not specified, or spec_k if in speculative mode.
    max_tokens = args.first_n_tokens
    if hasattr(args, "spec_k") and args.spec_k is not None:
        max_tokens = args.spec_k
    total_decoded = len(first_tokens) if first_tokens else 0

    for token_idx in range(chunk_length):
        if total_decoded >= max_tokens:
            break

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

                total_decoded += 1 

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

def run_global_mask_decompression_lean(
    args,
    first_tokens,
    bit_string,
    bitmap,
):
    """
    Decompress a bit string produced by the global-mask compressor.

    The LLM predicts next-token probabilities for each batch step, and the arithmetic
    decompressor uses those probabilities to recover the encoded token indices. The
    resulting batches are concatenated to recover the original token order.

    Args:
        args (argparse.Namespace): Same configuration used for compression.
        first_tokens (list[int]): First token from each batch.
        bit_string (str): The compressed bit string.
        bitmap (bytes): Serialized bitmap describing allowed vocabulary.

    Returns:
        tuple: (reconstructed_tokens, detoken_string, stats)
            - reconstructed_tokens (list[int]): Recovered token IDs (flattened).
            - detoken_string (str): Detokenized text.
            - stats (dict): Decompression timings and throughput.
    """
    print(f"\n----- Running Decompression: Global Token Mask (first_n_tokens={args.first_n_tokens}, kv_cache={args.use_kv_cache}) -----")

    # Initialize the token predictor.
    token_predictor = TokenPredictor(
        args,
        bitmap_data=bitmap)

    # Get the original tokens to know the starting token and the total length.
    decompressor = LLMDecompressor(bit_string)

    # Start decompression with the first token from the original data.
    prompts = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]

    chunk_length = args.first_n_tokens // args.batch_size # minimum tokens per batch
    extra = args.first_n_tokens % args.batch_size

    batches_length = [
        chunk_length + (1 if i < extra else 0)
        for i in range(args.batch_size)]
    
    input_tokens_cnt = 0
    inference_time = 0
    ac_time = 0
    data_copy_time = 0
    softmax_time = 0

    # Determine max tokens to decode, set to first_n_tokens if not specified, or spec_k if in speculative mode.
    max_tokens = args.first_n_tokens
    if hasattr(args, "spec_k") and args.spec_k is not None:
        max_tokens = args.spec_k
    total_decoded = len(first_tokens) if first_tokens else 0

    for token_idx in range(chunk_length):
        if total_decoded >= max_tokens:
            break

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

                total_decoded += 1 

        ac_time += time.perf_counter() - t0_ac
    
    reconstructed_tokens = list(chain.from_iterable(reconstructed_tokens))
    t0_detokenize = time.perf_counter()
    detoken_string = token_predictor.detokenize(reconstructed_tokens)


    return reconstructed_tokens, detoken_string


def run_global_mask_speculative_decompression(args,
    first_tokens,
    bit_string,
    bitmap):
    """
    Run speculative decompression using a global token mask.

    Args:
        args (argparse.Namespace): Experiment configuration.

    Returns:
        tuple: (reconstructed_tokens, detoken_string, stats)
            - reconstructed_tokens (list[int]): Recovered token IDs (flattened).
            - detoken_string (str): Detokenized text.
            - stats (dict): Decompression timings and throughput.
    """

    args.spec_k = 5

    # Initialize the token predictor.
    token_predictor_teacher = TokenPredictor(
        args,
        bitmap_data=bitmap)

    token_predictor_student = TokenPredictor(
        args,
        bitmap_data=bitmap)   

    # Get the original tokens to know the starting token and the total length.
    decompressor = LLMDecompressor(bit_string)

    # Start decompression with the first token from the original data.
    prompts = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]

    chunk_length = args.first_n_tokens // args.batch_size
    extra = args.first_n_tokens % args.batch_size

    batches_length = [
        chunk_length + (1 if i < extra else 0)
        for i in range(args.batch_size)]
    
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

        t0_inference = time.perf_counter()

        # STEP 1: Draft generation
        # make call to run_global_mask_decompression with student model
        draft_tokens_batch, _ = run_global_mask_decompression_lean(
            args,
            first_tokens,
            bit_string,
            bitmap)

        #draft_tokens_batch = [
        #    token_predictor_student.generate_draft(prompt, k=args.spec_k)
        #    for prompt in prompts]

        # STEP 2: Run target model once on extended prompts
        extended_prompts = [
            prompt + [draft_tokens_batch[i]]
            for i, prompt in enumerate(prompts)]

        _, probs_seq, _data_copy_time, _softmax_time = token_predictor_teacher.run_batched_inference(
            extended_prompts,
            enable_kv_cache=args.use_kv_cache)

        data_copy_time += _data_copy_time
        softmax_time += _softmax_time
        inference_time += time.perf_counter() - t0_inference

        t0_ac = time.perf_counter()

        # STEP 3: Verify + accept/reject
        for idx in range(len(prompts)):
            draft_tokens = draft_tokens_batch[idx]
            probs_steps = probs_seq[idx]

            accepted = 0

            for step, probs in enumerate(probs_steps):
                #TODO: something wrong with probs datatype or shape that's causing decompressor to fail
                decoded_idx = decompressor.decompress(probs)
                draft_token = draft_tokens[step]

                if decoded_idx == draft_token:
                    accepted += 1
                else:
                    # mismatch → fallback token
                    next_token = token_predictor_teacher.get_token_by_id(decoded_idx)
                    prompts[idx].append(next_token)
                    reconstructed_tokens[idx].append(next_token)
                    break

            # accept prefix
            if accepted > 0:
                accepted_tokens = draft_tokens[:accepted]
                prompts[idx].extend(accepted_tokens)
                reconstructed_tokens[idx].extend(accepted_tokens)

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
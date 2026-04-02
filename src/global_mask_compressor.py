"""
Global-mask LLM compression/decompression.

This module implements a compression experiment that uses a single global bitmap of
allowed tokens to reduce the LLM prediction space. Tokens are processed in batches
to amortize inference, and the resulting probability distributions are encoded via
arithmetic coding ("AC") or rank-based schemes ("bitpacked", "huffman") for the
transformer engine. Decompression mirrors compression using the same model and
bitmap to reconstruct tokens from the bitstream.
"""

from src.encoding import LLMCompressor, LLMDecompressor
from src.prediction import TokenDataPreparer, TokenPredictor
from itertools import chain
import numpy as np
import time
import torch
from tqdm import tqdm
from wandb.plot import line
from itertools import chain

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
    token_predictor = TokenPredictor(args, bitmap_data=bitmap)

    # Get the original tokens to know the starting token and the total length.
    decompressor = LLMDecompressor(bit_string)

    # Seed each batch with its initial token from the original data (required for autoregressive decoding)
    prompts = [[first_tokens[i]] for i in range(args.batch_size)]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size) if first_tokens]

    # Set minimum tokens per batch = chunk_length 
    chunk_length = args.first_n_tokens // args.batch_size
    extra = args.first_n_tokens % args.batch_size

    # Adjust length for extra tokens 
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
    total_decoded = len(first_tokens) if first_tokens else 0

    # Iterate through all tokens in chunk length 
    for token_idx in range(chunk_length):
        if total_decoded >= max_tokens: 
            # if max_tokens are reached, break 
            break

        # Bitstring is unbatched, therefore we have to calculate batch id again using chunk-length
        print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

        # Truncate context if it exceeds model limit (prevents overflow)
        if len(prompts[0]) >= args.context_length:
            prompts = [prompt[-args.retain_tokens:] for prompt in prompts]

        # 
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
                #print(f'probs_steps shape: {probs.shape}') --> probs_steps shape: (357,)
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



def run_global_mask_speculative_decompression(args, first_tokens, bit_string, bitmap):
    """
    Run speculative decompression using a global token mask.
    """
    args.spec_k = 5

    # Initialize token predictors
    token_predictor_teacher = TokenPredictor(args, bitmap_data=bitmap)
    token_predictor_student = TokenPredictor(args, bitmap_data=bitmap)

    decompressor = LLMDecompressor(bit_string)

    # Handle empty first_tokens
    if not first_tokens:
        raise ValueError("first_tokens cannot be empty")
    
    """ToDo: set batch size to one and keep only one first token + test function for draft function"""
    batch_size = len(first_tokens)

    # Initialize prompts as list of lists with one element
    prompts = [[first_tokens[i]] for i in range(batch_size)] 

    # Initialize list for reconstructed tokens
    reconstructed_tokens = [[first_tokens[i]] for i in range(batch_size)]

    # Determine length of chunks in each batch
    chunk_length = args.first_n_tokens // args.batch_size
    extra = args.first_n_tokens % args.batch_size
    batches_length = [chunk_length + (1 if i < extra else 0) for i in range(args.batch_size)]
    
    input_tokens_cnt = 0
    inference_time = 0
    ac_time = 0
    data_copy_time = 0
    softmax_time = 0

    t0_decompress = time.perf_counter()  # Start timer for full decompression

    # Continue until every batch item reaches its target length, fill chunk length with either speculative tokens or fallback
    while any(len(reconstructed_tokens[i]) < batches_length[i] for i in range(batch_size)):

        # Only process unfinished batches
        active_indices = [
            i for i in range(batch_size)
            if len(reconstructed_tokens[i]) < batches_length[i]]

        if not active_indices:
            break

        # Select batches / prompts that need to be processed 
        active_prompts = [prompts[i] for i in active_indices]

        # Shorten context_length if retain_token is set
        if len(active_prompts[0]) >= args.context_length:
            for i in active_indices:
                prompts[i] = prompts[i][-args.retain_tokens:]
            active_prompts = [prompts[i] for i in active_indices]

        # Token counting for throughput
        input_tokens_cnt += sum(len(prompts[i]) for i in active_indices)
        t0_inference = time.perf_counter()

        # STEP 1: Draft generation with student for "active" batches

        #draft_tokens_batch = [
        #    list(token_predictor_student.generate_draft(prompts[i], k=args.spec_k))
        #    for i in active_indices]
        
        #draft_tokens_batch = [
        #    list(token_predictor_student.generate_draft(prompts[i], k=args.spec_k)[0])
        #    for i in active_indices] # 192 = 32 * 6 

        draft_tokens_batch = [list(token_predictor_student.run_batched_inference(prompts[i], enable_kv_cache=False))
                                for i in active_indices]
        
            
        # STEP 2: Build teacher inputs correctly
        # Need k+1 prompts:
        #   p
        #   p+d1
        #   p+d1+d2
        #   ...
        #   p+d1+...+dk

        # Extend prompts by student model draft --> input for teacher model
        extended_prompts = [] 
        for prompt, draft_tokens in zip(active_prompts, draft_tokens_batch):
            current = prompt.copy()
            extended_prompts.append(current.copy())  # for verifying d1

            for d in draft_tokens:
                current = current + [int(d)]
                extended_prompts.append(current.copy())  # for next verification / final token

        print(f'Extended prompts Shape {len(extended_prompts)} x {len(extended_prompts[1])}')
        # Extended prompts Shape 192 x 1

        # Feed extended prompts to teacher, does not to be uniform length
        tokens_teacher, probs_seq, _data_copy_time, _softmax_time = token_predictor_teacher.run_batched_inference(
            extended_prompts, # feed in extended_prompts as list of lists
            enable_kv_cache=args.use_kv_cache,) # disable caching? 
        
        print('Compare: ')
        print('Teacher tokens', tokens_teacher)
        print('Student tokens', draft_tokens_batch)

        data_copy_time += _data_copy_time
        softmax_time += _softmax_time
        inference_time += time.perf_counter() - t0_inference

        t0_ac = time.perf_counter()
        group_size = args.spec_k + 1

        # Verification for active batches
        for local_idx, batch_idx in enumerate(active_indices):
            remaining = batches_length[batch_idx] - len(reconstructed_tokens[batch_idx])
            if remaining <= 0: 
                # if a batch is fully filled up with reconstructed tokens, skip it 
                continue

            # for each remaining batche, take draft_tokens to verification 
            draft_tokens = [int(x) for x in draft_tokens_batch[local_idx]] # length 5 
            print(len(draft_tokens)) 

            # Teacher probs for this sequence:
            # probs_steps[0] verifies d1
            # probs_steps[1] verifies d2
            # ...
            # probs_steps[k-1] verifies dk
            # probs_steps[k] gives next token after all accepted drafts
            start = local_idx * group_size
            end = start + group_size
            probs_steps = probs_seq[start:end]

            print('Start, end', start, end) # Start, end 0 6, Start, end 6 12

            accepted_tokens = []
            mismatch = False

            # Verify up to k draft tokens, but do not exceed remaining target length
            verify_count = min(args.spec_k, remaining)

            for step in range(verify_count):
                probs = probs_steps[step].detach().cpu().numpy().astype(np.float64)
                decoded_idx = int(decompressor.decompress(probs))
                draft_token = draft_tokens[step]

                # Debug info 
                print(
                    f"Debugging Info \n"
                    f"[VERIFY] batch={batch_idx} step={step} "
                    f"draft={draft_token} decoded idx={decoded_idx} "
                    f"{'✓' if decoded_idx == draft_token else '✗'} "
                    f"p(draft)={probs[draft_token]:.4e} "
                    f"p(decoded)={probs[decoded_idx]:.4e}")

                if decoded_idx == draft_token:
                    accepted_tokens.append(draft_token)
                else:
                    # Append accepted tokens first, then fallback token
                    prompts[batch_idx].extend(accepted_tokens)
                    reconstructed_tokens[batch_idx].extend(accepted_tokens)

                    prompts[batch_idx].append(decoded_idx)
                    reconstructed_tokens[batch_idx].append(decoded_idx)

                    mismatch = True
                    break

            if mismatch:
                continue

            # All verified draft tokens accepted
            prompts[batch_idx].extend(accepted_tokens)
            reconstructed_tokens[batch_idx].extend(accepted_tokens)

            remaining = batches_length[batch_idx] - len(reconstructed_tokens[batch_idx])

            # If all k accepted and we still need another token,
            # emit one more token from the final teacher distribution
            if verify_count == args.spec_k and remaining > 0:
                probs = probs_steps[args.spec_k].detach().cpu().numpy().astype(np.float64)
                decoded_idx = int(decompressor.decompress(probs))

                prompts[batch_idx].append(decoded_idx)
                reconstructed_tokens[batch_idx].append(decoded_idx)

        break

        ac_time += time.perf_counter() - t0_ac

    reconstructed_tokens_flat = list(chain.from_iterable(reconstructed_tokens))

    t0_detokenize = time.perf_counter()
    detoken_string = token_predictor_teacher.detokenize(reconstructed_tokens_flat)
    detokenize_time = time.perf_counter() - t0_detokenize

    decompression_time = time.perf_counter() - t0_decompress

    return reconstructed_tokens_flat, detoken_string, {
        "args": args.__dict__,
        "decompression_time_sec": decompression_time,
        "input_tokens_cnt": input_tokens_cnt,
        "total_decompression_time": decompression_time,
        "detokenize_time": detokenize_time,
        "inference_time": inference_time,
        "ac_time": ac_time,
        "data_copy_time": data_copy_time,
        "softmax_time": softmax_time,
        "throughput_kibibytes_per_sec": len(detoken_string) / 1024 / decompression_time,
        "inference_throughput_kibibytes_per_sec": len(detoken_string) / 1024 / max(inference_time, 1e-12),
    }




     

def other_function():
    # Iterate through all tokens in chunk length 
    for token_idx in range(chunk_length):
        print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

        # Truncate prompts to retain_tokens if needed
        if len(prompts[0]) >= args.context_length:
            prompts = [prompt[-args.retain_tokens:] for prompt in prompts]

        input_tokens_cnt += args.batch_size * len(prompts[0])
        t0_inference = time.perf_counter()

        # STEP 1: Draft generation
        draft_tokens_batch = [
            token_predictor_student.generate_draft(prompt, k=args.spec_k)
            for prompt in prompts]
        
        print(f'draft_tokens_batch shape: {len(draft_tokens_batch)} x {len(draft_tokens_batch[0])}')  # Should be [B, spec_k] = draft_tokens_batch shape: 32 x 5
        # Shape is 32 x 5 = Batch x spec_k, but we want to ensure it's a list of lists for the next steps
        print(f'Generated draft tokens {draft_tokens_batch[0]}')  # Generated draft tokens [13, 198, 198, 464, 1708]

        # Ensure draft_tokens_batch is list of lists
        draft_tokens_batch = [list(draft) if not isinstance(draft, list) else draft for draft in draft_tokens_batch]
        print(draft_tokens_batch[0])  # Print the first batch of draft tokens to verify it's a list

        # STEP 2: Prepare input for and teacher inference
        # This was wrong because we need to run the teacher on each extended prompt with the corresponding draft token, not just the first one. We need to loop through each prompt and its corresponding draft tokens to create the extended prompts correctly.

        extended_prompts = []
        # cumulative extensions of original prompts by draft tokens 
        for prompt, draft_tokens in zip(prompts, draft_tokens_batch):
            current = prompt.copy()  # start from the original prompt
    
            for draft_token in draft_tokens:
                current = current + [draft_token]  # grow cumulatively
                extended_prompts.append(current)


        #[[36393, 13], [36393, 13, 198], [36393, 13, 198, 198], [36393, 13, 198, 198, 464], [36393, 13, 198, 198, 464, 1708], [286, 262], [286, 262, 366], [286, 262, 366, 40], [286, 262, 366, 40, 12], [286, 262, 366, 40, 12, 20], [1231, 262], [1231, 262, 779], [1231, 262, 779, 286], [1231, 262, 779, 286, 257], [1231, 262, 779, 286, 257, 366], [290, 262], [290, 262, 366], [290, 262, 366, 40], [290, 262, 366, 40, 12], [290, 262, 366, 40, 12, 40], [4003, 11], [4003, 11, 262], [4003, 11, 262, 1708], [4003, 11, 262, 1708, 318], [4003, 11, 262, 1708, 318, 257], [15529, 198], [15529, 198, 198], [15529, 198, 198, 1659], [15529, 198, 198, 1659, 262], [15529, 198, 198, 1659, 262, 1708], [16652, 13], [16652, 13, 198], [16652, 13, 198, 198], [16652, 13, 198, 198, 464], [16652, 13, 198, 198, 464, 1708], [15529, 198], [15529, 198, 198], [15529, 198, 198, 1659], [15529, 198, 198, 1659, 262], [15529, 198, 198, 1659, 262, 1708], [2751, 198], [2751, 198, 198], [2751, 198, 198, 464], [2751, 198, 198, 464, 1708], [2751, 198, 198, 464, 1708, 318], [18005, 13], [18005, 13, 405], [18005, 13, 405, 198], [18005, 13, 405, 198, 198], [18005, 13, 405, 198, 198, 464], [38547, 13], [38547, 13, 198], [38547, 13, 198, 198], [38547, 13, 198, 198, 464], [38547, 13, 198, 198, 464, 1708], [198, 198], [198, 198, 464], [198, 198, 464, 1708], [198, 198, 464, 1708, 318], [198, 198, 464, 1708, 318, 257], [2, 198], [2, 198, 198], [2, 198, 198, 2], [2, 198, 198, 2, 198], [2, 198, 198, 2, 198, 198], [2, 198], [2, 198, 198], [2, 198, 198, 2], [2, 198, 198, 2, 198], [2, 198, 198, 2, 198, 198], [23, 13], [23, 13, 15], [23, 13, 15, 13], [23, 13, 15, 13, 15], [23, 13, 15, 13, 15, 13], [18938, 13], [18938, 13, 198], [18938, 13, 198, 198], [18938, 13, 198, 198, 464], [18938, 13, 198, 198, 464, 1708], [33042, 13], [33042, 13, 198], [33042, 13, 198, 198], [33042, 13, 198, 198, 464], [33042, 13, 198, 198, 464, 1708], [23, 13], [23, 13, 15], [23, 13, 15, 13], [23, 13, 15, 13, 15], [23, 13, 15, 13, 15, 13], [19, 13], [19, 13, 15], [19, 13, 15, 13], [19, 13, 15, 13, 15], [19, 13, 15, 13, 15, 13], [1433, 13], [1433, 13, 15], [1433, 13, 15, 13], [1433, 13, 15, 13, 15], [1433, 13, 15, 13, 15, 14], [2920, 13], [2920, 13, 198], [2920, 13, 198, 198], [2920, 13, 198, 198, 464], [2920, 13, 198, 198, 464, 1708], [198, 198], [198, 198, 464], [198, 198, 464, 1708], [198, 198, 464, 1708, 318], [198, 198, 464, 1708, 318, 257], [198, 198], [198, 198, 464], [198, 198, 464, 1708], [198, 198, 464, 1708, 318], [198, 198, 464, 1708, 318, 257], [44939, 13], [44939, 13, 198], [44939, 13, 198, 198], [44939, 13, 198, 198, 464], [44939, 13, 198, 198, 464, 1708], [2, 198], [2, 198, 198], [2, 198, 198, 2], [2, 198, 198, 2, 198], [2, 198, 198, 2, 198, 198], [2388, 13], [2388, 13, 405], [2388, 13, 405, 8], [2388, 13, 405, 8, 198], [2388, 13, 405, 8, 198, 198], [18005, 13], [18005, 13, 405], [18005, 13, 405, 198], [18005, 13, 405, 198, 198], [18005, 13, 405, 198, 198, 464], [1954, 13], [1954, 13, 198], [1954, 13, 198, 198], [1954, 13, 198, 198, 464], [1954, 13, 198, 198, 464, 1708], [1828, 13], [1828, 13, 15], [1828, 13, 15, 13], [1828, 13, 15, 13, 15], [1828, 13, 15, 13, 15, 14], [198, 198], [198, 198, 464], [198, 198, 464, 1708], [198, 198, 464, 1708, 318], [198, 198, 464, 1708, 318, 257], [198, 198], [198, 198, 464], [198, 198, 464, 1708], [198, 198, 464, 1708, 318], [198, 198, 464, 1708, 318, 257], [198, 198], [198, 198, 464], [198, 198, 464, 1708], [198, 198, 464, 1708, 318], [198, 198, 464, 1708, 318, 257]]
        # Extended prompts prepared. Their shape is 160 x 2

        print(extended_prompts) # --> outputs 5*32 , 5 = 160, 5 
        print(f'Extended prompts prepared. Their shape is {len(extended_prompts)} x {len(extended_prompts[0])}')  # Debug statement

        # run teacher inference once
        _, probs_seq, _data_copy_time, _softmax_time = token_predictor_teacher.run_batched_inference(
            extended_prompts,
            enable_kv_cache=args.use_kv_cache)

        print(f'probs_seq shape: {probs_seq.shape}') # Should be [B*spec_k, V]
        # probs_seq shape: torch.Size([160, 357]) 

        data_copy_time += _data_copy_time
        softmax_time += _softmax_time
        inference_time += time.perf_counter() - t0_inference

        # STEP 3: Verify and accept/reject
        t0_ac = time.perf_counter()

        # convert draft_tokens_batch to torch tensor
        draft_tokens_batch = torch.tensor(draft_tokens_batch)
        print(f'draft tokens batch: ', draft_tokens_batch.shape)
        # draft tokens batch:  torch.Size([32, 5])

        # Iterate over batches
        for batch_idx in tqdm(range(len(prompts)), desc="Processing prompts"):
            draft_tokens = draft_tokens_batch[batch_idx] # list object of len 5

            print(f'draft token shape : {draft_tokens.shape}')
            probs_steps = probs_seq[batch_idx]
            
            print(f'probs_steps shape: {probs_steps.shape}')  # Is [spec_k, V], should be (357,)
            # probs_steps shape: torch.Size([357])
            # for each element in batch, this is the probabilities predicted 

            probs_steps = probs_seq[batch_idx * args.spec_k:(batch_idx + 1) * args.spec_k]
            print(f'new shape probs_steps {probs_steps.shape}')

            accepted = 0 # length of accepted speculative tokens

            # Decompressing tokens. Verification of speculative tokens, 
            for step, probs_tensor in enumerate(probs_steps):
                # Convert to CPU numpy array
                probs = probs_tensor.detach().cpu().numpy().astype(np.float64)

                # Clip negatives
                # probs = np.clip(probs, 0, None)

                decoded_idx = decompressor.decompress(probs)
                draft_token = draft_tokens[step]

                if decoded_idx == draft_token:
                    accepted += 1
                else:
                    print('rejected')
                    # mismatch → fallback token
                    next_token = token_predictor_teacher.get_token_by_id(decoded_idx)
                    prompts[batch_idx].append(next_token)
                    reconstructed_tokens[batch_idx].append(next_token)
                    break

            if accepted > 0:
                accepted_tokens = draft_tokens[:accepted]
                prompts[batch_idx].extend(accepted_tokens)
                reconstructed_tokens[batch_idx].extend(accepted_tokens)

        ac_time += time.perf_counter() - t0_ac

    reconstructed_tokens = list(chain.from_iterable(reconstructed_tokens))

    t0_detokenize = time.perf_counter()
    detoken_string = token_predictor_teacher.detokenize(reconstructed_tokens)
    detokenize_time = time.perf_counter() - t0_detokenize

    decompression_time = time.perf_counter() - t0_decompress


    return reconstructed_tokens, detoken_string, {
        "args": args.__dict__,
        "decompression_time_sec": decompression_time,
        "input_tokens_cnt": input_tokens_cnt,
        "total_decompression_time": decompression_time,
        "detokenize_time": detokenize_time,
        "inference_time": inference_time,
        "ac_time": ac_time,
        "data_copy_time": data_copy_time,
        "softmax_time": softmax_time,
        "throughput_kibibytes_per_sec": len(detoken_string) / 1024 / decompression_time,
        "inference_throughput_kibibytes_per_sec": len(detoken_string) / 1024 / inference_time,
    }





def _speculative_batch_lengths(args):
    chunk_length = args.first_n_tokens // args.batch_size
    extra = args.first_n_tokens % args.batch_size
    batch_lengths = [
        chunk_length + (1 if batch_index < extra else 0)
        for batch_index in range(args.batch_size)
    ]
    return chunk_length, batch_lengths


def _trim_active_prompts(prompts, active_indices, retain_tokens, context_length):
    active_prompts = [prompts[index] for index in active_indices]
    if active_prompts and len(active_prompts[0]) >= context_length:
        for index in active_indices:
            prompts[index] = prompts[index][-retain_tokens:]
        active_prompts = [prompts[index] for index in active_indices]
    return active_prompts


def _build_teacher_prompts(active_prompts, draft_tokens_batch):
    teacher_prompts = []
    for prompt, draft_info in zip(active_prompts, draft_tokens_batch):
        draft_tokens = draft_info[0]
        current_prompt = list(prompt)
        teacher_prompts.append(current_prompt.copy())
        for token in draft_tokens:
            current_prompt.append(int(token))
            teacher_prompts.append(current_prompt.copy())
    return teacher_prompts


def _decode_teacher_distribution(distribution, decompressor):
    probs = distribution.detach().cpu().numpy().astype(np.float64)
    return int(decompressor.decompress(probs))


def _apply_verified_tokens(prompts, reconstructed_tokens, batch_idx, accepted_tokens):
    if accepted_tokens:
        prompts[batch_idx].extend(accepted_tokens)
        reconstructed_tokens[batch_idx].extend(accepted_tokens)


def run_global_mask_speculative_decompression_new(args, first_tokens, bit_string, bitmap):
    """Run speculative decompression using draft tokens plus teacher verification."""
    if not first_tokens:
        raise ValueError("first_tokens cannot be empty")

    # Keep the historical fixed draft length because the current tests and calling
    # code rely on it, even when args.spec_k is set differently by the caller.
    args.spec_k = 5

    teacher = TokenPredictor(args, bitmap_data=bitmap)
    student = TokenPredictor(args, bitmap_data=bitmap)
    decompressor = LLMDecompressor(bit_string)

    batch_size = len(first_tokens)
    prompts = [[int(token)] for token in first_tokens]
    reconstructed_tokens = [[int(token)] for token in first_tokens]
    _, batch_lengths = _speculative_batch_lengths(args)

    input_tokens_cnt = 0
    inference_time = 0.0
    ac_time = 0.0
    data_copy_time = 0.0
    softmax_time = 0.0
    start_time = time.perf_counter()

    while True:
        active_indices = [
            batch_idx
            for batch_idx in range(batch_size)
            if len(reconstructed_tokens[batch_idx]) < batch_lengths[batch_idx]
        ]
        if not active_indices:
            break

        active_prompts = _trim_active_prompts(
            prompts,
            active_indices,
            args.retain_tokens,
            args.context_length,
        )
        input_tokens_cnt += sum(len(prompts[batch_idx]) for batch_idx in active_indices)

        inference_start = time.perf_counter()
        draft_tokens_batch = [
            student.generate_draft(prompts[batch_idx], k=args.spec_k, enable_kv_cache=True)
            for batch_idx in active_indices
        ]
        teacher_prompts = _build_teacher_prompts(active_prompts, draft_tokens_batch)
        _, teacher_scores, copy_time, softmax_step_time = teacher.run_batched_inference(
            teacher_prompts,
            enable_kv_cache=args.use_kv_cache,
        )
        inference_time += time.perf_counter() - inference_start
        data_copy_time += copy_time
        softmax_time += softmax_step_time

        ac_start = time.perf_counter()
        group_size = args.spec_k + 1

        for local_idx, batch_idx in enumerate(active_indices):
            remaining = batch_lengths[batch_idx] - len(reconstructed_tokens[batch_idx])
            if remaining <= 0:
                continue

            draft_tokens = [int(token) for token in draft_tokens_batch[local_idx][0]]
            start = local_idx * group_size
            probs_steps = teacher_scores[start : start + group_size]
            accepted_tokens = []
            verify_count = min(args.spec_k, remaining)
            mismatch_token = None

            for step in range(verify_count):
                decoded_token = _decode_teacher_distribution(probs_steps[step], decompressor)
                if decoded_token == draft_tokens[step]:
                    accepted_tokens.append(decoded_token)
                    continue

                mismatch_token = decoded_token
                break

            _apply_verified_tokens(prompts, reconstructed_tokens, batch_idx, accepted_tokens)

            if mismatch_token is not None:
                prompts[batch_idx].append(mismatch_token)
                reconstructed_tokens[batch_idx].append(mismatch_token)
                continue

            remaining = batch_lengths[batch_idx] - len(reconstructed_tokens[batch_idx])
            if verify_count == args.spec_k and remaining > 0:
                extra_token = _decode_teacher_distribution(probs_steps[args.spec_k], decompressor)
                prompts[batch_idx].append(extra_token)
                reconstructed_tokens[batch_idx].append(extra_token)

        ac_time += time.perf_counter() - ac_start

    reconstructed_tokens_flat = list(chain.from_iterable(reconstructed_tokens))
    detokenize_start = time.perf_counter()
    detoken_string = teacher.detokenize(reconstructed_tokens_flat)
    detokenize_time = time.perf_counter() - detokenize_start
    decompression_time = time.perf_counter() - start_time

    return reconstructed_tokens_flat, detoken_string, {
        "args": args.__dict__,
        "decompression_time_sec": decompression_time,
        "input_tokens_cnt": input_tokens_cnt,
        "total_decompression_time": decompression_time,
        "detokenize_time": detokenize_time,
        "inference_time": inference_time,
        "ac_time": ac_time,
        "data_copy_time": data_copy_time,
        "softmax_time": softmax_time,
        "throughput_kibibytes_per_sec": len(detoken_string) / 1024 / decompression_time,
        "inference_throughput_kibibytes_per_sec": len(detoken_string) / 1024 / max(inference_time, 1e-12),
    }

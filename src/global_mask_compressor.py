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



def run_global_mask_speculative_decompression_old(args, first_tokens, bit_string, bitmap):
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
        draft_tokens_batch = [
            list(token_predictor_student.generate_draft(prompts[i], k=args.spec_k))
            for i in active_indices]
        
        #draft_tokens_batch = [
        #    list(token_predictor_student.generate_draft(prompts[i], k=args.spec_k)[0])
        #    for i in active_indices] # 192 = 32 * 6 

        #draft_tokens_batch = [list(token_predictor_student.run_batched_inference(prompts[i], enable_kv_cache=False))
        #                        for i in active_indices]
        
            
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


def run_global_mask_speculative_decompression_v1(args, first_tokens, bit_string, bitmap):
    """
    Run speculative decompression using:
      - TokenPredictor.generate_draft() for student drafting
      - TokenPredictor.run_batched_inference() for teacher verification

    This version fixes the token-space bug in the original implementation:
    decompressor outputs a column index into the returned score tensor, which
    must be mapped through tokens_list to recover the actual token id.
    """

    if not first_tokens:
        raise ValueError("first_tokens cannot be empty")

    if getattr(args, "spec_k", None) is None:
        args.spec_k = 5

    # Initialize predictors
    token_predictor_teacher = TokenPredictor(args, bitmap_data=bitmap)
    token_predictor_student = TokenPredictor(args, bitmap_data=bitmap)

    decompressor = LLMDecompressor(bit_string)

    batch_size = len(first_tokens)

    # Prompts used as model context
    prompts = [[tok] for tok in first_tokens]

    # Output tokens reconstructed for each batch item
    reconstructed_tokens = [[tok] for tok in first_tokens]

    # Distribute target token count across the *actual* batch size
    chunk_length = args.first_n_tokens // batch_size
    extra = args.first_n_tokens % batch_size
    target_lengths = [
        chunk_length + (1 if i < extra else 0)
        for i in range(batch_size)
    ]

    input_tokens_cnt = 0
    inference_time = 0.0
    ac_time = 0.0
    data_copy_time = 0.0
    softmax_time = 0.0

    t0_decompress = time.perf_counter()

    while any(len(reconstructed_tokens[i]) < target_lengths[i] for i in range(batch_size)):
        active_indices = [
            i for i in range(batch_size)
            if len(reconstructed_tokens[i]) < target_lengths[i]
        ]
        if not active_indices:
            break

        # Trim long contexts if needed
        active_prompts = []
        for i in active_indices:
            prompt_i = prompts[i]
            if len(prompt_i) >= args.context_length:
                prompt_i = prompt_i[-args.retain_tokens:]
                prompts[i] = prompt_i
            active_prompts.append(prompt_i[:])

        remaining_per_active = [
            target_lengths[i] - len(reconstructed_tokens[i])
            for i in active_indices
        ]

        # Do not draft more than the maximum remaining tokens needed by any active sequence
        current_k = min(args.spec_k, max(remaining_per_active))
        if current_k <= 0:
            break

        input_tokens_cnt += sum(len(p) for p in active_prompts)

        # ------------------------------------------------------------------
        # STEP 1: student draft generation
        # ------------------------------------------------------------------
        t0_inference = time.perf_counter()

        (
            draft_tokens_batch,
            _draft_scores,
            _draft_data_copy_time,
            _draft_softmax_time,
        ) = token_predictor_student.generate_draft(
            active_prompts,
            k=current_k,
            enable_kv_cache=False,
            full_draft=False,
        )

        data_copy_time += _draft_data_copy_time
        softmax_time += _draft_softmax_time

        # ------------------------------------------------------------------
        # STEP 2: build teacher verification prompts
        #
        # For each active prompt p and draft d1..dk, build:
        #   p
        #   p+d1
        #   p+d1+d2
        #   ...
        #   p+d1+...+dk
        #
        # That gives k+1 verifier distributions per active sequence:
        #   step 0 verifies d1
        #   step 1 verifies d2
        #   ...
        #   step k-1 verifies dk
        #   step k gives one extra next-token distribution if all drafts accepted
        # ------------------------------------------------------------------
        extended_prompts = []
        for prompt_i, draft_i in zip(active_prompts, draft_tokens_batch):
            cur = prompt_i[:]
            extended_prompts.append(cur[:])  # verifies first drafted token
            for drafted_tok in draft_i:
                cur = cur + [int(drafted_tok)]
                extended_prompts.append(cur[:])

        teacher_tokens_list, probs_seq, _teacher_data_copy_time, _teacher_softmax_time = (
            token_predictor_teacher.run_batched_inference(
                extended_prompts,
                enable_kv_cache=False,
            )
        )

        data_copy_time += _teacher_data_copy_time
        softmax_time += _teacher_softmax_time
        inference_time += time.perf_counter() - t0_inference

        # ------------------------------------------------------------------
        # STEP 3: arithmetic-coding accept/reject
        # ------------------------------------------------------------------
        t0_ac = time.perf_counter()

        group_size = current_k + 1

        for local_idx, batch_idx in enumerate(active_indices):
            remaining = target_lengths[batch_idx] - len(reconstructed_tokens[batch_idx])
            if remaining <= 0:
                continue

            draft_i = [int(tok) for tok in draft_tokens_batch[local_idx]]

            start = local_idx * group_size
            end = start + group_size
            probs_steps = probs_seq[start:end]  # shape: (k+1, reduced_vocab) or list of rows

            accepted_tokens = []
            mismatch = False

            # Only verify as many drafted tokens as we still need
            verify_count = min(len(draft_i), remaining)

            for step in range(verify_count):
                probs = probs_steps[step].detach().cpu().numpy().astype(np.float64)

                # decompressor returns a column index in probs
                decoded_col = int(decompressor.decompress(probs))
                decoded_token = teacher_tokens_list[decoded_col]

                drafted_token = draft_i[step]

                if decoded_token == drafted_token:
                    accepted_tokens.append(drafted_token)
                else:
                    # accept prefix, then append verifier-decoded fallback token
                    prompts[batch_idx].extend(accepted_tokens)
                    reconstructed_tokens[batch_idx].extend(accepted_tokens)

                    prompts[batch_idx].append(decoded_token)
                    reconstructed_tokens[batch_idx].append(decoded_token)

                    mismatch = True
                    break

            if mismatch:
                continue

            # All verified draft tokens accepted
            prompts[batch_idx].extend(accepted_tokens)
            reconstructed_tokens[batch_idx].extend(accepted_tokens)

            remaining = target_lengths[batch_idx] - len(reconstructed_tokens[batch_idx])

            # If all drafted tokens were accepted and we still need one more token,
            # consume one extra verifier distribution.
            if verify_count == len(draft_i) and remaining > 0:
                probs = probs_steps[len(draft_i)].detach().cpu().numpy().astype(np.float64)
                decoded_col = int(decompressor.decompress(probs))
                decoded_token = teacher_tokens_list[decoded_col]

                prompts[batch_idx].append(decoded_token)
                reconstructed_tokens[batch_idx].append(decoded_token)

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
        "throughput_kibibytes_per_sec": len(detoken_string) / 1024 / max(decompression_time, 1e-12),
        "inference_throughput_kibibytes_per_sec": len(detoken_string) / 1024 / max(inference_time, 1e-12),
    }


from itertools import chain
import time
import torch


def run_global_mask_speculative_decompression(
    args,
    first_tokens,
    bit_string,
    bitmap,
):
    """
    Speculative decompression for a global-token-mask AC stream.

    Draft tokens are proposed by a draft model, but the arithmetic decoder
    always consumes probabilities from the verifier model. A drafted token is
    accepted only if it matches the token recovered from the verifier
    distribution. On the first mismatch for a batch row, the verifier-decoded
    token is appended and the remaining drafted suffix for that row is dropped.

    Returns:
        tuple: (reconstructed_tokens, detoken_string, stats)
    """
    print(
        f"\n----- Running Speculative Decompression: "
        f"Global Token Mask (first_n_tokens={args.first_n_tokens}, "
        f"spec_k={args.spec_k}, verifier_kv_cache={args.use_kv_cache}) -----"
    )

    t0_decompress = time.perf_counter()

    # Verifier model: source of truth for AC decoding
    verifier = TokenPredictor(args, bitmap_data=bitmap)

    # Draft model: may be smaller/faster; for now can be same config
    # If you add args.draft_model_name, clone args and replace model_name here.
    draft_args = args
    draft = TokenPredictor(draft_args, bitmap_data=bitmap)

    decompressor = LLMDecompressor(bit_string)

    if not first_tokens:
        raise ValueError("first_tokens must contain one initial token per batch row")

    if len(first_tokens) != args.batch_size:
        raise ValueError(
            f"Expected {args.batch_size} first tokens, got {len(first_tokens)}"
        )

    prompts = [[first_tokens[i]] for i in range(args.batch_size)]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size)]

    # Per-batch target lengths
    chunk_length = args.first_n_tokens // args.batch_size
    extra = args.first_n_tokens % args.batch_size
    batches_length = [
        chunk_length + (1 if i < extra else 0)
        for i in range(args.batch_size)
    ]

    total_decoded = len(first_tokens)
    max_tokens = args.first_n_tokens

    input_tokens_cnt = 0
    inference_time_verifier = 0.0
    inference_time_draft = 0.0
    ac_time = 0.0
    data_copy_time_verifier = 0.0
    softmax_time_verifier = 0.0
    data_copy_time_draft = 0.0
    softmax_time_draft = 0.0

    accepted_draft_tokens = 0
    rejected_draft_tokens = 0
    verifier_steps = 0
    draft_cycles = 0

    while total_decoded < max_tokens:
        # Active rows still needing tokens
        active_rows = [
            i for i in range(args.batch_size)
            if len(reconstructed_tokens[i]) < batches_length[i]
        ]
        if not active_rows:
            break

        active_prompts = [prompts[i] for i in active_rows]

        # Truncate context if needed, but preserve reconstructed output separately
        if len(active_prompts[0]) >= args.context_length:
            for i in active_rows:
                prompts[i] = prompts[i][-args.retain_tokens:]
            active_prompts = [prompts[i] for i in active_rows]

            # KV cache is no longer valid after context truncation
            verifier.reset_kv_cache()
            draft.reset_kv_cache()

        remaining_per_row = [
            batches_length[i] - len(reconstructed_tokens[i])
            for i in active_rows
        ]
        current_k = min(args.spec_k, max(remaining_per_row))
        if current_k <= 0:
            break

        print(
            f"\rProcessing speculative cycle, active_rows={len(active_rows)}, k={current_k}",
            end=""
        )

        # ------------------------------------------------------------
        # 1) Draft phase
        # ------------------------------------------------------------
        t0_draft = time.perf_counter()
        _, draft_scores, _dcp_draft, _sm_draft, drafted_prompts = draft.generate_draft(
            [row[:] for row in active_prompts],
            k=current_k,
            enable_kv_cache=False,
            full_draft=True,
        )
        inference_time_draft += time.perf_counter() - t0_draft
        data_copy_time_draft += _dcp_draft
        softmax_time_draft += _sm_draft
        draft_cycles += 1

        # Recover drafted token lists from drafted_prompts delta
        drafted_tokens = []
        for base, full in zip(active_prompts, drafted_prompts):
            drafted_tokens.append(full[len(base):])

        # ------------------------------------------------------------
        # 2) Verifier + AC decode phase
        #    Each step uses verifier probs as source of truth.
        # ------------------------------------------------------------
        # We keep local working prompts for active rows during this cycle.
        local_prompts = [row[:] for row in active_prompts]
        local_done = [False] * len(active_rows)

        # Reset verifier cache and prefill from local prompts.
        verifier.reset_kv_cache()

        for step_idx in range(current_k):
            step_rows = [
                r for r in range(len(active_rows))
                if (not local_done[r])
                and (len(reconstructed_tokens[active_rows[r]]) < batches_length[active_rows[r]])
            ]
            if not step_rows:
                break

            step_prompts = [local_prompts[r] for r in step_rows]

            input_tokens_cnt += sum(len(p) for p in step_prompts)

            t0_ver = time.perf_counter()
            _, probs_values, _dcp_ver, _sm_ver = verifier.run_batched_inference(
                step_prompts,
                enable_kv_cache=args.use_kv_cache,
            )
            inference_time_verifier += time.perf_counter() - t0_ver
            data_copy_time_verifier += _dcp_ver
            softmax_time_verifier += _sm_ver
            verifier_steps += 1

            t0_ac = time.perf_counter()

            probs_np = probs_values.to(torch.float32).cpu().numpy()

            for j, r in enumerate(step_rows):
                global_row = active_rows[r]

                if len(reconstructed_tokens[global_row]) >= batches_length[global_row]:
                    local_done[r] = True
                    continue

                decoded_idx = decompressor.decompress(probs_np[j])
                decoded_token = verifier.get_token_by_id(decoded_idx)

                drafted_token = drafted_tokens[r][step_idx]

                # Source of truth is decoded_token from verifier distribution.
                prompts[global_row].append(decoded_token)
                reconstructed_tokens[global_row].append(decoded_token)
                local_prompts[r].append(decoded_token)
                total_decoded += 1

                if decoded_token == drafted_token:
                    accepted_draft_tokens += 1
                    # keep verifying next drafted token
                else:
                    rejected_draft_tokens += max(0, current_k - step_idx)
                    local_done[r] = True

                if total_decoded >= max_tokens:
                    break

            ac_time += time.perf_counter() - t0_ac

            if total_decoded >= max_tokens:
                break

        # Draft cache is not reused across cycles in this simple version
        draft.reset_kv_cache()

    print()

    flat_tokens = list(chain.from_iterable(reconstructed_tokens))

    t0_detok = time.perf_counter()
    detoken_string = verifier.detokenize(flat_tokens)
    detokenize_time = time.perf_counter() - t0_detok

    decompression_time = time.perf_counter() - t0_decompress

    return flat_tokens, detoken_string, {
        "args": args.__dict__,
        "decompression_time_sec": decompression_time,
        "input_tokens_cnt": input_tokens_cnt,
        "total_decompression_time": decompression_time,
        "detokenize_time": detokenize_time,
        "verifier_inference_time": inference_time_verifier,
        "draft_inference_time": inference_time_draft,
        "ac_time": ac_time,
        "verifier_data_copy_time": data_copy_time_verifier,
        "verifier_softmax_time": softmax_time_verifier,
        "draft_data_copy_time": data_copy_time_draft,
        "draft_softmax_time": softmax_time_draft,
        "verifier_steps": verifier_steps,
        "draft_cycles": draft_cycles,
        "accepted_draft_tokens": accepted_draft_tokens,
        "rejected_draft_tokens": rejected_draft_tokens,
        "acceptance_rate": (
            accepted_draft_tokens / max(1, accepted_draft_tokens + rejected_draft_tokens)
        ),
        "throughput_kibibytes_per_sec": len(detoken_string) / 1024 / decompression_time,
        "verifier_inference_throughput_kibibytes_per_sec": (
            len(detoken_string) / 1024 / max(inference_time_verifier, 1e-12)
        ),
    }
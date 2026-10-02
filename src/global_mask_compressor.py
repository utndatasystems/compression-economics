"""
Global-mask LLM compression/decompression.

This module implements a compression experiment that uses a single global bitmap of
allowed tokens to reduce the LLM prediction space. Tokens are processed in batches
to amortize inference, and the resulting probability distributions are encoded via
arithmetic coding ("AC" or independent "AC_MULTISTREAM" states) or rank-based
schemes ("bitpacked", "huffman") for the transformer engine.

Decompression mirrors compression using the same model and
bitmap to reconstruct tokens from the bitstream.
"""

import sys

from src.coding.encoding import LLMCompressor, LLMDecompressor, choose_pmatic_r
from src.coding.cuda_ac import encode_intervals_cuda
from src.coding.multistream_ac import MultistreamACDecoder, MultistreamACEncoder
from src.coding.device_ac import (
    CudaCDFMultistreamACDecoder,
    DeviceMultistreamACDecoder,
)
from src.coding.paired_ac import PairedACDecoder, PairedACEncoder, pack_standard_archive
from src.prediction import NGramTokenPredictor, TokenDataPreparer, TokenPredictor
from itertools import chain
from collections import defaultdict

import numpy as np
import time
import torch
from tqdm.auto import tqdm
from wandb.plot import line
from itertools import chain
from copy import copy

PMATIC_DELTA = 1e-3


def _rate(amount, seconds):
    """Report zero when no model inference was needed."""
    return amount / seconds if seconds else 0.0


def _make_token_predictor(args, bitmap_data):
    """Select the Transformer, n-gram, or optional vLLM predictor adapter."""
    if args.engine == "ngram":
        return NGramTokenPredictor(args, bitmap_data)
    if args.engine == "vllm":
        from src.vllm_prediction import VLLMTokenPredictor
        return VLLMTokenPredictor(args, bitmap_data)
    return TokenPredictor(args, bitmap_data)


def _get_pmatic_params(args):
    delta = getattr(args, "pmatic_delta", None)
    if delta is None:
        delta = PMATIC_DELTA
    r = getattr(args, "pmatic_r", None)
    if r is None:
        r = choose_pmatic_r(delta)
        args.pmatic_r = r
    args.pmatic_delta = delta
    return delta, r


def _make_arithmetic_decompressor(args, bit_string, alphabet_size):
    if args.encoding in {"AC", "ANS"}:
        return LLMDecompressor(bit_string, algorithm=args.encoding,
                               frequency_quantizer=getattr(args, "frequency_quantizer", "reference"))

    if args.encoding == "PMATIC":
        delta, r = _get_pmatic_params(args)
        return LLMDecompressor(
            bit_string,
            algorithm="PMATIC",
            frequency_quantizer=getattr(args, "frequency_quantizer", "reference"),
            alphabet_size=alphabet_size,
            delta=delta,
            r=r,
        )

    raise NotImplementedError(
        f"Encoding method '{args.encoding}' is not implemented for decompression."
    )


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
            encoding (str): "AC", "AC_MULTISTREAM", "ANS", "PMATIC", or a rank coder.

    Returns:
        tuple: (first_tokens, bit_string, bitmask_data, stats, args)
            - first_tokens (list[int]): First token from each batch (seed for decoding).
            - bit_string (list[int] | bytes): Coded bits or the MSAC payload.
            - bitmask_data (bytes): Serialized bitmap describing allowed vocabulary.
            - stats (dict): Compression statistics (sizes, timings, throughput).
            - args (argparse.Namespace): Possibly updated args from token preparation.
    """
    print(f"\n----- Running Compression: Global Token Mask (tokens={args.first_n_tokens}, kv_cache={args.use_kv_cache}) -----")
    use_tqdm = sys.stderr.isatty()

    t0_tokenize = time.perf_counter()
    input_token_cnt = 0

    # Initialize the token predictor to get tokens and probabilities from the model.
    token_data_preparer = TokenDataPreparer(args)
    data_tokens = token_data_preparer.get_data_tokens()
    args = token_data_preparer.get_args()
    if args.first_n_tokens is None:
        args.first_n_tokens = len(data_tokens)


    ac_backend = getattr(args, "ac_backend", "python")
    cuda_interval_backend = (
        args.encoding == "AC_TARGET_INTERVAL" and ac_backend == "cuda"
    )
    target_interval_quantizer = getattr(
        args, "target_interval_quantizer", "torch"
    )
    if ac_backend == "cuda" and (
        args.encoding != "AC_TARGET_INTERVAL" or args.engine != "transformer"
    ):
        raise ValueError(
            "CUDA AC encoding requires Transformer and AC_TARGET_INTERVAL"
        )
    if cuda_interval_backend and not torch.cuda.is_available():
        raise ValueError("CUDA AC encoding requires an available CUDA device")
    if target_interval_quantizer == "cuda_fused" and not cuda_interval_backend:
        raise ValueError(
            "fused CUDA target intervals require the CUDA AC interval backend"
        )

    if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"} and not 1 <= args.batch_size <= len(data_tokens):
        raise ValueError("AC_MULTISTREAM needs one seed token per nonempty batch")

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

    token_predictor = _make_token_predictor(args, bitmask_data)
    try:

        if args.encoding in {"AC", "ANS"}:
            llm_compressor = LLMCompressor(algorithm=args.encoding,
                                           frequency_quantizer=getattr(args, "frequency_quantizer", "reference"))
        elif args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}:
            if cuda_interval_backend:
                llm_compressor = None
            else:
                encoder_type = (
                    PairedACEncoder
                    if getattr(args, "ac_layout", "standard") == "paired"
                    else MultistreamACEncoder
                )
                llm_compressor = encoder_type(
                    args.batch_size,
                    backend=ac_backend,
                    threads=getattr(args, "ac_threads", None),
                    frequency_quantizer=getattr(
                        args, "frequency_quantizer", "reference"
                    ),
                    target_interval=args.encoding == "AC_TARGET_INTERVAL",
                )
        elif args.encoding == "PMATIC":
            delta, r = _get_pmatic_params(args)
            print(f"Using PMATIC compressor with delta={delta}, r={r}")
            alphabet_size = len(token_predictor.tokens_list)
            # alternatively alphabet_size = token_predictor.tokenizer.vocab_size
            llm_compressor = LLMCompressor(
                alphabet_size=alphabet_size,
                delta=delta,
                r=r,
                algorithm="PMATIC",
            )

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
        prediction_steps = max(batches_length) - 1
        if cuda_interval_backend:
            interval_shape = (prediction_steps, args.batch_size)
            interval_device = token_predictor.device
            cuda_interval_lows = torch.empty(
                interval_shape, dtype=torch.int64, device=interval_device
            )
            cuda_interval_highs = torch.empty_like(cuda_interval_lows)
            cuda_interval_totals = torch.empty_like(cuda_interval_lows)
            cuda_target_probs = torch.empty(
                interval_shape, dtype=torch.float64, device=interval_device
            )
        else:
            cuda_interval_lows = None
            cuda_interval_highs = None
            cuda_interval_totals = None
            cuda_target_probs = None
        cuda_inference_events = []
        cuda_interval_events = []
        cuda_ac_metrics = {}
        # Process each token in the dataset to compress it.
        # Only run steps that have at least one next symbol. When all batches have
        # equal length, the old ``range(chunk_length)`` performed one useless final
        # model call and inflated the model-work throughput counter.
        interval_transfer_bytes = 0
        interval_seconds = 0.0
        token_columns = {token: index for index, token in enumerate(token_predictor.tokens_list)}
        for token_idx in tqdm(range(prediction_steps), disable=not use_tqdm):
            # Append the current token from each batch to its prompt context.
            for i in range(args.batch_size):
                prompts[i].append(batches[i][token_idx])

            # Trim context to keep inference cost bounded.
            if len(prompts[0]) >= args.context_length:
                prompts = [prompt[-args.retain_tokens:] for prompt in prompts]

            actual_next_tokens = []
            valid_mask = []

            # Build a mask for batches that still have a "next token" at this step.
            for idx in range(args.batch_size):
                if token_idx + 1 < batches_length[idx]:
                    token = batches[idx][token_idx + 1]
                    actual_next_tokens.append(token_columns[token])
                    valid_mask.append(True)
                else:
                    actual_next_tokens.append(0)
                    valid_mask.append(False)

            # Run LLM inference. CUDA events preserve asynchronous execution
            # while avoiding a host synchronization at every prediction step.
            t0_inference = time.perf_counter()
            if cuda_interval_backend:
                inference_start = torch.cuda.Event(enable_timing=True)
                inference_end = torch.cuda.Event(enable_timing=True)
                inference_start.record()
            if args.encoding == "AC_TARGET_INTERVAL":
                targets = [batches[i][token_idx + 1] if valid_mask[i] else token_predictor.tokens_list[0]
                           for i in range(args.batch_size)]
                token_ids, probs_values, _data_copy_time, _softmax_time = token_predictor.run_batched_interval_inference(
                    prompts, targets, args.use_kv_cache)
                interval_transfer_bytes += getattr(token_predictor, "last_interval_transfer_bytes", 32 * len(prompts))
                interval_seconds += getattr(token_predictor, "last_interval_seconds", 0.0)
                interval_event = getattr(token_predictor, "last_interval_event", None)
                if cuda_interval_backend and interval_event is not None:
                    cuda_interval_events.append(interval_event)
            else:
                token_ids, probs_values, _data_copy_time, _softmax_time = token_predictor.run_batched_inference(prompts, args.use_kv_cache)
            input_token_cnt += getattr(
                token_predictor,
                "last_model_input_tokens",
                sum(len(prompt) for prompt in prompts),
            )
            data_copy_time += _data_copy_time
            softmax_time += _softmax_time
            if cuda_interval_backend:
                inference_end.record()
                cuda_inference_events.append((inference_start, inference_end))
            else:
                inference_time += time.perf_counter() - t0_inference

            if args.engine in {"transformer", "ngram", "vllm"}:
                if args.encoding == "AC_TARGET_INTERVAL":
                    if cuda_interval_backend:
                        for name in ("lows", "highs", "totals", "target_probs"):
                            if probs_values[name].device.type != "cuda":
                                raise ValueError(
                                    "CUDA interval backend received host interval data"
                                )
                        cuda_interval_lows[token_idx].copy_(probs_values["lows"])
                        cuda_interval_highs[token_idx].copy_(probs_values["highs"])
                        cuda_interval_totals[token_idx].copy_(probs_values["totals"])
                        active = torch.as_tensor(
                            valid_mask,
                            dtype=torch.bool,
                            device=probs_values["target_probs"].device,
                        )
                        cuda_target_probs[token_idx].copy_(torch.where(
                            active,
                            probs_values["target_probs"],
                            torch.ones_like(probs_values["target_probs"]),
                        ))
                    else:
                        started = time.perf_counter()
                        records = zip(
                            probs_values["lows"].tolist(),
                            probs_values["highs"].tolist(),
                            probs_values["totals"].tolist(),
                            probs_values["target_probs"].tolist(),
                        )
                        for idx, (low, high, total, probability) in enumerate(records):
                            if valid_mask[idx]:
                                llm_compressor.encode_interval(
                                    idx, low, high, total
                                )
                                entropy += -np.log2(max(probability, 1e-300))
                                probs_list.append(probability)
                        ac_time += time.perf_counter() - started
                elif args.encoding in {"AC", "ANS", "AC_MULTISTREAM"}:
                    t0_ac = time.perf_counter()
                    probs_cpu = probs_values.to(torch.float32).numpy()  # [B, V]

                    # Encode each valid batch's next token using arithmetic coding.
                    for idx, probs in enumerate(probs_cpu):
                        if not valid_mask[idx]:
                            continue
                        target_idx = actual_next_tokens[idx]
                        if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}:
                            llm_compressor.encode(idx, target_idx, probs)
                        else:
                            llm_compressor.next_token(target_idx, probs)
                        entropy += -np.log2(probs[target_idx])
                        probs_list.append(probs[target_idx])

                    ac_time += time.perf_counter() - t0_ac

                elif args.encoding == "PMATIC":
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

        if args.engine in {"transformer", "ngram", "vllm"}:
            if args.encoding in {"AC", "ANS"}:
                bit_string = llm_compressor.compress(encoding=args.encoding)
            elif args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}:
                if cuda_interval_backend:
                    device = token_predictor.device
                    lows = cuda_interval_lows
                    highs = cuda_interval_highs
                    totals = cuda_interval_totals
                    target_probabilities = cuda_target_probs
                    counts = torch.tensor(
                        [max(0, length - 1) for length in batches_length],
                        dtype=torch.int64,
                        device=device,
                    )
                    # One controlled boundary replaces one synchronization per
                    # prediction step and makes inference/coder timings separable.
                    torch.cuda.synchronize(device)
                    inference_time = sum(
                        start.elapsed_time(end) / 1000.0
                        for start, end in cuda_inference_events
                    )
                    interval_device_seconds = sum(
                        start.elapsed_time(end) / 1000.0
                        for start, end in cuda_interval_events
                    )
                    started = time.perf_counter()
                    entropy = float((
                        -torch.log2(target_probabilities.clamp_min(1e-300))
                    ).sum().item())
                    bit_string = encode_intervals_cuda(
                        lows,
                        highs,
                        totals,
                        counts,
                        metrics=cuda_ac_metrics,
                    )
                    if getattr(args, "ac_layout", "standard") == "paired":
                        bit_string = pack_standard_archive(bit_string)
                        cuda_ac_metrics["paired_archive_bytes"] = len(bit_string)
                    ac_time += time.perf_counter() - started
                else:
                    bit_string = llm_compressor.finish()
            elif args.encoding == "PMATIC":
                bit_string = llm_compressor.compress(encoding="PMATIC")
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
        total_arithmetic_code_size = len(bit_string) * (8 if isinstance(bit_string, bytes) else 1)
        if args.encoding == "huffman":
            # Estimate the size of the codebook in bits
            codebook_size = sum(len(code) for code in codebook.values())
            total_arithmetic_code_size += codebook_size
            print(f"Estimated codebook size (bits): {codebook_size}")

        # Calculate final size and compression ratio.
        final_size = total_arithmetic_code_size + total_bitmap_size
        original_text = token_predictor.detokenize(data_tokens)
        original_size_bytes = len(original_text.encode("utf-8"))

        return first_tokens, bit_string, bitmask_data, {
            "base_params": getattr(token_predictor, "base_params", 0),
            "adapter_params": getattr(token_predictor, "adapter_params", 0),
            "base_size_mb": getattr(token_predictor, "base_size_mb", 0.0),
            "adapter_size_mb": getattr(token_predictor, "adapter_size_mb", 0.0),
            "model_parameter_count_source": "config_estimate" if args.engine == "vllm" else "loaded_model",
            "args": args.__dict__,
            "chunk_length": chunk_length,
            "chunk_size": -1, # -1 indicates global mask, not chunking
            "original_size_bytes": original_size_bytes,
            "arithmetic_code_size_bytes": total_arithmetic_code_size / 8,
            "bitmap_size_bytes": total_bitmap_size / 8,
            "final_size_bytes": final_size / 8,
            "pure_compression_factor": original_size_bytes / (total_arithmetic_code_size / 8),
            "compression_factor": original_size_bytes / (final_size / 8),
            # Work counters. ``input_symbols_count`` measures useful source
            # progress, whereas ``model_input_tokens_count`` includes context
            # replay and therefore measures predictor work.
            "input_symbols_count": len(data_tokens),
            "model_input_tokens_count": input_token_cnt,
            "input_tokens_count": input_token_cnt,  # Deprecated compatibility alias.
            "entropy": float(entropy),
            # Timings
            "total_compression_time": total_compression_time,
            "tokenize_time": tokenize_time,
            "compression_time": compression_time,
            "inference_time": inference_time,
            "ac_time": ac_time,
            "msac_backend": (
                ac_backend
                if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}
                else None
            ),
            "msac_workers": (
                None
                if cuda_interval_backend
                else llm_compressor.threads
                if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}
                else None
            ),
            "msac_quantize_seconds": (
                interval_seconds
                if cuda_interval_backend
                else llm_compressor.quantize_seconds
                if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}
                else None
            ),
            "msac_range_encode_seconds": (
                cuda_ac_metrics.get("kernel_seconds")
                if cuda_interval_backend
                else llm_compressor.range_encode_seconds
                if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}
                else None
            ),
            "cuda_ac_metrics": cuda_ac_metrics or None,
            "target_interval_quantizer": (
                target_interval_quantizer
                if args.encoding == "AC_TARGET_INTERVAL"
                else None
            ),
            "interval_transfer_bytes": interval_transfer_bytes,
            "interval_conversion_seconds": (
                interval_device_seconds
                if cuda_interval_backend
                else (None if args.engine == "vllm" else interval_seconds)
            ),
            "interval_conversion_host_launch_seconds": (
                interval_seconds if cuda_interval_backend else None
            ),
            "data_copy_time": data_copy_time,
            "softmax_time": softmax_time,
            # Throughput
            "throughput_input_symbols_per_sec": len(data_tokens) / total_compression_time,
            "throughput_model_input_tokens_per_sec": input_token_cnt / total_compression_time,
            "throughput_tokens_per_sec": input_token_cnt / total_compression_time,  # Deprecated alias.
            "throughput_kibibytes_per_sec": original_size_bytes / 1024 / total_compression_time,
            "inference_throughput_input_symbols_per_sec": _rate(len(data_tokens), inference_time),
            "inference_throughput_model_input_tokens_per_sec": _rate(input_token_cnt, inference_time),
            "inference_throughput_tokens_per_sec": _rate(input_token_cnt, inference_time),  # Deprecated alias.
            "inference_throughput_kibibytes_per_sec": _rate(original_size_bytes / 1024, inference_time),
        }, args

    finally:
        cleanup = getattr(token_predictor, "cleanup", None)
        if cleanup is not None:
            cleanup()


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
        bit_string (list[int] | bytes): Coded bits or the MSAC payload.
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
    token_predictor = _make_token_predictor(args, bitmap)
    try:
        decode_backend = getattr(args, "ac_decode_backend", "host")
        device_decode = decode_backend in {"device", "cuda"}
        if device_decode and (args.encoding != "AC_TARGET_INTERVAL" or args.engine != "transformer"):
            raise ValueError("device decoding requires Transformer and AC_TARGET_INTERVAL")
        if decode_backend == "cuda" and token_predictor.device.type != "cuda":
            raise ValueError("CUDA CDF decoding requires a CUDA Transformer")
        if device_decode:
            token_ids_on_device = torch.tensor(token_predictor.tokens_list, dtype=torch.long,
                                               device=token_predictor.device)

        # The MSAC directory assigns one independent stream to each token batch.
        if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}:
            decoder_type = PairedACDecoder if getattr(args, "ac_layout", "standard") == "paired" else MultistreamACDecoder
            if decode_backend == "cuda":
                decompressor = CudaCDFMultistreamACDecoder(
                    bit_string,
                    token_predictor.device,
                    paired=getattr(args, "ac_layout", "standard") == "paired",
                )
            elif device_decode:
                decompressor = DeviceMultistreamACDecoder(
                    bit_string,
                    token_predictor.device,
                    paired=getattr(args, "ac_layout", "standard") == "paired",
                )
            else:
                decompressor = decoder_type(
                    bit_string,
                    frequency_quantizer=getattr(
                        args, "frequency_quantizer", "reference"
                    ),
                )
            if not device_decode and decompressor.target_interval != (args.encoding == "AC_TARGET_INTERVAL"):
                raise ValueError("archive encoding does not match MSAC quantization version")
            if decompressor.stream_count != args.batch_size:
                raise ValueError("MSAC stream count does not match batch size")
        else:
            decompressor = _make_arithmetic_decompressor(
                args, bit_string, alphabet_size=len(token_predictor.tokens_list)
            )

        if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"} and len(first_tokens) != args.batch_size:
            raise ValueError("MSAC seed count does not match batch size")

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
        if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}:
            expected_counts = [length - 1 for length in batches_length]
            if decompressor.symbol_counts != expected_counts:
                raise ValueError("MSAC symbol counts do not match batch lengths")

        input_tokens_cnt = 0
        inference_time = 0
        ac_time = 0
        data_copy_time = 0
        softmax_time = 0
        decoded_token_transfer_bytes = 0

        # Determine max tokens to decode, set to first_n_tokens if not specified
        max_tokens = args.first_n_tokens
        total_decoded = len(first_tokens) if first_tokens else 0

        # Iterate through all tokens in chunk length
        for token_idx in range(chunk_length):
            if total_decoded >= max_tokens:
                # if max_tokens are reached, break
                break

            # Bitstring is unbatched, therefore we have to calculate batch id again using chunk-length
            #print(f"\rProcessing batch {token_idx + 1}/{chunk_length}", end='')

            # Truncate context if it exceeds model limit (prevents overflow)
            if len(prompts[0]) >= args.context_length:
                prompts = [prompt[-args.retain_tokens:] for prompt in prompts]

            # Run LLM inference
            t0_inference = time.perf_counter()
            _, probs_values, _data_copy_time, _softmax_time = token_predictor.run_batched_inference(
                prompts, enable_kv_cache=args.use_kv_cache, scores_on_device=True
            ) if device_decode else token_predictor.run_batched_inference(
                prompts, enable_kv_cache=args.use_kv_cache
            )
            input_tokens_cnt += getattr(
                token_predictor,
                "last_model_input_tokens",
                sum(len(prompt) for prompt in prompts),
            )
            data_copy_time += _data_copy_time
            softmax_time += _softmax_time
            inference_time += time.perf_counter() - t0_inference

            t0_ac = time.perf_counter()
            # Device path keeps the CDF, lookup, and arithmetic state beside inference.
            if device_decode:
                active = [token_idx + 1 < length for length in batches_length]
                columns = decompressor.decode(probs_values, active)
                decoded_tokens = token_ids_on_device[columns].cpu().tolist()
                decoded_token_transfer_bytes += len(decoded_tokens) * token_ids_on_device.element_size()
                for idx, next_token in enumerate(decoded_tokens):
                    if active[idx]:
                        prompts[idx].append(next_token)
                        reconstructed_tokens[idx].append(next_token)
                        total_decoded += 1
            else:
                # Host decoder receives a full probability row for each stream.
                for idx, probs in enumerate(probs_values.to(torch.float32).numpy()):
                    if token_idx + 1 < batches_length[idx]:
                        # Decompress the next token's index from the bit string.
                        #print(f'probs_steps shape: {probs.shape}') --> probs_steps shape: (357,)
                        next_token_idx = (
                            decompressor.decode(idx, probs) if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}
                            else decompressor.decompress(probs)
                        )
                        next_token = token_predictor.get_token_by_id(next_token_idx)

                        # Append the decompressed token to the context for the next step.
                        prompts[idx].append(next_token)
                        reconstructed_tokens[idx].append(next_token)

                        total_decoded += 1

            ac_time += time.perf_counter() - t0_ac

        if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}:
            decompressor.assert_complete()
        reconstructed_tokens = list(chain.from_iterable(reconstructed_tokens))
        t0_detokenize = time.perf_counter()
        detoken_string = token_predictor.detokenize(reconstructed_tokens)
        detokenize_time = time.perf_counter() - t0_detokenize

        decompression_time = time.perf_counter() - t0_decompress

        return reconstructed_tokens, detoken_string, {
            "args": args.__dict__,
            "decompression_time_sec": decompression_time,
            "input_symbols_count": args.first_n_tokens,
            "model_input_tokens_count": input_tokens_cnt,
            "input_tokens_cnt": input_tokens_cnt,  # Deprecated compatibility alias.
            # Timings
            "total_decompression_time": decompression_time,
            "detokenize_time": detokenize_time,
            "inference_time": inference_time,
            "ac_time": ac_time,
            "ac_decode_backend": decode_backend,
            "cuda_decode_metrics": getattr(decompressor, "metrics", None),
            "decoded_token_transfer_bytes": decoded_token_transfer_bytes,
            "data_copy_time": data_copy_time,
            "softmax_time": softmax_time,
            # Throughput
            "throughput_input_symbols_per_sec": args.first_n_tokens / decompression_time,
            "throughput_model_input_tokens_per_sec": input_tokens_cnt / decompression_time,
            "throughput_kibibytes_per_sec": len(detoken_string) / 1024 / decompression_time,
            "inference_throughput_input_symbols_per_sec": _rate(args.first_n_tokens, inference_time),
            "inference_throughput_model_input_tokens_per_sec": _rate(input_tokens_cnt, inference_time),
            "inference_throughput_kibibytes_per_sec": _rate(len(detoken_string) / 1024, inference_time),
        }
    finally:
        cleanup = getattr(token_predictor, "cleanup", None)
        if cleanup is not None:
            cleanup()


def run_global_mask_speculative_decompression(
    args,
    first_tokens,
    bit_string,
    bitmap,
):
    """
    Speculative decompression for a global-token-mask AC stream.

    IMPORTANT:
    - If the draft and verifier are the same model/config, this function uses
      an identical-model fast path:
          * it decodes directly from the verifier distribution
          * it does NOT perform speculative comparison
          * every produced token is counted as accepted
      This guarantees 100% acceptance by construction.

    - If the draft and verifier differ, it falls back to speculative verification:
          * draft proposes k tokens
          * verifier + arithmetic coder is the source of truth
          * drafted tokens are accepted only while they exactly match the
            verifier-decoded tokens
          * on first mismatch, the verifier token is emitted and the remaining
            drafted suffix is abandoned

    Returns:
        tuple: (reconstructed_tokens_flat, detoken_string, stats)
    """
    if args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"} or args.engine == "vllm":
        raise ValueError("this engine/encoding requires standard decompression")


    print(
        f"\n----- Running Speculative Decompression: "
        f"Global Token Mask (first_n_tokens={args.first_n_tokens}, "
        f"spec_k={args.spec_k}, verifier_kv_cache={args.use_kv_cache}) -----"
    )

    t0_decompress = time.perf_counter()

    verifier = TokenPredictor(args, bitmap_data=bitmap)

    draft_model_name = getattr(args, "draft_model_name", None)
    verifier_model_name = getattr(args, "model_name", None)

    identical_model_fast_path = (draft_model_name is None or draft_model_name == verifier_model_name)
    print(f"Draft model: {draft_model_name}, Verifier model: {verifier_model_name}, "f"Using identical-model fast path: {identical_model_fast_path}")

    if draft_model_name is not None and draft_model_name != verifier_model_name:
        draft_args = copy(args)
        draft_args.model_name = draft_model_name
        # overwrite model_name with draft_model_name in draft_args to initialize the draft predictor with the correct model

    else:
        draft_args = args

    decompressor = _make_arithmetic_decompressor(
        args,
        bit_string,
        alphabet_size=len(verifier.tokens_list),
    )

    if not first_tokens:
        raise ValueError("first_tokens must contain one initial token per batch row")

    if len(first_tokens) != args.batch_size:
        raise ValueError(
            f"Expected {args.batch_size} first tokens, got {len(first_tokens)}"
        )

    prompts = [[first_tokens[i]] for i in range(args.batch_size)]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size)]

    # Per-batch target lengths (same semantics as your original code)
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
    abandoned_suffix_tokens = 0

    identical_model_fast_path = (
        draft_model_name is None or draft_model_name == verifier_model_name)

    def _truncate_context(seq):
        if len(seq) >= args.context_length:
            return seq[-args.retain_tokens:]
        return seq

    # ------------------------------------------------------------------
    # FAST PATH: identical draft/verifier
    #
    # Do NOT speculative-compare at all. Decode directly from verifier.
    # This is the only way to guarantee 0 rejections when "same model"
    # really means "same source of truth for decompression".
    # ------------------------------------------------------------------
    if identical_model_fast_path:
        print("\nUsing identical-model fast path (verifier-only decode).")

        while total_decoded < max_tokens:
            active_rows = [
                i for i in range(args.batch_size)
                if len(reconstructed_tokens[i]) < batches_length[i]
            ]
            if not active_rows:
                break

            for i in active_rows:
                prompts[i] = _truncate_context(prompts[i])

            active_prompts = [prompts[i] for i in active_rows]
            input_tokens_cnt += sum(len(p) for p in active_prompts)

            t0_ver = time.perf_counter()
            _, probs_values, _dcp_ver, _sm_ver = verifier.run_batched_inference(
                active_prompts,
                enable_kv_cache=args.use_kv_cache,
            )
            inference_time_verifier += time.perf_counter() - t0_ver
            data_copy_time_verifier += _dcp_ver
            softmax_time_verifier += _sm_ver
            verifier_steps += 1

            t0_ac = time.perf_counter()
            probs_np = probs_values.to(torch.float32).cpu().numpy()

            for j, global_row in enumerate(active_rows):
                if len(reconstructed_tokens[global_row]) >= batches_length[global_row]:
                    continue

                decoded_idx = decompressor.decompress(probs_np[j])
                decoded_token = verifier.get_token_by_id(decoded_idx)

                prompts[global_row].append(decoded_token)
                reconstructed_tokens[global_row].append(decoded_token)

                total_decoded += 1
                accepted_draft_tokens += 1  # by construction in fast path

                if total_decoded >= max_tokens:
                    break

            ac_time += time.perf_counter() - t0_ac

        flat_tokens = list(chain.from_iterable(reconstructed_tokens))

        t0_detok = time.perf_counter()
        detoken_string = verifier.detokenize(flat_tokens)
        detokenize_time = time.perf_counter() - t0_detok

        decompression_time = time.perf_counter() - t0_decompress

        return flat_tokens, detoken_string, {
            "args": args.__dict__,
            "mode": "identical_model_fast_path",
            "identical_model_fast_path": True,
            "decompression_time_sec": decompression_time,
            "input_tokens_cnt": input_tokens_cnt,
            "total_decompression_time": decompression_time,
            "detokenize_time": detokenize_time,
            "verifier_inference_time": inference_time_verifier,
            "draft_inference_time": 0.0,
            "ac_time": ac_time,
            "verifier_data_copy_time": data_copy_time_verifier,
            "verifier_softmax_time": softmax_time_verifier,
            "draft_data_copy_time": 0.0,
            "draft_softmax_time": 0.0,
            "verifier_steps": verifier_steps,
            "draft_cycles": 0,
            "accepted_draft_tokens": accepted_draft_tokens,
            "rejected_draft_tokens": 0,
            "abandoned_suffix_tokens": 0,
            "acceptance_rate": 1.0,
            "proposal_efficiency": 1.0,
            "throughput_kibibytes_per_sec": (
                len(detoken_string) / 1024 / max(decompression_time, 1e-12)
            ),
            "verifier_inference_throughput_kibibytes_per_sec": (
                len(detoken_string) / 1024 / max(inference_time_verifier, 1e-12)
            ),
        }

    # ------------------------------------------------------------------
    # FALLBACK: speculative verification for non-identical models
    #
    # This stays close to your original implementation.
    # verifier remains the source of truth for AC decode.
    # ------------------------------------------------------------------
    draft = TokenPredictor(draft_args, bitmap_data=bitmap)

    while total_decoded < max_tokens:
        active_rows = [
            i for i in range(args.batch_size)
            if len(reconstructed_tokens[i]) < batches_length[i]
        ]
        if not active_rows:
            break

        for i in active_rows:
            prompts[i] = _truncate_context(prompts[i])

        active_prompts = [prompts[i] for i in active_rows]

        remaining_per_row = [
            batches_length[i] - len(reconstructed_tokens[i])
            for i in active_rows
        ]
        current_k = min(args.spec_k, max(remaining_per_row))
        if current_k <= 0:
            break

        #print(
        #    f"\rProcessing speculative cycle, active_rows={len(active_rows)}, k={current_k}",
        #    end=""
        #)

        # --- Draft phase: generate k tokens from the draft model for each active row ---
                # 1) Draft phase
        # generate_draft() requires all prompts in a batch to have equal length,
        # so bucket rows by prompt length.

        t0_draft = time.perf_counter()

        drafted_tokens = [None] * len(active_rows)
        total_dcp_draft = 0.0
        total_sm_draft = 0.0

        length_buckets = defaultdict(list)
        for local_idx, row in enumerate(active_prompts):
            length_buckets[len(row)].append(local_idx)

        for _, bucket_local_indices in length_buckets.items():
            bucket_prompts = [active_prompts[idx][:] for idx in bucket_local_indices]

            _, _, _dcp_draft, _sm_draft, bucket_drafted_prompts = draft.generate_draft(
                bucket_prompts,
                k=current_k,
                enable_kv_cache=False,
                full_draft=True,
            )

            total_dcp_draft += _dcp_draft
            total_sm_draft += _sm_draft

            for bucket_pos, local_idx in enumerate(bucket_local_indices):
                base = active_prompts[local_idx]
                full = bucket_drafted_prompts[bucket_pos]
                delta = full[len(base):]

                if len(delta) < current_k:
                    raise RuntimeError(
                        f"Draft returned only {len(delta)} tokens, expected {current_k}"
                    )

                drafted_tokens[local_idx] = delta[:current_k]

        inference_time_draft += time.perf_counter() - t0_draft
        data_copy_time_draft += total_dcp_draft
        softmax_time_draft += total_sm_draft
        draft_cycles += 1

        # 2) Verifier + AC decode phase
        local_prompts = [row[:] for row in active_prompts]
        local_done = [False] * len(active_rows)

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
                enable_kv_cache=False,
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

                prompts[global_row].append(decoded_token)
                reconstructed_tokens[global_row].append(decoded_token)
                local_prompts[r].append(decoded_token)
                total_decoded += 1

                if decoded_token == drafted_token:
                    accepted_draft_tokens += 1
                else:
                    rejected_draft_tokens += 1
                    abandoned_suffix_tokens += max(0, current_k - step_idx - 1)
                    local_done[r] = True

                if total_decoded >= max_tokens:
                    break

            ac_time += time.perf_counter() - t0_ac

            if total_decoded >= max_tokens:
                break

    print()

    flat_tokens = list(chain.from_iterable(reconstructed_tokens))

    t0_detok = time.perf_counter()
    detoken_string = verifier.detokenize(flat_tokens)
    detokenize_time = time.perf_counter() - t0_detok

    decompression_time = time.perf_counter() - t0_decompress

    return flat_tokens, detoken_string, {
        "args": args.__dict__,
        "mode": "speculative_verification",
        "identical_model_fast_path": False,
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
        "abandoned_suffix_tokens": abandoned_suffix_tokens,
        "acceptance_rate": (
            accepted_draft_tokens / max(1, accepted_draft_tokens + rejected_draft_tokens)
        ),
        "proposal_efficiency": (
            accepted_draft_tokens
            / max(1, accepted_draft_tokens + rejected_draft_tokens + abandoned_suffix_tokens)
        ),
        "throughput_kibibytes_per_sec": (
            len(detoken_string) / 1024 / max(decompression_time, 1e-12)
        ),
        "verifier_inference_throughput_kibibytes_per_sec": (
            len(detoken_string) / 1024 / max(inference_time_verifier, 1e-12)
        ),
    }


def run_global_mask_speculative_decompression_old(
    args,
    first_tokens,
    bit_string,
    bitmap,):
    """
    Speculative decompression for a global-token-mask AC stream.

    Correctness rule:
    - The arithmetic-coded verifier distribution is always the source of truth.
    - Draft tokens are only used as proposals.
    - A drafted token is accepted iff it exactly matches the token decoded
      from the verifier distribution at that same position.
    - On the first mismatch, we emit the verifier-decoded token and abandon
      the rest of the drafted suffix for that row.

    This is the correct decompression analogue of speculative decoding.
    It does NOT use the paper's stochastic accept/reject resampling step,
    because doing so would alter the decoded AC stream.

    Returns:
        tuple: (reconstructed_tokens_flat, detoken_string, stats)
    """

    t
    print(
        f"\n----- Running Speculative Decompression: "
        f"Global Token Mask (first_n_tokens={args.first_n_tokens}, "
        f"spec_k={args.spec_k}, verifier_kv_cache={args.use_kv_cache}) -----"
    )

    verifier = TokenPredictor(args, bitmap_data=bitmap)

    draft_model_name = getattr(args, "draft_model_name", None)
    verifier_model_name = getattr(args, "model_name", None)

    identical_model_fast_path = (draft_model_name is None or draft_model_name == verifier_model_name)
    print(f"Draft model: {draft_model_name}, Verifier model: {verifier_model_name}, "f"Using identical-model fast path: {identical_model_fast_path}")

    # overwrite model_name with draft_model_name in draft_args to initialize the draft predictor with the correct model
    draft_args = args
    draft_args.model_name = draft_model_name

    draft = TokenPredictor(draft_args, bitmap_data=bitmap)
    decompressor = _make_arithmetic_decompressor(
        args,
        bit_string,
        alphabet_size=len(verifier.tokens_list),
    )

    prompts = [[first_tokens[i]] for i in range(args.batch_size)]
    reconstructed_tokens = [[first_tokens[i]] for i in range(args.batch_size)]

    # Total target tokens per row (including the initial token already present).
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
    abandoned_suffix_tokens = 0


    def _truncate_context_inplace(seq):
        if len(seq) >= args.context_length:
            return seq[-args.retain_tokens:]
        return seq

    while total_decoded < max_tokens:
        active_rows = [
            i for i in range(args.batch_size)
            if len(reconstructed_tokens[i]) < batches_length[i]
        ]
        if not active_rows:
            break

        for i in active_rows:
            prompts[i] = _truncate_context_inplace(prompts[i])

        active_prompts = [prompts[i] for i in active_rows]
        remaining_per_row = [
            batches_length[i] - len(reconstructed_tokens[i])
            for i in active_rows
        ]

        current_k = min(args.spec_k, max(remaining_per_row))
        if current_k <= 0:
            break

        #print(
        #    f"\rProcessing speculative cycle, active_rows={len(active_rows)}, k={current_k}",
        #    end=""
        #)

        # ------------------------------------------------------------------
        # 1) Draft phase: produce k proposals per active row
        # ------------------------------------------------------------------
        t0_draft = time.perf_counter()
        _, _, _dcp_draft, _sm_draft, drafted_prompts = draft.generate_draft(
            [row[:] for row in active_prompts],
            k=current_k,
            enable_kv_cache=False,   # correctness-first; enable only if wrapper supports true incremental mode
            full_draft=True,
        )
        inference_time_draft += time.perf_counter() - t0_draft
        data_copy_time_draft += _dcp_draft
        softmax_time_draft += _sm_draft
        draft_cycles += 1

        drafted_tokens = []
        effective_k_per_row = []

        for base, full, rem in zip(active_prompts, drafted_prompts, remaining_per_row):
            delta = full[len(base):]
            row_k = min(current_k, rem)
            if len(delta) < row_k:
                raise RuntimeError(
                    f"Draft returned only {len(delta)} tokens, expected at least {row_k}"
                )
            drafted_tokens.append(delta[:row_k])
            effective_k_per_row.append(row_k)

        # ------------------------------------------------------------------
        # 2) Verifier scoring phase:
        #    score every drafted prefix in one big batched call
        #
        # For each row with prompt P and drafted tokens d1..dk, we score:
        #   P
        #   P+d1
        #   P+d1+d2
        #   ...
        #   P+d1+...+d(k-1)
        #
        # Each scored prompt predicts the next token distribution.
        # That gives us verifier distributions for positions 1..k.
        # ------------------------------------------------------------------
        verify_prompts = []
        verify_index = []   # (row_local_idx, step_idx)

        for r, base in enumerate(active_prompts):
            row_k = effective_k_per_row[r]
            prefix = base[:]
            for step_idx in range(row_k):
                verify_prompts.append(prefix[:])
                verify_index.append((r, step_idx))
                prefix.append(drafted_tokens[r][step_idx])

        if not verify_prompts:
            break

        input_tokens_cnt += sum(len(p) for p in verify_prompts)

        t0_ver = time.perf_counter()
        _, probs_values, _dcp_ver, _sm_ver = verifier.run_batched_inference(
            verify_prompts,
            enable_kv_cache=False,
        )
        inference_time_verifier += time.perf_counter() - t0_ver
        data_copy_time_verifier += _dcp_ver
        softmax_time_verifier += _sm_ver
        verifier_steps += 1

        # probs_values: [sum_row_k, vocab]
        probs_values = probs_values.to(torch.float32).cpu()
        per_row_probs = [[] for _ in range(len(active_rows))]
        for idx, (r, step_idx) in enumerate(verify_index):
            per_row_probs[r].append(probs_values[idx])

        # ------------------------------------------------------------------
        # 3) AC decode against verifier distributions; accept matching draft prefix
        # ------------------------------------------------------------------
        t0_ac = time.perf_counter()

        for r, global_row in enumerate(active_rows):
            if total_decoded >= max_tokens:
                break

            row_k = effective_k_per_row[r]
            if row_k <= 0:
                continue

            accepted_prefix_len = 0
            row_done = False

            for step_idx in range(row_k):
                if len(reconstructed_tokens[global_row]) >= batches_length[global_row]:
                    row_done = True
                    break

                decoded_idx = decompressor.decompress(per_row_probs[r][step_idx].numpy())
                decoded_token = verifier.get_token_by_id(decoded_idx)
                drafted_token = drafted_tokens[r][step_idx]

                prompts[global_row].append(decoded_token)
                reconstructed_tokens[global_row].append(decoded_token)
                total_decoded += 1

                if decoded_token == drafted_token:
                    accepted_draft_tokens += 1
                    accepted_prefix_len += 1
                    print(decoded_token, end="")  # Debug print to visualize accepted tokens
                    print(drafted_token, end="")  # Debug print to visualize drafted tokens
                    break

                else:
                    rejected_draft_tokens += 1
                    abandoned_suffix_tokens += max(0, row_k - step_idx - 1)
                    row_done = True
                    break

                if total_decoded >= max_tokens:
                    row_done = True
                    break

            # If all row_k draft tokens matched, no rejection is counted.
            # We simply move on to the next speculative cycle.

        ac_time += time.perf_counter() - t0_ac

    print()

    flat_tokens = list(chain.from_iterable(reconstructed_tokens))

    t0_detok = time.perf_counter()
    detoken_string = verifier.detokenize(flat_tokens)
    detokenize_time = time.perf_counter() - t0_detok

    decompression_time = time.perf_counter() - t0_decompress

    return flat_tokens, detoken_string, {
        "args": args.__dict__,
        "mode": "speculative_verification_for_ac_decompression",
        "identical_model_fast_path": identical_model_fast_path,
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
        "abandoned_suffix_tokens": abandoned_suffix_tokens,
        "acceptance_rate": (
            accepted_draft_tokens / max(1, accepted_draft_tokens + rejected_draft_tokens)
        ),
        "proposal_efficiency": (
            accepted_draft_tokens /
            max(1, accepted_draft_tokens + rejected_draft_tokens + abandoned_suffix_tokens)
        ),
        "throughput_kibibytes_per_sec": (
            len(detoken_string) / 1024 / max(decompression_time, 1e-12)
        ),
        "verifier_inference_throughput_kibibytes_per_sec": (
            len(detoken_string) / 1024 / max(inference_time_verifier, 1e-12)
        ),
    }


def run_global_mask_speculative_decompression_pseudo_code():
    pass
    #draft k tokens,
        # --> can be imported from TokenPredictor.generate_draft with full_draft=True
    #score all drafted prefixes with the verifier,
        # --> can be imported from TokenPredictor.run_batched_inference on the set of all drafted prefixes
    #accept the longest exact prefix that matches verifier-driven decoding,
        # has to be implemented here or in helper function that takes verifier distributions and drafted tokens as input.
    #stop at first mismatch,
    #append the true verifier-decoded token,
    #discard the remaining drafted suffix.

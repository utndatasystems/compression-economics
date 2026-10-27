#!/usr/bin/env python3
"""Benchmark LLMZip arithmetic coding on an exact byte prefix.

The adapter preserves LLMZip's fixed sliding context, probability-to-frequency
rule, and reference arithmetic coder while using a Hugging Face causal LM.  A
batch size of one is intentional: upstream marks batched encoding as
incompatible with decoding, because arithmetic coding requires the encoder and
decoder to reproduce identical probability tables.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.papers.coding_survey.experiments.baseline_adapter import (
    build_baseline_result,
    git_revision,
    hugging_face_model_metadata,
    load_text_prefix,
    prepare_work_archive,
    read_framed_archive,
    torch_dtype,
    torch_environment,
    write_framed_archive,
    write_json,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/llm-baselines/llmzip-text8-first-10MiB.json"
)
MAGIC = b"LZP1"
FREQUENCY_SCALE = 10_000_000


class _NonClosingBytesIO(io.BytesIO):
    """Let upstream BitOutputStream flush without hiding its bytes."""

    def close(self) -> None:
        pass


def _load_arithmetic_coder(repository: Path):
    source = repository / "AC/arithmeticcoding.py"
    if not source.is_file():
        raise FileNotFoundError(f"LLMZip arithmetic coder not found: {source}")
    spec = importlib.util.spec_from_file_location("llmzip_arithmeticcoding", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cdf(logits: torch.Tensor) -> np.ndarray:
    """Apply upstream's ``cumsum(prob * 1e7 + 1)`` quantizer."""

    probabilities = torch.softmax(logits.float(), dim=-1)
    probabilities_np = probabilities.cpu().numpy().reshape(-1)
    cumulative = np.empty(probabilities_np.size + 1, dtype=np.uint64)
    cumulative[0] = 0
    cumulative[1:] = np.cumsum(probabilities_np * FREQUENCY_SCALE + 1)
    return cumulative


def _next_logits(model, context: torch.Tensor) -> torch.Tensor:
    output = model(
        input_ids=context,
        use_cache=False,
        logits_to_keep=1,
        return_dict=True,
    )
    return output.logits[0, -1]


def _seed_token_id(tokenizer, requested: int | None) -> tuple[int, str]:
    """Choose a deterministic initial context, including for BOS-less models."""

    if requested is not None:
        return requested, "explicit"
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id, "bos"
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id, "eos_fallback"
    raise ValueError(
        "tokenizer defines neither BOS nor EOS; pass --seed-token-id explicitly"
    )


def _progress(label: str, completed: int, total: int, last_percent: int) -> int:
    percent = completed * 100 // total
    if percent >= last_percent + 5 or completed == total:
        print(f"{label}: {percent}% ({completed:,}/{total:,} tokens)", flush=True)
        return percent
    return last_percent


@torch.inference_mode()
def _encode(
    model,
    tokens: torch.Tensor,
    seed_token_id: int,
    memory: int,
    arithmetic,
) -> bytes:
    payload_buffer = _NonClosingBytesIO()
    bit_output = arithmetic.BitOutputStream(payload_buffer)
    encoder = arithmetic.ArithmeticEncoder(32, bit_output)
    device_tokens = tokens.to(device="cuda", dtype=torch.long)
    seed = torch.tensor([[seed_token_id]], device="cuda", dtype=torch.long)
    last_percent = -5
    for index in range(device_tokens.numel()):
        if index == 0:
            context = seed
        elif index < memory:
            context = torch.cat((seed, device_tokens[:index].unsqueeze(0)), dim=1)
        else:
            context = device_tokens[index - memory : index].unsqueeze(0)
        cumulative = _cdf(_next_logits(model, context))
        encoder.write(cumulative, int(tokens[index].item()))
        last_percent = _progress("Encoder", index + 1, device_tokens.numel(), last_percent)
    encoder.finish()
    bit_output.close()
    return payload_buffer.getvalue()


@torch.inference_mode()
def _decode(
    model,
    payload: bytes,
    token_count: int,
    seed_token_id: int,
    memory: int,
    vocabulary_size: int,
    arithmetic,
) -> torch.Tensor:
    bit_input = arithmetic.BitInputStream(io.BytesIO(payload))
    decoder = arithmetic.ArithmeticDecoder(32, bit_input)
    tokens = torch.empty(token_count, device="cuda", dtype=torch.long)
    seed = torch.tensor([[seed_token_id]], device="cuda", dtype=torch.long)
    last_percent = -5
    for index in range(token_count):
        if index == 0:
            context = seed
        elif index < memory:
            context = torch.cat((seed, tokens[:index].unsqueeze(0)), dim=1)
        else:
            context = tokens[index - memory : index].unsqueeze(0)
        cumulative = _cdf(_next_logits(model, context))
        token = decoder.read(cumulative, vocabulary_size)
        tokens[index] = token
        last_percent = _progress("Decoder", index + 1, token_count, last_percent)
    bit_input.close()
    return tokens.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llmzip-repo", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--input-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--tokenizer")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--seed-token-id",
        type=int,
        help="initial context token (defaults to BOS, then EOS for BOS-less models)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--memory", type=int, default=511)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("LLMZip benchmark requires a visible CUDA device")
    if args.input_bytes < 1 or args.memory < 1:
        parser.error("input-bytes and memory must be positive")

    upstream = args.llmzip_repo.resolve()
    arithmetic = _load_arithmetic_coder(upstream)
    try:
        source, text = load_text_prefix(args.input, args.input_bytes)
        work_dir, archive_path = prepare_work_archive(
            args.output, args.work_dir, "llmzip.lzp"
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))

    tokenizer_name = args.tokenizer or args.model
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=args.model_revision,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=torch_dtype(args.dtype),
        local_files_only=True,
    ).eval().cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    try:
        seed_token_id, seed_token_source = _seed_token_id(
            tokenizer, args.seed_token_id
        )
    except ValueError as error:
        parser.error(str(error))

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.decode(token_ids).encode("utf-8") != source:
        parser.error("tokenizer does not round-trip the input bytes")
    tokens = torch.tensor(token_ids, dtype=torch.long)
    warm_context = torch.tensor([[seed_token_id]], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        warm_logits = _next_logits(model, warm_context)
    vocabulary_size = warm_logits.numel()
    if not 0 <= seed_token_id < vocabulary_size:
        parser.error(
            f"seed token ID {seed_token_id} is outside model vocabulary "
            f"[0, {vocabulary_size})"
        )
    if token_ids and max(token_ids) >= vocabulary_size:
        parser.error("tokenizer emitted an ID outside the model vocabulary")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    payload = _encode(
        model,
        tokens,
        seed_token_id,
        args.memory,
        arithmetic,
    )
    torch.cuda.synchronize()
    compression_seconds = time.perf_counter() - started

    archive_metadata = {
        "algorithm": "LLMZip fixed-window arithmetic coding",
        "arithmetic_state_bits": 32,
        "dtype": args.dtype,
        "format": "LZP1",
        "frequency_scale": FREQUENCY_SCALE,
        "memory": args.memory,
        "model": args.model,
        "model_revision": args.model_revision,
        "payload_bytes": len(payload),
        "seed_token_id": seed_token_id,
        "seed_token_source": seed_token_source,
        "softmax_dtype": "float32",
        "token_count": len(token_ids),
        "tokenizer": tokenizer_name,
        "vocabulary_size": vocabulary_size,
    }
    write_framed_archive(archive_path, MAGIC, archive_metadata, payload)
    extracted, extracted_payload = read_framed_archive(
        archive_path, MAGIC, codec_label="LLMZip"
    )

    started = time.perf_counter()
    decoded_tokens = _decode(
        model,
        extracted_payload,
        extracted["token_count"],
        extracted["seed_token_id"],
        extracted["memory"],
        extracted["vocabulary_size"],
        arithmetic,
    )
    decoded_text = tokenizer.decode(decoded_tokens.tolist())
    torch.cuda.synchronize()
    decompression_seconds = time.perf_counter() - started
    recovered = decoded_text.encode("utf-8")
    exact = recovered == source
    if not exact:
        (work_dir / "decoded-mismatch.bin").write_bytes(recovered)
        raise AssertionError("LLMZip did not recover the exact input bytes")

    result = build_baseline_result(
        codec="llmzip",
        implementation={
            "repository": "https://github.com/vcskaushik/LLMzip",
            "revision": git_revision(upstream),
            "adapter_archive": "LZP1 (repository-local accounting wrapper)",
            "upstream_arithmetic_coder": "AC/arithmeticcoding.py",
        },
        input_path=args.input,
        source=source,
        model=hugging_face_model_metadata(
            name=args.model,
            requested_revision=args.model_revision,
            tokenizer=tokenizer_name,
            dtype=args.dtype,
        ),
        configuration={
            "encoder_batch_size": 1,
            "decoder_batch_size": 1,
            "memory": args.memory,
            "seed_token_id": seed_token_id,
            "seed_token_source": seed_token_source,
            "token_count": len(token_ids),
            "tokenizer_size": len(tokenizer),
            "model_vocabulary_size": vocabulary_size,
            "frequency_rule": "uint64(cumsum(softmax(float32) * 1e7 + 1))",
        },
        archive_path=archive_path,
        payload_bytes=len(payload),
        model_load_seconds=model_load_seconds,
        compression_seconds=compression_seconds,
        decompression_seconds=decompression_seconds,
        exact_byte_recovery=exact,
        environment=torch_environment(
            torch, transformers_version=transformers.__version__
        ),
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

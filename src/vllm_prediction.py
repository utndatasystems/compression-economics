"""
vLLM-based token prediction backend for compression experiments.

This module provides VLLMTokenPredictor, a drop-in alternative to TokenPredictor
that uses the vLLM engine for higher-throughput inference. It supports two modes:

- Step-by-step mode (run_batched_inference): captures full logits via a
  logits_processor callback. Used for AC encoding and all decompression paths.
- Bulk rank mode (run_bulk_rank_inference): uses prompt_logprobs to extract
  ranks for an entire sequence in a single forward pass. Used for rank-based
  compression (bitpacked/huffman) — the biggest performance win.

vLLM manages its own KV cache internally via prefix caching, so context trimming
is not needed. This means results may differ from the transformer engine (typically
better compression due to full context).
"""

import time
import os
import glob
import hashlib
import subprocess
import tempfile
import uuid
import inspect
import torch
import numpy as np
from pyroaring import BitMap
from packaging.version import InvalidVersion, Version

try:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor
    VLLM_AVAILABLE = True
except ImportError:
    vllm = None
    LLM = None
    SamplingParams = None
    LogitsProcessor = object
    VLLM_AVAILABLE = False

from src.hf_cache import get_model_cache_dir


DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.9
MIN_VLLM_GPU_MEMORY_UTILIZATION = 0.05
GPU_MEMORY_HEADROOM_GIB = 1.0
MIN_NATIVE_AC_VLLM_VERSION = Version("0.17.1")
VLLM_FULL_LOGITS_CAPTURE_ENV = "COMPRESSION_ECONOMICS_VLLM_LOGITS_CAPTURE"
VLLM_FULL_LOGITS_PROCESSOR_FQCN = (
    "src.vllm_prediction:VLLMFullLogitsCaptureProcessor"
)
VLLM_AC_LOGIT_QUANTIZATION_STEP = 0.01


def stabilize_ac_logits(logits: torch.Tensor) -> torch.Tensor:
    """Quantize captured logits so repeated vLLM calls map to the same AC tables."""
    if logits.ndim != 2:
        raise ValueError("AC logits must be a 2D tensor.")
    return torch.round(logits / VLLM_AC_LOGIT_QUANTIZATION_STEP) * VLLM_AC_LOGIT_QUANTIZATION_STEP


def prompt_signature(prompt_tokens) -> str:
    hasher = hashlib.sha1()
    hasher.update(",".join(str(token_id) for token_id in prompt_tokens).encode("utf-8"))
    return hasher.hexdigest()


def _get_vllm_version():
    if not VLLM_AVAILABLE:
        return None
    version_string = getattr(vllm, "__version__", None)
    if version_string is None:
        return None
    try:
        return Version(version_string)
    except InvalidVersion:
        return None


def probe_vllm_ac_support(args=None):
    """Return (is_supported, reason) for the native vLLM AC logits adapter."""
    if not VLLM_AVAILABLE:
        return False, "vLLM is not installed."

    version = _get_vllm_version()
    if version is None:
        return False, "Unable to determine the installed vLLM version."
    if version < MIN_NATIVE_AC_VLLM_VERSION:
        return (
            False,
            f"vLLM {version} is installed, but native AC support requires "
            f"vLLM >= {MIN_NATIVE_AC_VLLM_VERSION}.",
        )

    try:
        llm_init_parameters = inspect.signature(LLM.__init__).parameters
    except (TypeError, ValueError):
        return False, "Unable to inspect the vLLM LLM constructor."

    if "logits_processors" not in llm_init_parameters:
        return (
            False,
            "The installed vLLM build does not expose the global logits "
            "processor hook required by the native AC adapter.",
        )

    if args is not None and getattr(args, "tensor_parallel_size", 1) != 1:
        return (
            False,
            "Native vLLM AC is currently verified only for tensor_parallel_size=1.",
        )

    return True, "native-full-logits-adapter-available"


def vllm_supports_ac_distribution(args=None):
    """Return whether the native vLLM AC adapter is available."""
    supported, _reason = probe_vllm_ac_support(args)
    return supported


class VLLMFullLogitsCaptureProcessor(LogitsProcessor):
    """vLLM logits processor that writes dense next-token logits to disk."""

    @classmethod
    def validate_params(cls, sampling_params):
        return None

    def __init__(self, vllm_config, device, is_pin_memory):
        self.capture_path = os.environ.get(VLLM_FULL_LOGITS_CAPTURE_ENV)
        self.capture_index = 0
        self.row_signatures = []
        self.row_cursor = 0

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.capture_path:
            return logits

        cpu_logits = logits.detach().to("cpu", dtype=torch.float32)
        row_count = cpu_logits.shape[0]
        shard_signatures = self.row_signatures[self.row_cursor : self.row_cursor + row_count]
        self.row_cursor += row_count
        shard_path = f"{self.capture_path}.{self.capture_index:06d}.pt"
        tmp_path = f"{shard_path}.tmp"
        torch.save({"logits": cpu_logits, "signatures": shard_signatures}, tmp_path)
        os.replace(tmp_path, shard_path)
        self.capture_index += 1
        return logits

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        self.row_cursor = 0
        if not batch_update:
            return None

        size = batch_update.batch_size
        if len(self.row_signatures) < size:
            self.row_signatures.extend([None] * (size - len(self.row_signatures)))

        for index in batch_update.removed:
            if index < len(self.row_signatures):
                self.row_signatures[index] = None

        for index, _params, prompt_tok_ids, _output_tok_ids in batch_update.added:
            self.row_signatures[index] = prompt_signature(prompt_tok_ids or [])

        for source_index, dest_index, directionality in batch_update.moved:
            source_signature = self.row_signatures[source_index]
            dest_signature = self.row_signatures[dest_index]
            self.row_signatures[dest_index] = source_signature
            if directionality.name == "SWAP":
                self.row_signatures[source_index] = dest_signature
            else:
                self.row_signatures[source_index] = None

        self.row_signatures = self.row_signatures[:size]
        return None


class VLLMFullLogitsAdapter:
    """Compatibility layer for dense next-token logits on vLLM 0.17.1+."""

    def __init__(self, args):
        supported, reason = probe_vllm_ac_support(args)
        if not supported:
            raise RuntimeError(reason)

        capture_name = f"compression-economics-vllm-{os.getpid()}-{uuid.uuid4().hex}.pt"
        self.capture_path = os.path.join(tempfile.gettempdir(), capture_name)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ[VLLM_FULL_LOGITS_CAPTURE_ENV] = self.capture_path

    def get_llm_init_kwargs(self):
        return {"logits_processors": [VLLM_FULL_LOGITS_PROCESSOR_FQCN]}

    def clear_capture(self):
        for shard_path in glob.glob(f"{self.capture_path}.*.pt"):
            os.remove(shard_path)

    def pop_captured_logits(self, expected_rows, expected_vocab_size, expected_prompt_signatures=None):
        shard_paths = sorted(glob.glob(f"{self.capture_path}.*.pt"))
        if not shard_paths:
            raise RuntimeError(
                "vLLM native AC logits capture did not produce a logits tensor. "
                "The installed vLLM worker may not have loaded the compatibility adapter."
            )

        shard_payloads = [torch.load(shard_path, map_location="cpu") for shard_path in shard_paths]
        self.clear_capture()

        normalized_parts = []
        captured_signatures = []
        for payload in shard_payloads:
            logits = payload
            shard_signatures = None
            if isinstance(payload, dict):
                logits = payload.get("logits")
                shard_signatures = payload.get("signatures")
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError(
                    "vLLM native AC logits capture returned an unexpected payload type."
                )
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            normalized_parts.append(logits)
            if shard_signatures is not None:
                captured_signatures.extend(shard_signatures)

        logits = torch.cat(normalized_parts, dim=0)

        if logits.shape[0] != expected_rows:
            raise RuntimeError(
                "vLLM native AC logits capture returned an unexpected batch size: "
                f"expected {expected_rows}, got {tuple(logits.shape)}."
            )
        if logits.shape[1] < expected_vocab_size:
            raise RuntimeError(
                "vLLM native AC logits capture returned an unexpected vocabulary size: "
                f"expected {expected_vocab_size}, got {tuple(logits.shape)}."
            )

        if logits.shape[1] > expected_vocab_size:
            logits = logits[:, :expected_vocab_size]

        if expected_prompt_signatures is not None and captured_signatures:
            logits = self._reorder_logits_by_prompt_signature(
                logits,
                captured_signatures,
                expected_prompt_signatures,
            )

        return logits

    def _reorder_logits_by_prompt_signature(self, logits, captured_signatures, expected_prompt_signatures):
        if len(captured_signatures) != logits.shape[0]:
            raise RuntimeError(
                "vLLM native AC logits capture produced mismatched row metadata."
            )

        used_indices = set()
        reordered_indices = []
        for signature in expected_prompt_signatures:
            match_index = next(
                (
                    idx
                    for idx, captured_signature in enumerate(captured_signatures)
                    if idx not in used_indices and captured_signature == signature
                ),
                None,
            )
            if match_index is None:
                raise RuntimeError(
                    "vLLM native AC logits capture could not realign rows to the input prompt order."
                )
            used_indices.add(match_index)
            reordered_indices.append(match_index)

        return logits.index_select(0, torch.tensor(reordered_indices, dtype=torch.long))


def _query_nvidia_smi_gpus():
    """Return visible GPU memory stats from nvidia-smi, or an empty list if unavailable."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception:
        return []

    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
            total_gib = int(parts[1]) / 1024.0
            free_gib = int(parts[2]) / 1024.0
        except ValueError:
            continue
        if total_gib <= 0:
            continue
        gpus.append(
            {
                "index": index,
                "total_gib": total_gib,
                "free_gib": free_gib,
                "free_fraction": free_gib / total_gib,
            }
        )
    return gpus


def _maybe_select_vllm_gpus(tensor_parallel_size):
    """Prefer the freest GPUs when the user has not pinned a device explicitly."""
    if tensor_parallel_size < 1:
        return []

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return []

    gpus = _query_nvidia_smi_gpus()
    if len(gpus) < tensor_parallel_size:
        return []

    selected_gpus = sorted(
        gpus,
        key=lambda gpu: (gpu["free_gib"], gpu["total_gib"], gpu["index"]),
        reverse=True,
    )[:tensor_parallel_size]

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu["index"]) for gpu in selected_gpus
    )
    return selected_gpus


def _resolve_gpu_memory_utilization(args, selected_gpus):
    """Clamp vLLM memory reservation so startup succeeds on shared GPUs."""
    requested = getattr(args, "gpu_memory_utilization", None)
    if requested is None:
        requested = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION

    if not 0 < requested <= 1:
        raise ValueError("gpu_memory_utilization must be in the range (0, 1].")

    if not selected_gpus:
        return requested

    limiting_gpu = min(selected_gpus, key=lambda gpu: gpu["free_fraction"])
    safe_fraction = max(
        MIN_VLLM_GPU_MEMORY_UTILIZATION,
        (limiting_gpu["free_gib"] - GPU_MEMORY_HEADROOM_GIB) / limiting_gpu["total_gib"],
    )
    safe_fraction = min(requested, safe_fraction)
    return round(safe_fraction, 3)


class VLLMTokenPredictor:
    def __init__(self, args, bitmap_data):
        """
        Initialize the vLLM predictor and load the model.

        Args:
            args (argparse.Namespace): Experiment configuration. Expected fields:
                model_name (str): HuggingFace model name.
                encoding (str): "AC", "bitpacked", or "huffman".
                reduce_tokens (bool): Whether to use a reduced token list.
                tensor_parallel_size (int): Number of GPUs for tensor parallelism.
            bitmap_data (bytes | None): Serialized roaring bitmap of allowed tokens.
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install -e '.[vllm]'"
            )

        if getattr(args, "lora_path", None) is not None:
            raise ValueError(
                "LoRA adapters are not supported with engine='vllm' in this version. "
                "Use engine='transformer' for LoRA support."
            )

        self.args = args
        self.full_logits_adapter = None
        cache_dir = get_model_cache_dir()
        tp_size = getattr(args, "tensor_parallel_size", 1)
        selected_gpus = _maybe_select_vllm_gpus(tp_size)
        effective_gpu_memory_utilization = _resolve_gpu_memory_utilization(
            args,
            selected_gpus,
        )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print(f"Initializing vLLM engine for {args.model_name} (TP={tp_size})...")
        if selected_gpus:
            selected_gpu_ids = ", ".join(str(gpu["index"]) for gpu in selected_gpus)
            print(f"Selected GPU(s) for vLLM: {selected_gpu_ids}")
        print(
            "Using vLLM gpu_memory_utilization="
            f"{effective_gpu_memory_utilization:.3f}"
        )
        llm_init_kwargs = {}
        if args.encoding == "AC":
            self.full_logits_adapter = VLLMFullLogitsAdapter(args)
            llm_init_kwargs.update(self.full_logits_adapter.get_llm_init_kwargs())

        self.llm = LLM(
            model=args.model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=effective_gpu_memory_utilization,
            enable_prefix_caching=True,
            download_dir=cache_dir,
            **llm_init_kwargs,
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Reconstruct reduced token list from bitmap (same as TokenPredictor).
        self.reduce_tokens = args.reduce_tokens
        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.index_tensor = torch.tensor(self.tokens_list, dtype=torch.long, device="cpu")

        # Model parameter counting — inspect vLLM internals with fallback.
        self.adapter_params = 0
        self.adapter_size_mb = 0.0
        try:
            model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
            model = model_runner.model
            total_bytes = 0
            total_params = 0
            for p in model.parameters():
                total_params += p.numel()
                total_bytes += p.numel() * p.element_size()
            self.base_params = total_params
            self.base_size_mb = total_bytes / (1024 ** 2)
        except Exception:
            # Fallback: estimate from config
            self.base_params = 0
            self.base_size_mb = 0.0

        print(f"vLLM engine ready. Model: {args.model_name}, params: {self.base_params:,}")

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """
        Run one-step inference and return scores (probabilities or logits).

        For arithmetic coding, this uses the native full-logits adapter to
        capture dense next-token logits before sampler truncation. For rank-based
        encodings, it keeps using vLLM prompt_logprobs. vLLM manages KV caching
        internally via prefix caching.

        Args:
            prompts (List[List[int]]): Batched tokenized prompts.
            enable_kv_cache (bool): Accepted for interface compatibility; ignored
                (vLLM always uses prefix caching).

        Returns:
            Tuple[List[int], torch.Tensor, float, float]:
                (tokens_list, scores, data_copy_time, softmax_time)
        """
        softmax_time = 0.0

        if self.args.encoding == "AC":
            if self.full_logits_adapter is None:
                raise RuntimeError(
                    "Native vLLM AC inference was requested without an initialized logits adapter."
                )

            self.full_logits_adapter.clear_capture()
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                detokenize=False,
            )

            _ = self.llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            t0_copy = time.perf_counter()
            full_logits = self.full_logits_adapter.pop_captured_logits(
                expected_rows=len(prompts),
                expected_vocab_size=self.tokenizer.vocab_size,
                expected_prompt_signatures=[prompt_signature(prompt) for prompt in prompts],
            )
            scores = full_logits.index_select(1, self.index_tensor)
            scores = stabilize_ac_logits(scores)
            data_copy_time = time.perf_counter() - t0_copy

            t0_softmax = time.perf_counter()
            probs_cpu = torch.softmax(scores, dim=-1)
            softmax_time = time.perf_counter() - t0_softmax
            return self.tokens_list, probs_cpu, data_copy_time, softmax_time

        elif self.args.encoding in ("bitpacked", "huffman"):
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=-1,
                detokenize=False,
            )

            t0 = time.perf_counter()
            outputs = self.llm.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            _ = time.perf_counter() - t0

            t0_copy = time.perf_counter()
            scores = torch.empty(
                (len(outputs), len(self.tokens_list)),
                dtype=torch.float32,
                device="cpu",
            )
            for row_idx, output in enumerate(outputs):
                completion = output.outputs[0]
                if completion.logprobs is None or len(completion.logprobs) == 0:
                    raise RuntimeError("vLLM did not return next-token logprobs.")
                next_token_logprobs = completion.logprobs[0]
                for col_idx, token_id in enumerate(self.tokens_list):
                    logprob = next_token_logprobs.get(token_id)
                    scores[row_idx, col_idx] = (
                        float("-inf") if logprob is None else logprob.logprob
                    )
            data_copy_time = time.perf_counter() - t0_copy
            return self.tokens_list, scores, data_copy_time, softmax_time
        else:
            raise NotImplementedError(
                f"Encoding method '{self.args.encoding}' is not implemented."
            )

    def run_bulk_rank_inference(self, data_tokens, batch_size):
        """
        Process entire sequences in a single forward pass using prompt_logprobs.

        For rank-based compression (bitpacked/huffman), this is dramatically faster
        than token-by-token inference because vLLM computes logprobs for all
        positions in one pass.

        Args:
            data_tokens (List[int]): Full list of token IDs from the dataset.
            batch_size (int): Number of contiguous chunks to split tokens into.

        Returns:
            Tuple[List[int], float]:
                (rank_list, inference_time)
                rank_list: 0-indexed ranks for each token (interleaved by batch).
                inference_time: wall-clock time for the vLLM generate call.
        """
        # Split tokens into contiguous chunks (same logic as global_mask_compressor).
        chunk_length = len(data_tokens) // batch_size
        extra = len(data_tokens) % batch_size

        chunks = []
        start = 0
        for i in range(batch_size):
            size = chunk_length + (1 if i < extra else 0)
            end = start + size
            chunks.append(data_tokens[start:end])
            start = end

        batches_length = [len(c) for c in chunks]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
        )

        print("Running vLLM bulk rank inference...")
        t0 = time.perf_counter()
        outputs = self.llm.generate(
            chunks,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        inference_time = time.perf_counter() - t0

        # Build rank_list in the same interleaved order as the token-by-token loop:
        # for each token position, iterate over all batches.
        # outputs[i].prompt_logprobs[pos] is a dict {token_id: Logprob} for position pos.
        # Position 0 has no logprobs (no preceding context), so ranks start from position 1.
        # The rank in the Logprob object is 1-indexed; we subtract 1 for 0-indexed.

        # Pre-extract per-batch rank arrays for efficient interleaving.
        batch_ranks = []
        for batch_idx in range(batch_size):
            output = outputs[batch_idx]
            prompt_logprobs = output.prompt_logprobs
            ranks = []
            # prompt_logprobs[0] is None (first token), ranks start at position 1
            for pos in range(1, batches_length[batch_idx]):
                actual_token = chunks[batch_idx][pos]
                logprob_dict = prompt_logprobs[pos]
                if actual_token in logprob_dict:
                    vllm_rank = logprob_dict[actual_token].rank
                    ranks.append(vllm_rank - 1)  # Convert 1-indexed → 0-indexed
                else:
                    # Token not in top prompt_logprobs; rank is at least prompt_logprobs count
                    ranks.append(1)  # fallback — shouldn't happen with prompt_logprobs=1
            batch_ranks.append(ranks)

        # Interleave: for token_idx in range(chunk_length), for batch in range(batch_size)
        rank_list = []
        for token_idx in range(chunk_length):
            for batch_idx in range(batch_size):
                if token_idx < len(batch_ranks[batch_idx]):
                    rank_list.append(batch_ranks[batch_idx][token_idx])

        return rank_list, inference_time

    def detokenize(self, token_ids):
        """Convert a list of token IDs back to a string."""
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        """Get the token ID at a given index in the reduced token list."""
        return self.tokens_list[token_id]

    def count_parameters(self, model=None):
        """Return (total_params, trainable_params) tuple."""
        return self.base_params, 0

    def estimate_model_size_mb(self, model=None):
        """Return (total_size_mb, trainable_size_mb) tuple."""
        return self.base_size_mb, 0.0

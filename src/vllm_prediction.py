"""
vLLM-based token prediction for compression experiments.

This module provides:
- VLLMTokenPredictor: runs batched inference via vLLM with dense logit capture
  for arithmetic coding (AC) and rank-based encoding schemes.
- probe_vllm_ac_support: checks whether the installed vLLM supports the native
  AC path (global logits_processors on LLM).
- CaptureLogitsProcessor: a vLLM v1 LogitsProcessor that captures dense logits
  via POSIX shared memory for retrieval by the host process.
"""

import inspect
import struct
import time
import importlib
import torch
import numpy as np
from multiprocessing import shared_memory
from pyroaring import BitMap

# from src.hf_cache import get_model_cache_dir

try:
    from vllm.v1.sample.logits_processor import LogitsProcessor
    _VLLM_IMPORT_ERROR = None
except Exception as exc:
    LogitsProcessor = object
    _VLLM_IMPORT_ERROR = exc

# Well-known shared memory name used by the processor ↔ host handshake.
_SHM_NAME = "vllm_ac_logits_capture"

# Layout of the shared memory buffer:
#   bytes [0:4]   — uint32 num_rows  (0 = "not ready")
#   bytes [4:8]   — uint32 num_cols  (padded vocab width written by processor)
#   bytes [8: ]   — float32 logits data  (num_rows * num_cols * 4 bytes)
_HEADER_BYTES = 8


def probe_vllm_ac_support(args):
    """
    Check whether the installed vLLM version supports the native AC path.

    Returns:
        tuple: (supported: bool, reason: str | None)
    """
    supported, reason = probe_vllm_backend_support(args)
    if not supported:
        return False, reason

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    try:
        import vllm
        sig = inspect.signature(vllm.LLM.__init__)
        if "logits_processors" not in sig.parameters:
            return False, "vllm.LLM does not accept logits_processors"
    except Exception as e:
        return False, f"failed to inspect vllm.LLM.__init__: {e}"

    return True, None


def probe_vllm_backend_support(args):
    """
    Check whether the installed vLLM runtime can be imported and used.

    Returns:
        tuple: (supported: bool, reason: str | None)
    """
    del args

    try:
        importlib.import_module("vllm")
    except ImportError:
        return False, "vllm is not installed"
    except Exception as exc:
        return False, f"failed to import vllm: {exc}"

    if _VLLM_IMPORT_ERROR is not None:
        return False, f"failed to import vllm internals: {_VLLM_IMPORT_ERROR}"

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    return True, None


class CaptureLogitsProcessor(LogitsProcessor):
    """
    vLLM v1 LogitsProcessor that writes dense next-token logits into
    POSIX shared memory so the host process can read them.

    vLLM v1 runs the engine core in a separate subprocess.  A plain
    Python dict or global variable would be invisible to the host.
    Shared memory solves this without disk I/O.
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self._shm = None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [num_active_seqs, padded_vocab_size]  (GPU tensor)
        rows, cols = logits.shape
        data_bytes = rows * cols * 4  # float32
        total = _HEADER_BYTES + data_bytes

        # Lazily attach to (or create) the shared memory segment.
        if self._shm is None or self._shm.size < total:
            if self._shm is not None:
                self._shm.close()
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
                self._shm = None
            # Try to attach to existing segment; recreate if missing or too small.
            try:
                candidate = shared_memory.SharedMemory(
                    name=_SHM_NAME, create=False
                )
                if candidate.size < total:
                    candidate.close()
                    candidate.unlink()
                    raise FileNotFoundError("too small")
                self._shm = candidate
            except FileNotFoundError:
                self._shm = shared_memory.SharedMemory(
                    name=_SHM_NAME, create=True, size=total
                )

        buf = self._shm.buf
        # Write header: rows, cols
        struct.pack_into("II", buf, 0, rows, cols)
        # Write logits as float32 (CPU copy)
        logits_np = logits.float().cpu().numpy()
        buf[_HEADER_BYTES: _HEADER_BYTES + data_bytes] = logits_np.tobytes()

        return logits  # pass-through — do not modify sampling

    def is_argmax_invariant(self) -> bool:
        # MUST return False so vLLM calls apply() even during greedy decode.
        return False

    def update_state(self, batch_update) -> None:
        pass


class VLLMTokenPredictor:
    """
    Token predictor backed by vLLM for next-token probability extraction.

    V1 (current): processes one prompt at a time through ``llm.generate()``
    to guarantee deterministic row ordering.  vLLM prefix caching provides
    KV reuse across incrementally-growing prompts.
    """

    def __init__(self, args, bitmap_data):
        supported, reason = probe_vllm_backend_support(args)
        if not supported:
            raise ValueError(f"vLLM backend is not available: {reason}")

        from vllm import LLM
        from transformers import AutoTokenizer

        # cache_dir = get_model_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name 
        )
        self.args = args

        tp = getattr(args, "tensor_parallel_size", 1)
        gpu_mem = getattr(args, "gpu_memory_utilization", 0.9)

        # Pre-allocate shared memory for logit capture.
        # vLLM pads vocab to multiples of 64/128; use 2× headroom for safety.
        padded_vocab = (self.tokenizer.vocab_size + 1024) * 2
        shm_size = _HEADER_BYTES + padded_vocab * 4
        # Clean up any stale segment from a previous run.
        try:
            old = shared_memory.SharedMemory(name=_SHM_NAME, create=False)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        self._shm = shared_memory.SharedMemory(
            name=_SHM_NAME, create=True, size=shm_size
        )
        # Zero the header (rows=0 → "not ready").
        struct.pack_into("II", self._shm.buf, 0, 0, 0)

        self.llm = LLM(
            model=args.model_name,
            logits_processors=[CaptureLogitsProcessor],
            # tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            enforce_eager=True,
            enable_prefix_caching=True,
            max_model_len=args.context_length,
            max_num_seqs=args.batch_size,
        )

        # Expose parameter counts (approximate, from HF config).
        self._estimate_params_from_config(args.model_name)

        # Reconstruct allowed-token list from bitmap.
        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.device = torch.device("cuda")
        self.index_tensor = torch.tensor(
            self.tokens_list, dtype=torch.long, device=self.device
        )
        self.reduce_tokens = args.reduce_tokens

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """
        Run one-step inference for each prompt and return scores.

        V1: processes prompts sequentially (one ``llm.generate`` call each)
        to guarantee captured-logit row order matches prompt order.

        Returns the same 4-tuple as TokenPredictor.run_batched_inference:
            (tokens_list, scores, data_copy_time, softmax_time)
        """
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=1, temperature=0)

        all_logits = []
        data_copy_time = 0.0

        for prompt_tokens in prompts:
            # Reset the shared-memory header (rows=0 → "not ready").
            struct.pack_into("II", self._shm.buf, 0, 0, 0)

            # Run one-prompt generate.  vLLM prefix caching handles KV reuse.
            # PromptType accepts list[int] as token IDs directly.
            self.llm.generate(
                prompts=[prompt_tokens],
                sampling_params=sp,
                use_tqdm=False,
            )

            # Read captured logits from shared memory.
            t0 = time.perf_counter()
            rows, cols = struct.unpack_from("II", self._shm.buf, 0)
            if rows == 0:
                raise RuntimeError(
                    "CaptureLogitsProcessor did not fire. "
                    "Ensure is_argmax_invariant() returns False."
                )

            data_bytes = rows * cols * 4
            raw = bytes(self._shm.buf[_HEADER_BYTES: _HEADER_BYTES + data_bytes])
            logits_np = np.frombuffer(raw, dtype=np.float32).reshape(rows, cols)
            logits_row = torch.from_numpy(logits_np.copy()).to(self.device)
            data_copy_time += time.perf_counter() - t0

            all_logits.append(logits_row)

        # Stack: [batch_size, padded_vocab]
        logits = torch.cat(all_logits, dim=0)

        # Slice off vLLM padding columns beyond the real vocabulary.
        vocab_size = self.tokenizer.vocab_size
        if logits.shape[1] > vocab_size:
            logits = logits[:, :vocab_size]

        # Optionally reduce to allowed-token subset.
        if self.reduce_tokens:
            logits = logits.index_select(1, self.index_tensor)

        softmax_time = 0.0
        if self.args.encoding == "AC":
            t0 = time.perf_counter()
            probs = torch.softmax(logits.float(), dim=-1)
            softmax_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            probs_cpu = probs.cpu()
            data_copy_time += time.perf_counter() - t0

            return self.tokens_list, probs_cpu, data_copy_time, softmax_time
        elif self.args.encoding in ("bitpacked", "huffman"):
            return self.tokens_list, logits, data_copy_time, softmax_time
        else:
            raise NotImplementedError(
                f"Encoding method '{self.args.encoding}' is not implemented."
            )


    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]

    def _get_distinct_tokens(self):
        return self.tokens_list

    def cleanup(self):
        """Release shared memory resources."""
        if hasattr(self, "_shm") and self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None

    def __del__(self):
        self.cleanup()

    def _estimate_params_from_config(self, model_name):
        """Rough parameter estimate from HF config (avoids loading full weights)."""
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name)
            # Most HF configs don't expose param count directly;
            # use a rough formula for transformer decoder-only models.
            h = getattr(cfg, "hidden_size", 0)
            n_layers = getattr(cfg, "num_hidden_layers", 0)
            v = getattr(cfg, "vocab_size", 0)
            # ~12 * h^2 per layer + vocab embeddings
            estimated = n_layers * 12 * h * h + v * h
            self.base_params = estimated
            # Assume bf16 → 2 bytes per param
            self.base_size_mb = estimated * 2 / (1024 ** 2)
        except Exception:
            self.base_params = 0
            self.base_size_mb = 0.0
        self.adapter_params = 0
        self.adapter_size_mb = 0.0

"""
Batched vLLM token prediction for compression experiments.

The predictor keeps the same public contract as ``TokenPredictor`` while using
vLLM's offline batching API.  Each compression step submits the entire prompt
batch in one ``LLM.generate`` call and asks vLLM for the full next-token score
vector, avoiding the sequential shared-memory logits-capture path.
"""

import importlib
import os
import struct
import time
import uuid
from multiprocessing import shared_memory
from typing import Any

import torch
from pyroaring import BitMap

try:
    from vllm.v1.sample.logits_processor import LogitsProcessor as _VLLMLogitsProcessor
except Exception:
    _VLLMLogitsProcessor = object

_SHM_ENV_VAR = "COMPRESSION_ECONOMICS_VLLM_LOGITS_SHM"
_VOCAB_SHM_ENV_VAR = "COMPRESSION_ECONOMICS_VLLM_VOCAB_SHM"
_VOCAB_LEN_ENV_VAR = "COMPRESSION_ECONOMICS_VLLM_VOCAB_LEN"
_SHM_HEADER_BYTES = 24


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

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    return True, None


def probe_vllm_ac_support(args):
    """Backward-compatible alias for callers probing vLLM AC support."""
    return probe_vllm_backend_support(args)


class BatchedLogitsCaptureProcessor(_VLLMLogitsProcessor):
    """
    vLLM batch logits processor that copies dense logits to shared memory.

    Request order is recovered from ``SamplingParams.extra_args["ce_row_id"]``.
    The processor is instantiated inside vLLM's worker process, so shared memory
    is used as the handoff back to this process.
    """

    _ROW_ID_ARG = "ce_row_id"
    _TARGET_IDS_ARG = "ce_target_token_ids"

    @classmethod
    def validate_params(cls, sampling_params):
        extra_args = sampling_params.extra_args or {}
        row_id = extra_args.get(cls._ROW_ID_ARG)
        if row_id is not None and not isinstance(row_id, int):
            raise ValueError(f"{cls._ROW_ID_ARG} must be an int, got {type(row_id)}")
        target_token_ids = extra_args.get(cls._TARGET_IDS_ARG)
        if target_token_ids is not None and not isinstance(target_token_ids, list):
            raise ValueError(
                f"{cls._TARGET_IDS_ARG} must be a list[int], got {type(target_token_ids)}"
            )

    def __init__(self, vllm_config, device, is_pin_memory):
        del vllm_config, device, is_pin_memory
        from vllm.v1.sample.logits_processor.builtin import process_dict_updates

        self._process_dict_updates = process_dict_updates
        self.req_info: dict[int, dict[str, Any]] = {}
        self._shm = None
        self._shm_name = os.environ.get(_SHM_ENV_VAR)
        if not self._shm_name:
            raise RuntimeError(f"{_SHM_ENV_VAR} is not set")
        self._vocab_shm = None
        self._capture_token_ids = None
        self._vocab_shm_name = os.environ.get(_VOCAB_SHM_ENV_VAR)
        self._vocab_len = int(os.environ.get(_VOCAB_LEN_ENV_VAR, "0"))

    def is_argmax_invariant(self) -> bool:
        # Return False so vLLM calls apply() even for greedy sampling.
        return False

    def update_state(self, batch_update):
        def extract_request_info(params, prompt_token_ids, output_token_ids):
            del prompt_token_ids
            self.validate_params(params)
            extra_args = params.extra_args or {}
            row_id = extra_args.get(self._ROW_ID_ARG)
            if row_id is None:
                return None
            return {
                "row_id": row_id,
                "target_token_ids": extra_args.get(self._TARGET_IDS_ARG),
                "output_token_ids": output_token_ids,
            }

        self._process_dict_updates(self.req_info, batch_update, extract_request_info)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits

        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)

        rows, _ = logits.shape
        header = self._shm.buf[:_SHM_HEADER_BYTES]
        (
            expected_rows,
            expected_steps,
            max_cols,
            _captured_cols,
            ready_count,
            _reserved,
        ) = struct.unpack_from("IIIIII", header, 0)

        capture_logits = self._select_capture_logits(logits)
        cols = capture_logits.shape[1]
        if cols > max_cols:
            raise RuntimeError(
                f"Captured logits have {cols} columns, but shared memory only "
                f"has capacity for {max_cols}"
            )

        flags_offset = _SHM_HEADER_BYTES
        flag_count = expected_rows * expected_steps
        data_offset = flags_offset + flag_count * 4
        cpu_logits = capture_logits.detach().to(dtype=torch.float32, device="cpu")

        ready = ready_count
        for batch_idx, info in self.req_info.items():
            row_id = info["row_id"]
            output_token_ids = info["output_token_ids"]
            step_id = len(output_token_ids)
            if (
                batch_idx >= rows
                or row_id is None
                or row_id >= expected_rows
                or step_id >= expected_steps
            ):
                continue

            flat_index = row_id * expected_steps + step_id
            row_start = data_offset + (flat_index * max_cols * 4)
            row_end = row_start + (cols * 4)
            self._shm.buf[row_start:row_end] = cpu_logits[batch_idx].numpy().tobytes()

            flag_offset = flags_offset + flat_index * 4
            was_ready = struct.unpack_from("I", self._shm.buf, flag_offset)[0]
            if not was_ready:
                struct.pack_into("I", self._shm.buf, flag_offset, 1)
                ready += 1

            target_token_ids = info.get("target_token_ids")
            if target_token_ids is not None and step_id < len(target_token_ids):
                target_token_id = int(target_token_ids[step_id])
                logits[batch_idx].fill_(float("-inf"))
                logits[batch_idx, target_token_id] = 0.0

        struct.pack_into("II", header, 12, cols, ready)
        return logits

    def _select_capture_logits(self, logits):
        if not self._vocab_shm_name or self._vocab_len <= 0:
            return logits

        if self._capture_token_ids is None:
            self._vocab_shm = shared_memory.SharedMemory(
                name=self._vocab_shm_name, create=False
            )
            token_ids = torch.frombuffer(
                self._vocab_shm.buf,
                dtype=torch.int64,
                count=self._vocab_len,
            ).clone()
            self._capture_token_ids = token_ids.to(logits.device)

        return logits.index_select(1, self._capture_token_ids)


class VLLMTokenPredictor:
    """
    Token predictor backed by vLLM for dense next-token score extraction.

    vLLM owns KV-cache management internally.  When ``enable_prefix_caching`` is
    enabled, repeated incremental prompts can reuse cached prefixes while still
    being submitted as a full batch on every compression step.
    """

    def __init__(self, args, bitmap_data):
        supported, reason = probe_vllm_backend_support(args)
        if not supported:
            raise ValueError(f"vLLM backend is not available: {reason}")

        if getattr(args, "lora_path", None) is not None:
            raise NotImplementedError(
                "--engine vllm does not currently support --lora_path in this project."
            )

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, cache_dir=".cache"
        )
        self.device = torch.device("cuda")

        gpu_mem = getattr(args, "gpu_memory_utilization", 0.9)
        tensor_parallel_size = getattr(args, "tensor_parallel_size", 1)
        enable_prefix_caching = bool(getattr(args, "use_kv_cache", True))
        self.vocab_size = self._get_config_vocab_size(
            args.model_name, self.tokenizer.vocab_size
        )
        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.vocab_size))

        self.max_batch_size = args.batch_size
        self.max_window_size = max(1, int(getattr(args, "vllm_window_size", 1)))
        self.capture_vocab_size = (
            len(self.tokens_list) if args.reduce_tokens else self.vocab_size
        )
        self.max_vocab_cols = self._estimate_padded_vocab_size(self.capture_vocab_size)
        self._shm_name = f"ce_vllm_logits_{os.getpid()}_{uuid.uuid4().hex}"
        shm_size = (
            _SHM_HEADER_BYTES
            + self.max_batch_size * self.max_window_size * 4
            + self.max_batch_size * self.max_window_size * self.max_vocab_cols * 4
        )
        self._shm = shared_memory.SharedMemory(
            name=self._shm_name, create=True, size=shm_size
        )
        os.environ[_SHM_ENV_VAR] = self._shm_name

        self._vocab_shm = None
        self._vocab_shm_name = None
        if args.reduce_tokens:
            vocab_tensor = torch.tensor(self.tokens_list, dtype=torch.int64)
            self._vocab_shm_name = f"ce_vllm_vocab_{os.getpid()}_{uuid.uuid4().hex}"
            self._vocab_shm = shared_memory.SharedMemory(
                name=self._vocab_shm_name,
                create=True,
                size=vocab_tensor.numel() * vocab_tensor.element_size(),
            )
            self._vocab_shm.buf[: vocab_tensor.numel() * 8] = vocab_tensor.numpy().tobytes()
            os.environ[_VOCAB_SHM_ENV_VAR] = self._vocab_shm_name
            os.environ[_VOCAB_LEN_ENV_VAR] = str(vocab_tensor.numel())
        else:
            os.environ.pop(_VOCAB_SHM_ENV_VAR, None)
            os.environ.pop(_VOCAB_LEN_ENV_VAR, None)

        self.llm = LLM(
                    model=args.model_name,
                    tokenizer=args.model_name,
                    gpu_memory_utilization=gpu_mem,
                    tensor_parallel_size=tensor_parallel_size,
                    enable_prefix_caching=enable_prefix_caching,
                    max_model_len=args.context_length,
                    max_num_seqs=args.batch_size,
                    attention_config={"backend": "FLASHINFER"},           # <-- new
                    logits_processors=[BatchedLogitsCaptureProcessor],
                )

        self.sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            detokenize=False,
            ignore_eos=True,
        )

        actual_vocab_size = self._get_vocab_size()
        if not args.reduce_tokens and actual_vocab_size != self.vocab_size:
            self.vocab_size = actual_vocab_size
            self.tokens_list = list(range(self.vocab_size))
            self.capture_vocab_size = self.vocab_size
        else:
            self.vocab_size = actual_vocab_size
        self.index_tensor = torch.tensor(
            self.tokens_list, dtype=torch.long, device=self.device
        )
        self.reduce_tokens = args.reduce_tokens

        self._estimate_params_from_config(args.model_name)

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """
        Run one vLLM batched generation step and return next-token scores.

        Args:
            prompts: list[list[int]] tokenized prompts.
            enable_kv_cache: Kept for interface compatibility. vLLM prefix
                caching is configured at engine construction time.
        """
        del enable_kv_cache

        request_prompts = [
            {"prompt_token_ids": list(prompt)}
            for prompt in prompts
        ]

        self._reset_capture_buffer(len(prompts), 1)

        sampling_params = [
            self.sampling_params.clone()
            for _ in prompts
        ]
        for row_id, params in enumerate(sampling_params):
            extra_args = dict(params.extra_args or {})
            extra_args[BatchedLogitsCaptureProcessor._ROW_ID_ARG] = row_id
            params.extra_args = extra_args

        self.llm.generate(
            request_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        t0 = time.perf_counter()
        logits = self._read_captured_logits(len(prompts), 1, squeeze=True)
        data_copy_time = time.perf_counter() - t0

        softmax_time = 0.0
        if self.args.encoding in {"AC", "PMATIC"}:
            t0 = time.perf_counter()
            probs = torch.softmax(logits.float(), dim=-1)
            softmax_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            probs_cpu = probs.cpu()
            data_copy_time += time.perf_counter() - t0
            return self.tokens_list, probs_cpu, data_copy_time, softmax_time

        if self.args.encoding in {"bitpacked", "huffman"}:
            return self.tokens_list, logits, data_copy_time, softmax_time

        raise NotImplementedError(
            f"Encoding method '{self.args.encoding}' is not implemented."
        )

    def run_batched_forced_inference(self, prompts, target_token_windows):
        from vllm import SamplingParams

        if len(prompts) != len(target_token_windows):
            raise ValueError("prompts and target_token_windows must have equal length")
        if not prompts:
            raise ValueError("prompts must not be empty")

        expected_steps = max(len(row) for row in target_token_windows)
        if expected_steps <= 0:
            raise ValueError("target_token_windows must contain at least one token")
        if expected_steps > self.max_window_size:
            raise ValueError(
                f"Forced window has {expected_steps} steps, but predictor was "
                f"initialized for vllm_window_size={self.max_window_size}"
            )

        request_prompts = [
            {"prompt_token_ids": list(prompt)}
            for prompt in prompts
        ]
        self._reset_capture_buffer(len(prompts), expected_steps)

        sampling_params = []
        for row_id, target_token_ids in enumerate(target_token_windows):
            params = SamplingParams(
                max_tokens=expected_steps,
                min_tokens=expected_steps,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                detokenize=False,
                ignore_eos=True,
                extra_args={
                    BatchedLogitsCaptureProcessor._ROW_ID_ARG: row_id,
                    BatchedLogitsCaptureProcessor._TARGET_IDS_ARG: list(target_token_ids),
                },
            )
            sampling_params.append(params)

        self.llm.generate(
            request_prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        t0 = time.perf_counter()
        logits = self._read_captured_logits(
            len(prompts), expected_steps, squeeze=False
        )
        data_copy_time = time.perf_counter() - t0

        softmax_time = 0.0
        if self.args.encoding in {"AC", "PMATIC"}:
            t0 = time.perf_counter()
            probs = torch.softmax(logits.float(), dim=-1)
            softmax_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            probs_cpu = probs.cpu()
            data_copy_time += time.perf_counter() - t0
            return self.tokens_list, probs_cpu, data_copy_time, softmax_time

        if self.args.encoding in {"bitpacked", "huffman"}:
            return self.tokens_list, logits, data_copy_time, softmax_time

        raise NotImplementedError(
            f"Encoding method '{self.args.encoding}' is not implemented."
        )

    def _reset_capture_buffer(self, expected_rows, expected_steps):
        if expected_rows > self.max_batch_size:
            raise ValueError(
                f"Got {expected_rows} prompts, but vLLM predictor was initialized "
                f"for batch_size={self.max_batch_size}"
            )
        if expected_steps > self.max_window_size:
            raise ValueError(
                f"Got {expected_steps} steps, but vLLM predictor was initialized "
                f"for vllm_window_size={self.max_window_size}"
            )
        struct.pack_into(
            "IIIIII",
            self._shm.buf,
            0,
            expected_rows,
            expected_steps,
            self.max_vocab_cols,
            0,
            0,
            0,
        )
        flags_offset = _SHM_HEADER_BYTES
        flag_bytes = expected_rows * expected_steps * 4
        self._shm.buf[flags_offset: flags_offset + flag_bytes] = (
            b"\x00" * flag_bytes
        )

    def _read_captured_logits(self, expected_rows, expected_steps, squeeze):
        (
            rows,
            steps,
            max_cols,
            cols,
            ready_count,
            _reserved,
        ) = struct.unpack_from("IIIIII", self._shm.buf, 0)
        expected_count = expected_rows * expected_steps
        if rows != expected_rows or steps != expected_steps or ready_count != expected_count:
            raise RuntimeError(
                f"Captured logits are incomplete: expected {expected_rows}x"
                f"{expected_steps} rows, header has rows={rows}, steps={steps}, "
                f"ready_count={ready_count}"
            )
        if cols <= 0:
            raise RuntimeError("Captured logits header did not record a vocab width")

        flags_offset = _SHM_HEADER_BYTES
        flag_count = rows * steps
        data_offset = flags_offset + flag_count * 4
        tensors = []
        for step_id in range(steps):
            step_rows = []
            for row_id in range(rows):
                flat_index = row_id * steps + step_id
                flag = struct.unpack_from(
                    "I", self._shm.buf, flags_offset + flat_index * 4
                )[0]
                if not flag:
                    raise RuntimeError(
                        f"Missing captured logits for row {row_id}, step {step_id}"
                    )
                row_start = data_offset + flat_index * max_cols * 4
                row_end = row_start + cols * 4
                row = torch.frombuffer(
                    self._shm.buf[row_start:row_end],
                    dtype=torch.float32,
                ).clone()
                step_rows.append(row[: self.capture_vocab_size])
            tensors.append(torch.stack(step_rows, dim=0))

        result = torch.stack(tensors, dim=0).to(self.device)
        if squeeze:
            return result[0]
        return result

    def _outputs_to_dense_scores(self, outputs, expected_rows):
        if len(outputs) != expected_rows:
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {expected_rows} prompts"
            )

        rows = []
        for output in outputs:
            if not output.outputs:
                raise RuntimeError("vLLM output did not contain a completion")
            logprobs = output.outputs[0].logprobs
            if not logprobs:
                raise RuntimeError(
                    "vLLM did not return next-token scores. "
                    "Expected SamplingParams(logprobs=-1)."
                )
            rows.append(self._position_logprobs_to_dense(logprobs, position=0))

        return torch.stack(rows, dim=0).to(self.device)

    def _position_logprobs_to_dense(self, logprobs: Any, position: int):
        """
        Convert one returned vLLM logprob/logit position to a dense vocab row.

        Supports both ``FlatLogprobs`` and the older list-of-dicts shape.  vLLM
        may return raw logits when ``logprobs_mode='raw_logits'``; callers treat
        the result as generic scores.
        """
        row = torch.full(
            (self.vocab_size,), float("-inf"), dtype=torch.float32
        )

        if hasattr(logprobs, "start_indices") and hasattr(logprobs, "logprobs"):
            start = logprobs.start_indices[position]
            end = logprobs.end_indices[position]
            values = logprobs.logprobs[start:end]
            token_ids = logprobs.token_ids[start:end]

            # Some vLLM paths represent full-vocab logprobs as values only.
            if not token_ids and len(values) >= self.vocab_size:
                row.copy_(torch.tensor(values[: self.vocab_size], dtype=torch.float32))
                return row

            for token_id, value in zip(token_ids, values):
                if 0 <= token_id < self.vocab_size:
                    row[token_id] = float(value)
            return row

        position_logprobs = logprobs[position]
        for token_id, logprob_obj in position_logprobs.items():
            if 0 <= token_id < self.vocab_size:
                row[token_id] = float(logprob_obj.logprob)
        return row

    def _get_vocab_size(self):
        model_config = getattr(getattr(self.llm, "llm_engine", None), "model_config", None)
        if model_config is not None and hasattr(model_config, "get_vocab_size"):
            return int(model_config.get_vocab_size())
        return int(self.tokenizer.vocab_size)

    @staticmethod
    def _estimate_padded_vocab_size(vocab_size):
        return ((int(vocab_size) + 8191) // 8192 + 1) * 8192

    @staticmethod
    def _get_config_vocab_size(model_name, fallback):
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_name, cache_dir=".cache")
            return max(int(fallback), int(getattr(cfg, "vocab_size", fallback)))
        except Exception:
            return int(fallback)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]

    def _get_distinct_tokens(self):
        return self.tokens_list

    def cleanup(self):
        shutdown = getattr(self.llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
        if getattr(self, "_shm", None) is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None
        if os.environ.get(_SHM_ENV_VAR) == getattr(self, "_shm_name", None):
            os.environ.pop(_SHM_ENV_VAR, None)
        if getattr(self, "_vocab_shm", None) is not None:
            self._vocab_shm.close()
            try:
                self._vocab_shm.unlink()
            except FileNotFoundError:
                pass
            self._vocab_shm = None
        if os.environ.get(_VOCAB_SHM_ENV_VAR) == getattr(self, "_vocab_shm_name", None):
            os.environ.pop(_VOCAB_SHM_ENV_VAR, None)
            os.environ.pop(_VOCAB_LEN_ENV_VAR, None)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    def _estimate_params_from_config(self, model_name):
        """Rough parameter estimate from HF config without loading weights twice."""
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_name, cache_dir=".cache")
            h = getattr(cfg, "hidden_size", 0)
            n_layers = getattr(cfg, "num_hidden_layers", 0)
            v = getattr(cfg, "vocab_size", 0)
            estimated = n_layers * 12 * h * h + v * h
            self.base_params = estimated
            self.base_size_mb = estimated * 2 / (1024 ** 2)
        except Exception:
            self.base_params = 0
            self.base_size_mb = 0.0
        self.adapter_params = 0
        self.adapter_size_mb = 0.0

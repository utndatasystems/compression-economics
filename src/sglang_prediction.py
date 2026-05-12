"""
SGlang-based token prediction for compression experiments.

This module provides:
- SGLangTokenPredictor: runs batched inference via SGlang's embedded engine API
  using per-request token logprob probes for arithmetic coding (AC) and rank-based
  encoding schemes.
- probe_sglang_ac_support: checks whether the installed SGlang runtime can be used
  as an embedded GPU backend in the current environment.
"""

import time
from importlib import metadata

import torch
from pyroaring import BitMap

from src.hf_cache import get_model_cache_dir


def _parse_version(version_str):
    parts = []
    for token in version_str.replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def probe_sglang_ac_support(args):
    """
    Check whether the installed SGlang version supports the predictor contract.

    Returns:
        tuple: (supported: bool, reason: str | None)
    """
    try:
        import sglang as sgl  # noqa: F401
    except ImportError:
        return False, "sglang is not installed"

    try:
        version = metadata.version("sglang")
    except metadata.PackageNotFoundError:
        return False, "sglang package metadata is unavailable"

    if _parse_version(version) < (0, 5, 9):
        return False, (
            f"sglang {version} is too old for this repo; install sglang>=0.5.9 "
            "to avoid the AutoImageProcessor.register compatibility error"
        )

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    return True, None


class SGLangTokenPredictor:
    """
    Token predictor backed by SGlang's embedded Engine API.

    SGlang exposes next-token logprobs for arbitrary token ID probe lists through
    `token_ids_logprob`. We query the active token set directly and reconstruct the
    tensor returned by the existing predictor interface from those per-token values.
    This keeps compression and decompression engine-agnostic, at the cost of a
    heavier decode path when the active token set is large.
    """

    def __init__(self, args, bitmap_data):
        import sglang as sgl
        from transformers import AutoTokenizer

        cache_dir = get_model_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=cache_dir,
        )
        self.args = args
        self.device = torch.device("cuda")
        self._use_streaming_session_kv = bool(
            getattr(args, "sglang_use_streaming_session_kv", False)
        )
        self._session_timeout = getattr(args, "sglang_session_timeout", None)
        self._session_capacity = max(
            int(getattr(args, "context_length", 0) or 0),
            int(getattr(args, "retain_tokens", 0) or 0),
            1,
        )
        self._session_lanes = []
        self._session_batch_size = None

        tp = getattr(args, "tensor_parallel_size", 1)
        mem_fraction = getattr(args, "sglang_mem_fraction_static", 0.8)
        deterministic = getattr(args, "sglang_enable_deterministic_inference", True)

        try:
            self.engine = sgl.Engine(
                model_path=args.model_name,
                skip_tokenizer_init=True,
                tp_size=tp,
                mem_fraction_static=mem_fraction,
                enable_deterministic_inference=deterministic,
            )
        except TypeError as exc:
            if "AutoImageProcessor.register() got multiple values for argument 'exist_ok'" in str(exc):
                raise RuntimeError(
                    "Incompatible SGlang/Transformers environment detected. "
                    "This repo expects sglang>=0.5.9 with the current Transformers version. "
                    "Upgrade SGlang in the active environment and retry."
                ) from exc
            raise

        self._estimate_params_from_config(args.model_name, cache_dir)

        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.reduce_tokens = args.reduce_tokens
        self._probe_token_ids = [int(token_id) for token_id in self.tokens_list]

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """
        Run one-step SGlang inference and return the same 4-tuple as other backends.
        """
        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        }

        if self._should_use_streaming_session_kv(enable_kv_cache):
            outputs = self._generate_with_streaming_sessions(prompts, sampling_params)
        else:
            self._reset_streaming_sessions()
            token_ids_logprob = [self._probe_token_ids for _ in prompts]
            outputs = self.engine.generate(
                input_ids=prompts,
                sampling_params=sampling_params,
                return_logprob=True,
                top_logprobs_num=0,
                token_ids_logprob=token_ids_logprob,
                lora_path=getattr(self.args, "lora_path", None),
            )

        if isinstance(outputs, dict):
            outputs = [outputs]

        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"SGlang returned {len(outputs)} outputs for {len(prompts)} prompts."
            )

        t0_data_copy = time.perf_counter()
        rows = [self._extract_requested_logprobs(output) for output in outputs]
        logprob_tensor = torch.tensor(rows, dtype=torch.float32)
        data_copy_time = time.perf_counter() - t0_data_copy

        softmax_time = 0.0
        if self.args.encoding == "AC":
            t0_softmax = time.perf_counter()
            probs = torch.softmax(logprob_tensor, dim=-1)
            softmax_time = time.perf_counter() - t0_softmax
            return self.tokens_list, probs, data_copy_time, softmax_time
        if self.args.encoding in ("bitpacked", "huffman"):
            return self.tokens_list, logprob_tensor.to(self.device), data_copy_time, softmax_time

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
        self._reset_streaming_sessions()
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def __del__(self):
        self.cleanup()

    def _extract_requested_logprobs(self, output):
        meta = output.get("meta_info", {})
        output_token_ids_logprobs = meta.get("output_token_ids_logprobs")
        if not output_token_ids_logprobs:
            raise RuntimeError(
                "SGlang did not return output_token_ids_logprobs for the probed token set."
            )

        first_step = output_token_ids_logprobs[0]
        if len(first_step) != len(self._probe_token_ids):
            raise RuntimeError(
                "SGlang returned a different number of token logprobs than requested: "
                f"expected {len(self._probe_token_ids)}, got {len(first_step)}"
            )

        parsed_by_token_id = {}
        ordered_values = []
        for item in first_step:
            if isinstance(item, dict):
                value = item.get("logprob")
                token_id = item.get("token_id")
            elif isinstance(item, (list, tuple)) and item:
                value = item[0]
                token_id = item[1] if len(item) > 1 else None
            else:
                value = None
                token_id = None

            if value is None:
                raise RuntimeError(f"Unexpected SGlang token logprob payload: {item!r}")

            if token_id is None:
                ordered_values.append(float(value))
            else:
                parsed_by_token_id[int(token_id)] = float(value)

        if parsed_by_token_id:
            missing = [token_id for token_id in self._probe_token_ids if token_id not in parsed_by_token_id]
            if missing:
                raise RuntimeError(
                    "SGlang did not return logprobs for all requested token IDs; "
                    f"missing {len(missing)} entries."
                )
            return [parsed_by_token_id[token_id] for token_id in self._probe_token_ids]

        return ordered_values

    def _estimate_params_from_config(self, model_name, cache_dir):
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            hidden_size = getattr(cfg, "hidden_size", 0)
            num_layers = getattr(cfg, "num_hidden_layers", 0)
            vocab_size = getattr(cfg, "vocab_size", 0)
            estimated = num_layers * 12 * hidden_size * hidden_size + vocab_size * hidden_size
            self.base_params = estimated
            self.base_size_mb = estimated * 2 / (1024 ** 2)
        except Exception:
            self.base_params = 0
            self.base_size_mb = 0.0
        self.adapter_params = 0
        self.adapter_size_mb = 0.0

    def _should_use_streaming_session_kv(self, enable_kv_cache):
        return (
            enable_kv_cache
            and self._use_streaming_session_kv
            and hasattr(self.engine, "open_session")
            and hasattr(self.engine, "close_session")
        )

    def _generate_with_streaming_sessions(self, prompts, sampling_params):
        self._ensure_streaming_session_lanes(len(prompts))

        outputs = []
        for lane_idx, prompt in enumerate(prompts):
            input_ids, session_params = self._prepare_streaming_request(lane_idx, prompt)
            output = self.engine.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=True,
                top_logprobs_num=0,
                token_ids_logprob=self._probe_token_ids,
                lora_path=getattr(self.args, "lora_path", None),
                session_params=session_params,
            )
            if isinstance(output, list):
                if len(output) != 1:
                    raise RuntimeError(
                        "SGlang streaming-session call returned more than one output "
                        f"for lane {lane_idx}."
                    )
                output = output[0]
            self._session_lanes[lane_idx]["last_prompt"] = [int(token_id) for token_id in prompt]
            outputs.append(output)

        return outputs

    def _ensure_streaming_session_lanes(self, batch_size):
        if self._session_batch_size == batch_size:
            return

        self._reset_streaming_sessions()
        self._session_lanes = [self._new_streaming_session_lane() for _ in range(batch_size)]
        self._session_batch_size = batch_size

    def _prepare_streaming_request(self, lane_idx, prompt):
        prompt_ids = [int(token_id) for token_id in prompt]
        lane = self._session_lanes[lane_idx]

        if lane["session_id"] is None:
            lane["session_id"] = self._open_streaming_session()
            return prompt_ids, {"id": lane["session_id"]}

        previous_prompt = lane["last_prompt"]
        if self._is_prefix_extension(previous_prompt, prompt_ids):
            delta_ids = prompt_ids[len(previous_prompt):]
            return delta_ids, {"id": lane["session_id"]}

        self._reset_streaming_session_lane(lane)
        lane["session_id"] = self._open_streaming_session()
        return prompt_ids, {"id": lane["session_id"]}

    def _open_streaming_session(self):
        return self.engine.open_session(
            capacity_of_str_len=self._session_capacity,
            streaming=True,
            timeout=self._session_timeout,
        )

    def _reset_streaming_sessions(self):
        for lane in self._session_lanes:
            self._reset_streaming_session_lane(lane)
        self._session_lanes = []
        self._session_batch_size = None

    def _reset_streaming_session_lane(self, lane):
        session_id = lane.get("session_id")
        if session_id is not None:
            try:
                self.engine.close_session(session_id)
            except Exception:
                pass
        lane["session_id"] = None
        lane["last_prompt"] = None

    @staticmethod
    def _new_streaming_session_lane():
        return {"session_id": None, "last_prompt": None}

    @staticmethod
    def _is_prefix_extension(previous_prompt, prompt):
        return (
            previous_prompt is not None
            and len(prompt) > len(previous_prompt)
            and prompt[: len(previous_prompt)] == previous_prompt
        )
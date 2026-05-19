"""
TensorRT-LLM-based token prediction for compression experiments.

This module provides:
- TensorRTTokenPredictor: runs batched inference against a prebuilt TensorRT-LLM
  engine directory and returns next-token scores through the existing predictor
  contract.
- probe_tensorrt_ac_support: checks whether TensorRT-LLM can be used in the
  current environment with the supplied engine directory.
"""

import inspect
import os
import time

import torch
from pyroaring import BitMap

from src.hf_cache import get_model_cache_dir, resolve_pretrained_model_source


def probe_tensorrt_ac_support(args):
    """
    Check whether TensorRT-LLM is available for the predictor contract.

    Returns:
        tuple: (supported: bool, reason: str | None)
    """
    try:
        from tensorrt_llm.runtime import ModelRunner  # noqa: F401
    except ImportError:
        return False, "tensorrt_llm is not installed"

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    engine_dir = getattr(args, "tensorrt_engine_dir", None)
    if not engine_dir:
        return False, "tensorrt_engine_dir is required"

    if not os.path.isdir(engine_dir):
        return False, f"TensorRT engine directory not found: {engine_dir}"

    return True, None


def _select_supported_kwargs(factory, kwargs):
    """Return only keyword arguments accepted by a callable, if introspection works."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return kwargs

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs

    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def _last_step_from_tensor(logits, prompt_lengths=None):
    """Normalize TensorRT-LLM logit outputs to a [batch, vocab] tensor."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor logits, got {type(logits)!r}")

    if logits.ndim == 2:
        return logits

    if logits.ndim == 3:
        if prompt_lengths is None:
            return logits[:, -1, :]

        rows = []
        for batch_index, prompt_length in enumerate(prompt_lengths):
            step_index = max(int(prompt_length) - 1, 0)
            step_index = min(step_index, logits.shape[1] - 1)
            rows.append(logits[batch_index, step_index, :])
        return torch.stack(rows, dim=0)

    if logits.ndim == 4:
        return logits[:, -1, -1, :]

    raise RuntimeError(f"Unsupported TensorRT logits rank: {logits.ndim}")


def extract_last_step_logits(output_payload, prompt_lengths=None):
    """
    Extract a [batch, vocab] tensor from TensorRT-LLM generate output payloads.
    """
    if isinstance(output_payload, torch.Tensor):
        return _last_step_from_tensor(output_payload, prompt_lengths)

    if isinstance(output_payload, dict):
        for key in ("generation_logits", "context_logits"):
            if key in output_payload and output_payload[key] is not None:
                logits = output_payload[key]
                return extract_last_step_logits(logits, prompt_lengths)

        raise RuntimeError(
            "TensorRT-LLM generate() did not return generation_logits or context_logits."
        )

    if isinstance(output_payload, (list, tuple)):
        if not output_payload:
            raise RuntimeError("TensorRT-LLM generate() returned an empty logits payload.")

        if all(isinstance(item, torch.Tensor) for item in output_payload):
            normalized = []
            for item in output_payload:
                squeezed = item.squeeze()
                if squeezed.ndim == 1:
                    normalized.append(squeezed)
                else:
                    normalized.append(_last_step_from_tensor(item).squeeze(0))
            return torch.stack(normalized, dim=0)

        if len(output_payload) == 1:
            return extract_last_step_logits(output_payload[0], prompt_lengths)

    raise RuntimeError(
        "Unsupported TensorRT-LLM logits payload type: "
        f"{type(output_payload)!r}"
    )


class TensorRTTokenPredictor:
    """Token predictor backed by a prebuilt TensorRT-LLM engine directory."""

    def __init__(self, args, bitmap_data):
        from transformers import AutoTokenizer

        supported, reason = probe_tensorrt_ac_support(args)
        if not supported:
            raise ValueError(f"TensorRT backend is not available: {reason}")

        if getattr(args, "lora_path", None) is not None:
            raise ValueError(
                "TensorRT backend does not yet support LoRA adapters in this repo."
            )

        cache_dir = get_model_cache_dir()
        model_source = resolve_pretrained_model_source(args.model_name)
        if os.path.isdir(model_source):
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                cache_dir=cache_dir,
            )
        self.args = args
        self.device = torch.device("cuda")
        self.engine_dir = args.tensorrt_engine_dir

        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.pad_token_id

        self.runner, self.runner_name = self._build_runner()
        self._estimate_params_from_config(model_source, cache_dir)

        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.reduce_tokens = args.reduce_tokens
        self.index_tensor = torch.tensor(
            self.tokens_list, dtype=torch.long, device=self.device
        )

        self._sampling_config = self._build_sampling_config()

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """Run one-step TensorRT-LLM inference and return predictor scores."""
        del enable_kv_cache

        prompt_tensors = [
            torch.tensor(prompt, dtype=torch.int32, device=self.device)
            for prompt in prompts
        ]
        prompt_lengths = [len(prompt) for prompt in prompts]

        generate_kwargs = {
            "batch_input_ids": prompt_tensors,
            "return_dict": True,
            "output_generation_logits": True,
            "streaming": False,
            "max_new_tokens": 1,
            "end_id": self.eos_token_id,
            "pad_id": self.pad_token_id,
        }
        if self._sampling_config is not None:
            if self.runner_name == "ModelRunnerCpp":
                # TensorRT-LLM 1.0.0's C++ runner leaves sampling_config_list
                # undefined when a prebuilt SamplingConfig is supplied. Route
                # through the legacy kwargs path so the runtime constructs its
                # own per-request sampling config objects.
                for source_name, target_name in (
                    ("num_beams", "num_beams"),
                    ("beam_width", "num_beams"),
                    ("top_k", "top_k"),
                    ("top_p", "top_p"),
                    ("temperature", "temperature"),
                    ("num_return_sequences", "num_return_sequences"),
                    ("random_seed", "random_seed"),
                ):
                    value = getattr(self._sampling_config, source_name, None)
                    if value is not None and target_name not in generate_kwargs:
                        generate_kwargs[target_name] = value
            else:
                generate_kwargs["sampling_config"] = self._sampling_config

        call_kwargs = _select_supported_kwargs(self.runner.generate, generate_kwargs)

        t0_inference = time.perf_counter()
        outputs = self.runner.generate(**call_kwargs)
        inference_elapsed = time.perf_counter() - t0_inference

        logits = extract_last_step_logits(outputs, prompt_lengths).to(self.device)

        vocab_size = self.tokenizer.vocab_size
        if logits.shape[-1] > vocab_size:
            logits = logits[:, :vocab_size]

        if self.reduce_tokens:
            logits = logits.index_select(1, self.index_tensor)

        data_copy_time = inference_elapsed
        softmax_time = 0.0

        if self.args.encoding == "AC":
            t0_softmax = time.perf_counter()
            probs = torch.softmax(logits.float(), dim=-1)
            softmax_time = time.perf_counter() - t0_softmax

            t0_data_copy = time.perf_counter()
            probs_cpu = probs.cpu()
            data_copy_time += time.perf_counter() - t0_data_copy
            return self.tokens_list, probs_cpu, data_copy_time, softmax_time

        if self.args.encoding in ("bitpacked", "huffman"):
            return self.tokens_list, logits, data_copy_time, softmax_time

        raise NotImplementedError(
            f"Encoding method '{self.args.encoding}' is not implemented."
        )

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]

    def cleanup(self):
        self.runner = None

    def __del__(self):
        self.cleanup()

    def _build_runner(self):
        from tensorrt_llm.runtime import ModelRunner

        common_kwargs = {
            "engine_dir": self.engine_dir,
            "max_output_len": 1,
            "lora_dir": None,
            "rank": 0,
            "debug_mode": False,
        }

        try:
            from tensorrt_llm.runtime import ModelRunnerCpp

            cpp_kwargs = {
                **common_kwargs,
                "max_batch_size": getattr(self.args, "batch_size", 1),
                "max_input_len": getattr(self.args, "context_length", None),
                "gather_generation_logits": True,
            }
            runner = ModelRunnerCpp.from_dir(
                **_select_supported_kwargs(ModelRunnerCpp.from_dir, cpp_kwargs)
            )
            return runner, "ModelRunnerCpp"
        except Exception:
            runner = ModelRunner.from_dir(
                **_select_supported_kwargs(ModelRunner.from_dir, common_kwargs)
            )
            return runner, "ModelRunner"

    def _build_sampling_config(self):
        try:
            from tensorrt_llm.runtime import SamplingConfig

            sampling_config = SamplingConfig(
                end_id=self.eos_token_id,
                pad_id=self.pad_token_id,
                max_new_tokens=1,
                top_k=1,
                top_p=0.0,
                temperature=1.0,
            )
            # Older TensorRT-LLM Python runtimes expose num_beams on
            # SamplingConfig while ModelRunnerCpp still reads beam_width.
            # Mirror the field so both code paths work.
            if hasattr(sampling_config, "num_beams") and not hasattr(
                sampling_config, "beam_width"
            ):
                sampling_config.beam_width = sampling_config.num_beams
            return sampling_config
        except Exception:
            return None

    def _estimate_params_from_config(self, model_name, cache_dir):
        try:
            from transformers import AutoConfig

            if os.path.isdir(model_name):
                cfg = AutoConfig.from_pretrained(model_name)
            else:
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
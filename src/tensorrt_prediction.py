"""
TensorRT-LLM next-token prediction backend.

This module adapts TensorRT-LLM's LLM API to the predictor contract used by the
compression pipeline: batched token-id prompts in, next-token logits or
probabilities out.
"""

import importlib
import inspect
import os
import shutil
import time
from pathlib import Path

import torch
from pyroaring import BitMap
from transformers import AutoConfig, AutoTokenizer


def probe_tensorrt_backend_support(args):
    """
    Check whether TensorRT-LLM can be used in the current runtime.

    Returns:
        tuple: (supported: bool, reason: str | None)
    """
    del args

    try:
        importlib.import_module("tensorrt_llm")
    except ImportError:
        return False, "tensorrt_llm is not installed"
    except Exception as exc:
        return False, f"failed to import tensorrt_llm: {exc}"

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    return True, None


def _select_supported_kwargs(factory, kwargs):
    """Return only kwargs accepted by a callable when introspection is available."""
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
    """Normalize TensorRT-LLM logits to a [batch, vocab] tensor."""
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
    """Extract a [batch, vocab] logits tensor from TensorRT-LLM outputs."""
    if isinstance(output_payload, torch.Tensor):
        return _last_step_from_tensor(output_payload, prompt_lengths)

    if isinstance(output_payload, dict):
        for key in ("generation_logits", "context_logits"):
            if key in output_payload and output_payload[key] is not None:
                return extract_last_step_logits(output_payload[key], prompt_lengths)
        raise RuntimeError(
            "TensorRT-LLM generate() did not return generation_logits or context_logits."
        )

    if isinstance(output_payload, (list, tuple)):
        if not output_payload:
            raise RuntimeError("TensorRT-LLM generate() returned an empty payload.")

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
    """
    Token predictor backed by TensorRT-LLM's TensorRT engine backend.

    The first construction can build and cache a TensorRT engine. Later
    constructions with the same engine directory load the cached engine.
    """

    def __init__(self, args, bitmap_data):
        supported, reason = probe_tensorrt_backend_support(args)
        if not supported:
            raise ValueError(f"TensorRT-LLM backend is not available: {reason}")

        if getattr(args, "is_seq2seq", False):
            raise NotImplementedError(
                "--engine tensorrt does not support seq2seq models yet"
            )
        if getattr(args, "is_mamba", False):
            raise NotImplementedError(
                "--engine tensorrt does not support Mamba models yet"
            )
        if getattr(args, "lora_path", None) is not None:
            raise NotImplementedError(
                "--engine tensorrt does not support --lora_path yet"
            )

        LLM, SamplingParams = self._import_tensorrt_llm_api()

        self.args = args
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, cache_dir=".cache"
        )
        self.vocab_size = self._get_config_vocab_size(
            args.model_name, self.tokenizer.vocab_size
        )

        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.vocab_size))

        self.reduce_tokens = bool(getattr(args, "reduce_tokens", False))
        self.index_tensor = torch.tensor(self.tokens_list, dtype=torch.long, device=self.device)
        self.llm = None
        self.runner = None
        self.runner_name = None

        self.engine_dir = Path(
            getattr(args, "tensorrt_engine_dir", None)
            or self._default_engine_dir(args)
        )
        self.args.tensorrt_engine_dir = str(self.engine_dir)

        max_batch_size = int(getattr(args, "batch_size", 1))
        context_length = int(getattr(args, "context_length", 1024))

        if not self._has_cached_engine(self.engine_dir):
            self.engine_dir.parent.mkdir(parents=True, exist_ok=True)
            self.llm = LLM(
                model=args.model_name,
                tokenizer=args.model_name,
                max_batch_size=max_batch_size,
                max_input_len=context_length,
                max_seq_len=context_length + 1,
                gather_generation_logits=True,
            )
            self._cache_built_engine(self.engine_dir)

        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

        self.eos_token_id = self.tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.pad_token_id

        self._sampling_config = self._build_sampling_config()

        try:
            self.runner, self.runner_name = self._build_runner()
        except Exception as exc:
            print(
                "Warning: failed to initialize TensorRT-LLM ModelRunner; "
                f"falling back to LLM.generate(). Reason: {exc}"
            )
            if self.llm is None:
                self.llm = LLM(
                    model=str(self.engine_dir),
                    tokenizer=args.model_name,
                    gather_generation_logits=True,
                )

        self.sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            detokenize=False,
            ignore_eos=True,
            return_generation_logits=True,
        )

        actual_vocab_size = self._get_runtime_vocab_size()
        if not self.reduce_tokens and actual_vocab_size != self.vocab_size:
            self.vocab_size = actual_vocab_size
            self.tokens_list = list(range(self.vocab_size))
            self.index_tensor = torch.tensor(
                self.tokens_list, dtype=torch.long, device=self.device
            )
        else:
            self.vocab_size = actual_vocab_size

        self._estimate_params_from_config(args.model_name)

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """
        Run one TensorRT-LLM generation step and return next-token scores.

        TensorRT-LLM owns runtime KV-cache management. The flag is accepted for
        interface compatibility with the transformer predictor.
        """
        del enable_kv_cache

        if self.runner is not None:
            logits, data_copy_time = self._run_model_runner(prompts)
        else:
            logits, data_copy_time = self._run_llm_generate(prompts)

        return self._finalize_logits(logits, data_copy_time)

    def _run_llm_generate(self, prompts):
        request_prompts = [{"prompt_token_ids": list(prompt)} for prompt in prompts]
        outputs = self.llm.generate(
            request_prompts,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        if not isinstance(outputs, list):
            outputs = [outputs]
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"TensorRT-LLM returned {len(outputs)} outputs for {len(prompts)} prompts"
            )

        t0 = time.perf_counter()
        logits = self._outputs_to_logits(outputs)
        data_copy_time = time.perf_counter() - t0
        return logits, data_copy_time

    def _run_model_runner(self, prompts):
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
        t0 = time.perf_counter()
        outputs = self.runner.generate(**call_kwargs)
        data_copy_time = time.perf_counter() - t0
        logits = extract_last_step_logits(outputs, prompt_lengths).to(self.device)
        return logits, data_copy_time

    def _finalize_logits(self, logits, data_copy_time):
        if logits.shape[1] < self.vocab_size:
            raise RuntimeError(
                f"TensorRT-LLM returned {logits.shape[1]} logits columns, "
                f"but vocab_size is {self.vocab_size}"
            )
        if logits.shape[1] > self.vocab_size:
            logits = logits[:, :self.vocab_size]

        if self.reduce_tokens:
            logits = logits.index_select(1, self.index_tensor.to(logits.device))

        softmax_time = 0.0
        if self.args.encoding in {"AC", "AC_MULTISTREAM", "AC_TARGET_INTERVAL", "PMATIC"}:
            t0 = time.perf_counter()
            probs = torch.softmax(logits.float(), dim=-1)
            softmax_time = time.perf_counter() - t0

            if self.args.encoding in {"AC_MULTISTREAM", "AC_TARGET_INTERVAL"}:
                return self.tokens_list, probs, data_copy_time, softmax_time

            t0 = time.perf_counter()
            probs_cpu = probs.cpu()
            data_copy_time += time.perf_counter() - t0
            return self.tokens_list, probs_cpu, data_copy_time, softmax_time

        if self.args.encoding in {"bitpacked", "huffman"}:
            return self.tokens_list, logits, data_copy_time, softmax_time

        raise NotImplementedError(
            f"Encoding method '{self.args.encoding}' is not implemented."
        )

    def run_batched_inference_cachefree(self, prompts):
        return self.run_batched_inference(prompts, enable_kv_cache=False)

    def _outputs_to_logits(self, outputs):
        rows = []
        for output in outputs:
            if not getattr(output, "outputs", None):
                raise RuntimeError("TensorRT-LLM output did not contain a completion")
            completion = output.outputs[0]
            logits = getattr(completion, "generation_logits", None)
            if logits is None:
                logits = getattr(output, "generation_logits", None)
            if logits is None:
                raise RuntimeError(
                    "TensorRT-LLM did not return generation logits. "
                    "Expected SamplingParams(return_generation_logits=True) and "
                    "LLM(gather_generation_logits=True)."
                )
            logits = torch.as_tensor(logits)
            if logits.dim() == 3:
                logits = logits[0]
            if logits.dim() == 2:
                logits = logits[-1]
            if logits.dim() != 1:
                raise RuntimeError(
                    f"Expected one logits row per output, got shape {tuple(logits.shape)}"
                )
            rows.append(logits.float())

        return torch.stack(rows, dim=0).to(self.device)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]

    def _get_distinct_tokens(self):
        return self.tokens_list

    def cleanup(self):
        shutdown = getattr(getattr(self, "llm", None), "shutdown", None)
        if callable(shutdown):
            shutdown()
        self.runner = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    @staticmethod
    def _import_tensorrt_llm_api():
        try:
            from tensorrt_llm._tensorrt_engine import LLM
        except ImportError:
            from tensorrt_llm import LLM
        from tensorrt_llm import SamplingParams

        return LLM, SamplingParams

    def _cache_built_engine(self, engine_dir):
        save = getattr(self.llm, "save", None)
        if callable(save):
            save(str(engine_dir))
            return

        built_engine_dir = getattr(self.llm, "_engine_dir", None)
        if built_engine_dir is None:
            print(
                "Warning: TensorRT-LLM did not expose save() or _engine_dir; "
                "engine cache will not be persisted for this run."
            )
            return

        built_engine_dir = Path(built_engine_dir)
        if built_engine_dir.resolve() == engine_dir.resolve():
            return
        shutil.copytree(built_engine_dir, engine_dir, dirs_exist_ok=True)

    def _build_runner(self):
        from tensorrt_llm.runtime import ModelRunner

        common_kwargs = {
            "engine_dir": str(self.engine_dir),
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
            if hasattr(sampling_config, "num_beams") and not hasattr(
                sampling_config, "beam_width"
            ):
                sampling_config.beam_width = sampling_config.num_beams
            return sampling_config
        except Exception:
            return None

    @staticmethod
    def _default_engine_dir(args):
        safe_model = str(args.model_name).strip("/").replace("/", "_")
        return os.path.join(
            "trt_engines",
            safe_model,
            f"ctx{args.context_length}_batch{args.batch_size}",
        )

    @staticmethod
    def _has_cached_engine(engine_dir):
        engine_dir = Path(engine_dir)
        if not engine_dir.is_dir():
            return False
        return any(engine_dir.iterdir())

    def _get_runtime_vocab_size(self):
        # TensorRT-LLM may pad the logits dimension. The HF config/tokenizer
        # vocabulary remains the semantic vocabulary used by the compressor.
        return int(self.vocab_size)

    @staticmethod
    def _get_config_vocab_size(model_name, fallback):
        try:
            cfg = AutoConfig.from_pretrained(model_name, cache_dir=".cache")
            return max(int(fallback), int(getattr(cfg, "vocab_size", fallback)))
        except Exception:
            return int(fallback)

    def _estimate_params_from_config(self, model_name):
        try:
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

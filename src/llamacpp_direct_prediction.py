"""
Direct llama.cpp token prediction for compression experiments.

This module uses llama-cpp-python directly in-process, keeping tokenization,
detokenization, logits, and prompt-cache reuse aligned with the loaded GGUF.
"""

import os
import time

import torch
from pyroaring import BitMap

from src.cache_prompt_state import freeze_prompt, prompt_extends_one_token


def _load_llamacpp_dependencies():
    try:
        import llama_cpp
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for the llama.cpp direct backend. "
            "Install it with pip install llama-cpp-python or the repo's llama.cpp direct extra."
        ) from exc
    return llama_cpp


def _resolve_llamacpp_direct_runtime_config(args):
    context_length = max(1, int(getattr(args, "context_length", 512) or 512))
    logical_cpus = max(1, os.cpu_count() or 1)
    thread_count = max(1, int(getattr(args, "llamacpp_threads", logical_cpus) or logical_cpus))
    thread_count_batch = max(
        1,
        int(getattr(args, "llamacpp_direct_threads_batch", 0) or logical_cpus),
    )
    n_batch = max(
        1,
        min(
            context_length,
            int(getattr(args, "llamacpp_direct_n_batch", 0) or context_length),
        ),
    )
    n_ubatch = max(
        1,
        min(
            n_batch,
            int(getattr(args, "llamacpp_direct_n_ubatch", 0) or n_batch),
        ),
    )
    return {
        "context_length": context_length,
        "n_threads": thread_count,
        "n_threads_batch": thread_count_batch,
        "n_batch": n_batch,
        "n_ubatch": n_ubatch,
        "use_mmap": bool(getattr(args, "llamacpp_direct_use_mmap", True)),
        "use_mlock": bool(getattr(args, "llamacpp_direct_use_mlock", False)),
    }


def _build_llamacpp_model_kwargs(args, *, vocab_only=False, logits_all=False):
    runtime_config = _resolve_llamacpp_direct_runtime_config(args)
    return {
        "model_path": args.llamacpp_model_path,
        "n_ctx": runtime_config["context_length"],
        "n_batch": runtime_config["n_batch"],
        "n_ubatch": runtime_config["n_ubatch"],
        "n_threads": runtime_config["n_threads"],
        "n_threads_batch": runtime_config["n_threads_batch"],
        "n_gpu_layers": int(getattr(args, "llamacpp_n_gpu_layers", 0) or 0),
        "use_mmap": runtime_config["use_mmap"],
        "use_mlock": runtime_config["use_mlock"],
        "vocab_only": vocab_only,
        "logits_all": logits_all,
        "verbose": False,
    }


def _create_llamacpp_model(args, llama_cpp_module, *, vocab_only=False, logits_all=False):
    return llama_cpp_module.Llama(
        **_build_llamacpp_model_kwargs(
            args,
            vocab_only=vocab_only,
            logits_all=logits_all,
        )
    )


def _estimate_llamacpp_parameter_count(model, llama_cpp_module):
    count_fn = getattr(llama_cpp_module, "llama_model_n_params", None)
    model_ptr = getattr(model, "model", None)
    if callable(count_fn) and model_ptr is not None:
        try:
            return int(count_fn(model_ptr))
        except (TypeError, ValueError):
            return 0
    return 0


def probe_llamacpp_direct_support(args):
    """Check whether the direct llama.cpp binding can satisfy this repo's predictor contract."""
    model_path = getattr(args, "llamacpp_model_path", None)
    if not model_path:
        return False, "llamacpp_model_path is required"
    if not os.path.isfile(model_path):
        return False, f"GGUF model not found: {model_path}"

    try:
        _load_llamacpp_dependencies()
    except ImportError as exc:
        return False, str(exc)

    return True, None


def estimate_llamacpp_direct_artifact_size_mb(model_path):
    if not model_path or not os.path.isfile(model_path):
        return 0.0
    return os.path.getsize(model_path) / (1024 ** 2)


class LlamaCppDirectTokenizer:
    """Tokenizer wrapper that uses the loaded llama.cpp GGUF tokenizer."""

    def __init__(self, args, model=None, llama_cpp_module=None):
        self.args = args
        self._llama_cpp = llama_cpp_module or _load_llamacpp_dependencies()
        self._owns_model = model is None
        self.model = model or _create_llamacpp_model(
            args,
            self._llama_cpp,
            vocab_only=True,
            logits_all=False,
        )
        self.vocab_size = int(self.model.n_vocab())
        if self.vocab_size <= 0:
            raise RuntimeError("llama.cpp tokenizer did not expose a positive vocabulary size")

    def encode(self, text, truncation=False, max_length=None):
        token_ids = [
            int(token_id)
            for token_id in self.model.tokenize(
                text.encode("utf-8"),
                add_bos=False,
                special=False,
            )
        ]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids):
        return self.detokenize(token_ids)

    def detokenize(self, token_ids):
        return self.model.detokenize(
            [int(token_id) for token_id in token_ids],
            special=False,
        ).decode("utf-8", errors="ignore")

    def cleanup(self):
        if self._owns_model and self.model is not None:
            close = getattr(self.model, "close", None)
            if callable(close):
                close()
            self.model = None


class LlamaCppDirectTokenPredictor:
    """Token predictor backed by a direct llama-cpp-python model binding."""

    def __init__(self, args, bitmap_data):
        supported, reason = probe_llamacpp_direct_support(args)
        if not supported:
            raise ValueError(f"llama.cpp direct backend is not available: {reason}")
        if getattr(args, "lora_path", None) is not None:
            raise ValueError("llama.cpp direct backend does not support LoRA adapters in this repo.")

        self.args = args
        self.llama_cpp = _load_llamacpp_dependencies()
        self.model = _create_llamacpp_model(args, self.llama_cpp, vocab_only=False, logits_all=True)
        self.tokenizer = LlamaCppDirectTokenizer(args, model=self.model, llama_cpp_module=self.llama_cpp)

        self.base_params = _estimate_llamacpp_parameter_count(self.model, self.llama_cpp)
        self.base_size_mb = estimate_llamacpp_direct_artifact_size_mb(args.llamacpp_model_path)
        self.adapter_params = 0
        self.adapter_size_mb = 0.0

        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.index_tensor = (
            torch.tensor(self.tokens_list, dtype=torch.long)
            if self.tokens_list != list(range(self.tokenizer.vocab_size))
            else None
        )
        self._slot_prompts = {}
        self._slot_states = {}

    def _ensure_slot_cache(self):
        if not hasattr(self, "_slot_prompts") or self._slot_prompts is None:
            self._slot_prompts = {}
        if not hasattr(self, "_slot_states") or self._slot_states is None:
            self._slot_states = {}

    def _reset_slot(self, slot_id):
        self._ensure_slot_cache()
        self.model.reset()
        self._slot_prompts.pop(slot_id, None)
        self._slot_states.pop(slot_id, None)

    def _prepare_eval_tokens(self, slot_id, prompt_tokens, enable_kv_cache):
        self._ensure_slot_cache()
        if not enable_kv_cache:
            self._reset_slot(slot_id)
            return [int(token_id) for token_id in prompt_tokens]

        previous_prompt = self._slot_prompts.get(slot_id)
        previous_state = self._slot_states.get(slot_id)
        if previous_prompt is None or previous_state is None:
            self._reset_slot(slot_id)
            return [int(token_id) for token_id in prompt_tokens]

        if not prompt_extends_one_token(previous_prompt, prompt_tokens):
            self._reset_slot(slot_id)
            return [int(token_id) for token_id in prompt_tokens]

        self.model.load_state(previous_state)
        return [int(prompt_tokens[-1])]

    def _get_last_logits(self):
        eval_logits = getattr(self.model, "eval_logits", None)
        if eval_logits is not None and len(eval_logits) > 0:
            return torch.tensor(eval_logits[-1], dtype=torch.float32)

        scores = getattr(self.model, "scores", None)
        n_tokens = getattr(self.model, "n_tokens", 0)
        if scores is not None and n_tokens > 0:
            return torch.tensor(scores[n_tokens - 1], dtype=torch.float32)

        raise RuntimeError("llama.cpp model did not expose logits for the evaluated prompt")

    def _evaluate_prompt(self, prompt_tokens, slot_id, enable_kv_cache):
        if not prompt_tokens:
            raise ValueError("llama.cpp direct backend requires non-empty prompts for next-token prediction")

        self._ensure_slot_cache()
        eval_tokens = self._prepare_eval_tokens(slot_id, prompt_tokens, enable_kv_cache)
        self.model.eval(eval_tokens)
        logits = self._get_last_logits()

        if enable_kv_cache:
            self._slot_prompts[slot_id] = freeze_prompt(prompt_tokens)
            self._slot_states[slot_id] = self.model.save_state()
        else:
            self._slot_prompts.pop(slot_id, None)
            self._slot_states.pop(slot_id, None)

        return logits

    def _select_active_logits(self, logits):
        if self.index_tensor is None:
            return logits
        return logits.index_select(0, self.index_tensor)

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        self._ensure_slot_cache()
        rows = []
        for slot_id, prompt_tokens in enumerate(prompts):
            logits = self._evaluate_prompt(prompt_tokens, slot_id, enable_kv_cache)
            reduced_logits = self._select_active_logits(logits)
            rows.append(reduced_logits)

        reduced_logits = torch.stack(rows, dim=0)
        if self.args.encoding == "AC":
            t0_softmax = time.perf_counter()
            probs = torch.softmax(reduced_logits, dim=1)
            softmax_time = time.perf_counter() - t0_softmax
            return self.tokens_list, probs, 0.0, softmax_time
        if self.args.encoding in ("bitpacked", "huffman"):
            return self.tokens_list, reduced_logits, 0.0, 0.0
        raise NotImplementedError(f"Encoding method '{self.args.encoding}' is not implemented.")

    def detokenize(self, token_ids):
        return self.tokenizer.detokenize(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]

    def cleanup(self):
        if hasattr(self, "model") and self.model is not None:
            close = getattr(self.model, "close", None)
            if callable(close):
                close()
            self.model = None

    def __del__(self):
        self.cleanup()
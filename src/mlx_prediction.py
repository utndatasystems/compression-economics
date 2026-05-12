"""MLX-LM based token prediction for Apple Silicon compression experiments."""

import glob
import json
import os
import platform
import time

import numpy as np
import torch
from pyroaring import BitMap

from src.cache_prompt_state import freeze_prompts, prompts_extend_one_token


_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
)


def _load_mlx_dependencies():
    try:
        import mlx.core as mx
    except ImportError as exc:
        return None, None, None, None, None, f"mlx is not installed: {exc}"

    try:
        from mlx_lm import load as mlx_load
        from mlx_lm.models import cache as mlx_cache
        from mlx_lm.utils import get_total_parameters, load_tokenizer
    except ImportError as exc:
        return mx, None, None, None, None, f"mlx-lm is not installed: {exc}"

    return mx, mlx_load, load_tokenizer, mlx_cache, get_total_parameters, None


def get_mlx_model_source(args):
    """Return the local path or Hugging Face repo to load through MLX-LM."""
    explicit_source = getattr(args, "mlx_model_source", None)
    return explicit_source or args.model_name


def get_mlx_tokenizer_source(args):
    """Pick the tokenizer source for the MLX backend."""
    explicit_source = getattr(args, "mlx_tokenizer_source", None)
    if explicit_source:
        return explicit_source

    model_source = get_mlx_model_source(args)
    if os.path.isdir(model_source):
        if any(os.path.exists(os.path.join(model_source, filename)) for filename in _TOKENIZER_FILENAMES):
            return model_source

    return model_source


def list_mlx_weight_files(model_source):
    """Return local MLX safetensor shards when the model source is a directory."""
    if not model_source or not os.path.isdir(model_source):
        return []
    return sorted(glob.glob(os.path.join(model_source, "model*.safetensors")))


def load_mlx_tokenizer(tokenizer_source):
    """Load an MLX tokenizer without loading the full model."""
    _, _, load_tokenizer_fn, _, _, reason = _load_mlx_dependencies()
    if reason is not None:
        raise ValueError(f"MLX tokenizer is not available: {reason}")
    return load_tokenizer_fn(tokenizer_source)


def estimate_mlx_artifact_size_mb(model_source):
    """Estimate the size of a local MLX model directory."""
    if not model_source or not os.path.isdir(model_source):
        return 0.0

    total_bytes = 0
    for root, _, filenames in os.walk(model_source):
        for filename in filenames:
            total_bytes += os.path.getsize(os.path.join(root, filename))
    return total_bytes / (1024 ** 2)


def estimate_mlx_parameter_count(args):
    """Read parameter count metadata from a local MLX model directory when available."""
    model_source = get_mlx_model_source(args)
    if not model_source or not os.path.isdir(model_source):
        return 0

    index_path = os.path.join(model_source, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return 0

    try:
        with open(index_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0

    total_parameters = metadata.get("metadata", {}).get("total_parameters", 0)
    if isinstance(total_parameters, int):
        return total_parameters
    return 0


def probe_mlx_backend_support(args):
    """Check whether the MLX backend can satisfy this repo's predictor contract."""
    if platform.system() != "Darwin":
        return False, "MLX backend requires macOS"
    if platform.machine() not in {"arm64", "aarch64"}:
        return False, "MLX backend requires Apple Silicon (arm64)"
    if getattr(args, "lora_path", None) is not None:
        return False, "MLX backend does not yet support LoRA adapters in this repo"

    model_source = get_mlx_model_source(args)
    if not model_source:
        return False, "mlx_model_source or model_name is required"
    if os.path.exists(model_source) and not os.path.isdir(model_source):
        return False, f"MLX model source must be a directory or Hugging Face repo: {model_source}"
    if os.path.isdir(model_source) and not list_mlx_weight_files(model_source):
        return False, f"No MLX safetensor weights found in: {model_source}"

    _, _, _, _, _, reason = _load_mlx_dependencies()
    if reason is not None:
        return False, reason

    return True, None


class MLXTokenPredictor:
    """Token predictor backed by an MLX-LM causal LM on Apple Silicon."""

    def __init__(self, args, bitmap_data):
        supported, reason = probe_mlx_backend_support(args)
        if not supported:
            raise ValueError(f"MLX backend is not available: {reason}")

        mx, mlx_load, load_tokenizer_fn, mlx_cache, get_total_parameters_fn, _ = _load_mlx_dependencies()
        self.args = args
        self.mx = mx
        self._mlx_cache = mlx_cache
        self._get_total_parameters = get_total_parameters_fn

        model_source = get_mlx_model_source(args)
        tokenizer_source = get_mlx_tokenizer_source(args)
        self.model, loaded_tokenizer = mlx_load(model_source)
        if tokenizer_source == model_source:
            self.tokenizer = loaded_tokenizer
        else:
            self.tokenizer = load_tokenizer_fn(tokenizer_source)

        self.base_params = int(get_total_parameters_fn(self.model))
        self.base_size_mb = estimate_mlx_artifact_size_mb(model_source)
        self.adapter_params = 0
        self.adapter_size_mb = 0.0

        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.index_tensor = torch.tensor(self.tokens_list, dtype=torch.long)
        self.reduce_tokens = args.reduce_tokens
        self._prompt_cache = None
        self._cached_prompts = None

    def _to_mx_tokens(self, prompts):
        return self.mx.array(prompts, dtype=self.mx.uint32)

    def _forward_full_prompt(self, prompts):
        input_ids = self._to_mx_tokens(prompts)
        return self.model(input_ids)

    def _forward_incremental(self, prompts):
        can_advance_cache = (
            self._prompt_cache is not None
            and prompts_extend_one_token(self._cached_prompts, prompts)
        )
        if not can_advance_cache:
            self._prompt_cache = self._mlx_cache.make_prompt_cache(self.model)
            input_ids = self._to_mx_tokens(prompts[0])
        else:
            input_ids = self._to_mx_tokens([prompts[0][-1]])

        return self.model(input_ids[None], cache=self._prompt_cache)

    def _mx_logits_to_torch(self, logits):
        self.mx.eval(logits)
        return torch.from_numpy(np.asarray(logits)).clone()

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        if not prompts:
            raise ValueError("prompts must not be empty")

        data_copy_time = 0.0
        use_incremental_cache = enable_kv_cache and len(prompts) == 1

        if use_incremental_cache:
            mx_logits = self._forward_incremental(prompts)
            self._cached_prompts = freeze_prompts(prompts)
        else:
            mx_logits = self._forward_full_prompt(prompts)
            self._prompt_cache = None
            self._cached_prompts = None

        t0_data_copy = time.perf_counter()
        logits = self._mx_logits_to_torch(mx_logits[:, -1, :])
        data_copy_time += time.perf_counter() - t0_data_copy

        if self.reduce_tokens:
            logits = logits.index_select(1, self.index_tensor)

        softmax_time = 0.0
        if self.args.encoding == "AC":
            t0_softmax = time.perf_counter()
            probs = torch.softmax(logits, dim=-1)
            softmax_time = time.perf_counter() - t0_softmax
            return self.tokens_list, probs.cpu(), data_copy_time, softmax_time
        if self.args.encoding in ("bitpacked", "huffman"):
            return self.tokens_list, logits, data_copy_time, softmax_time
        raise NotImplementedError(f"Encoding method '{self.args.encoding}' is not implemented.")

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]

    def cleanup(self):
        self._prompt_cache = None
        self._cached_prompts = None
        self.mx.clear_cache()
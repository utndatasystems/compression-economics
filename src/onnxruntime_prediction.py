"""ONNX Runtime-based token prediction for CPU-oriented compression experiments."""

import os
import time

import torch
from pyroaring import BitMap
from transformers import AutoConfig, AutoTokenizer

from src.cache_prompt_state import freeze_prompts, prompts_extend_one_token
from src.hf_cache import get_model_cache_dir


_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
)


def _load_onnxruntime_dependencies():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return None, None, f"onnxruntime is not installed: {exc}"

    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError as exc:
        return ort, None, f"optimum[onnxruntime] is not installed: {exc}"

    return ort, ORTModelForCausalLM, None


def list_onnx_model_files(model_dir):
    """Return ONNX graph and external-data files under a model directory."""
    if not model_dir or not os.path.isdir(model_dir):
        return []

    model_files = []
    for root, _, filenames in os.walk(model_dir):
        for filename in filenames:
            if filename.endswith((".onnx", ".ort", ".data")) or ".onnx_data" in filename:
                model_files.append(os.path.join(root, filename))
    return sorted(model_files)


def get_onnx_tokenizer_source(args):
    """Pick the tokenizer source for the ONNX Runtime backend."""
    explicit_source = getattr(args, "onnx_tokenizer_source", None)
    if explicit_source:
        return explicit_source

    model_dir = getattr(args, "onnx_model_dir", None)
    if model_dir and os.path.isdir(model_dir):
        if any(os.path.exists(os.path.join(model_dir, filename)) for filename in _TOKENIZER_FILENAMES):
            return model_dir

    return args.model_name


def estimate_onnx_artifact_size_mb(model_dir):
    """Estimate the size of an exported ONNX artifact directory."""
    if not model_dir or not os.path.isdir(model_dir):
        return 0.0

    total_bytes = 0
    for root, _, filenames in os.walk(model_dir):
        for filename in filenames:
            total_bytes += os.path.getsize(os.path.join(root, filename))
    return total_bytes / (1024 ** 2)


def estimate_onnx_parameter_count(args):
    """Approximate parameter count from config metadata when available."""
    cache_dir = get_model_cache_dir()
    config_source = getattr(args, "onnx_model_dir", None) or args.model_name

    try:
        config = AutoConfig.from_pretrained(config_source, cache_dir=cache_dir)
    except Exception:
        if config_source == args.model_name:
            return 0
        try:
            config = AutoConfig.from_pretrained(args.model_name, cache_dir=cache_dir)
        except Exception:
            return 0

    num_parameters = getattr(config, "num_parameters", None)
    if isinstance(num_parameters, int) and num_parameters > 0:
        return num_parameters

    hidden_size = getattr(config, "hidden_size", 0) or getattr(config, "d_model", 0)
    num_hidden_layers = getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0)
    vocab_size = getattr(config, "vocab_size", 0)
    if hidden_size and num_hidden_layers and vocab_size:
        return int(num_hidden_layers * 12 * hidden_size * hidden_size + vocab_size * hidden_size)
    return 0


def build_onnx_session_options(args, ort_module):
    """Build ONNX Runtime session options from CLI arguments."""
    options = ort_module.SessionOptions()

    intra_threads = getattr(args, "onnx_intra_op_threads", 0)
    inter_threads = getattr(args, "onnx_inter_op_threads", 0)
    if intra_threads and intra_threads > 0:
        options.intra_op_num_threads = intra_threads
    if inter_threads and inter_threads > 0:
        options.inter_op_num_threads = inter_threads

    optimization_name = getattr(args, "onnx_graph_optimization_level", "ORT_ENABLE_ALL")
    try:
        options.graph_optimization_level = getattr(
            ort_module.GraphOptimizationLevel,
            optimization_name,
        )
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported ONNX graph optimization level: {optimization_name}"
        ) from exc

    return options


def probe_onnxruntime_backend_support(args):
    """Check whether the ONNX Runtime backend can satisfy this repo's predictor contract."""
    model_dir = getattr(args, "onnx_model_dir", None)
    if not model_dir:
        return False, "onnx_model_dir is required"
    if not os.path.isdir(model_dir):
        return False, f"ONNX model directory not found: {model_dir}"
    if not list_onnx_model_files(model_dir):
        return False, f"No ONNX model files found in: {model_dir}"
    if getattr(args, "lora_path", None) is not None:
        return False, "ONNX Runtime backend does not support LoRA adapters in this repo"

    requested_provider = getattr(args, "onnx_execution_provider", "CPUExecutionProvider")
    if requested_provider != "CPUExecutionProvider":
        return False, "ONNX Runtime backend is currently scoped to CPUExecutionProvider"

    ort_module, _, reason = _load_onnxruntime_dependencies()
    if reason is not None:
        return False, reason

    available_providers = set(ort_module.get_available_providers())
    if requested_provider not in available_providers:
        return False, (
            f"Requested ONNX Runtime provider '{requested_provider}' is unavailable; "
            f"available providers: {sorted(available_providers)}"
        )

    return True, None


class ONNXRuntimeTokenPredictor:
    """Token predictor backed by an exported ONNX Runtime causal LM."""

    def __init__(self, args, bitmap_data):
        supported, reason = probe_onnxruntime_backend_support(args)
        if not supported:
            raise ValueError(f"ONNX Runtime backend is not available: {reason}")

        ort_module, ort_model_cls, _ = _load_onnxruntime_dependencies()
        self.args = args
        self.device = torch.device("cpu")
        tokenizer_source = get_onnx_tokenizer_source(args)
        tokenizer_kwargs = {}
        if not os.path.isdir(tokenizer_source):
            tokenizer_kwargs["cache_dir"] = get_model_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        self.model = ort_model_cls.from_pretrained(
            args.onnx_model_dir,
            provider=getattr(args, "onnx_execution_provider", "CPUExecutionProvider"),
            session_options=build_onnx_session_options(args, ort_module),
            use_io_binding=False,
        )

        self.base_params = estimate_onnx_parameter_count(args)
        self.base_size_mb = estimate_onnx_artifact_size_mb(args.onnx_model_dir)
        self.adapter_params = 0
        self.adapter_size_mb = 0.0

        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.index_tensor = torch.tensor(self.tokens_list, dtype=torch.long)
        self.reduce_tokens = args.reduce_tokens
        self._past_key_values = None
        self._cached_prompt_len = 0
        self._cached_prompts = None

    def _build_input_ids(self, prompts):
        return torch.tensor(prompts, dtype=torch.long)

    def _build_attention_mask(self, batch_size, sequence_length):
        return torch.ones((batch_size, sequence_length), dtype=torch.long)

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        if not prompts:
            raise ValueError("prompts must not be empty")

        data_copy_time = 0.0
        batch_size = len(prompts)
        prompt_length = len(prompts[0])
        with torch.inference_mode():
            if not enable_kv_cache:
                input_ids = self._build_input_ids(prompts)
                attention_mask = self._build_attention_mask(batch_size, prompt_length)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                self._past_key_values = None
                self._cached_prompt_len = 0
                self._cached_prompts = None
            else:
                can_advance_cache = (
                    self._past_key_values is not None
                    and prompts_extend_one_token(self._cached_prompts, prompts)
                )
                reset_cache = not can_advance_cache
                if reset_cache:
                    input_ids = self._build_input_ids(prompts)
                    attention_mask = self._build_attention_mask(batch_size, prompt_length)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                else:
                    delta = torch.tensor([row[-1:] for row in prompts], dtype=torch.long)
                    attention_mask = self._build_attention_mask(batch_size, prompt_length)
                    outputs = self.model(
                        input_ids=delta,
                        attention_mask=attention_mask,
                        past_key_values=self._past_key_values,
                        use_cache=True,
                    )
                self._past_key_values = getattr(outputs, "past_key_values", None)
                self._cached_prompt_len = prompt_length
                self._cached_prompts = freeze_prompts(prompts)

            logits = torch.as_tensor(outputs.logits)[:, -1, :]
            if self.reduce_tokens:
                logits = logits.index_select(1, self.index_tensor.to(logits.device))

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
        self._past_key_values = None
        self._cached_prompt_len = 0
        self._cached_prompts = None
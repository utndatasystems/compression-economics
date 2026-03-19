"""
llama.cpp-based token prediction for compression experiments.

This module manages a local llama-server process, uses the server's runtime
tokenization APIs, and exposes the same predictor contract as the other engines.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from urllib import error, parse, request

import torch
from pyroaring import BitMap


def _resolve_llamacpp_binary(binary):
    if not binary:
        return None
    if os.path.isabs(binary) or os.path.sep in binary:
        if os.path.isfile(binary) and os.access(binary, os.X_OK):
            return binary
        return None
    return shutil.which(binary)


def probe_llamacpp_ac_support(args):
    """Check whether llama.cpp can satisfy this repo's predictor contract."""
    if not getattr(args, "reduce_tokens", True):
        return False, "llama.cpp backend currently requires reduce_tokens=True"

    model_path = getattr(args, "llamacpp_model_path", None)
    if not model_path:
        return False, "llamacpp_model_path is required"
    if not os.path.isfile(model_path):
        return False, f"GGUF model not found: {model_path}"

    binary = getattr(args, "llamacpp_binary", "llama-server")
    resolved = _resolve_llamacpp_binary(binary)
    if resolved is None:
        return False, f"llama-server binary not found or not executable: {binary}"

    return True, None


def build_llamacpp_server_command(args, parallelism, slot_save_path=None):
    """Build the managed llama-server command line."""
    binary = _resolve_llamacpp_binary(getattr(args, "llamacpp_binary", "llama-server"))
    if binary is None:
        binary = getattr(args, "llamacpp_binary", "llama-server")

    command = [
        binary,
        "--model",
        args.llamacpp_model_path,
        "--host",
        getattr(args, "llamacpp_host", "127.0.0.1"),
        "--port",
        str(getattr(args, "llamacpp_port", 8080)),
        "--ctx-size",
        str(getattr(args, "context_length", 0)),
        "--parallel",
        str(max(1, int(parallelism))),
        "--threads",
        str(max(1, int(getattr(args, "llamacpp_threads", 1)))),
        "--n-gpu-layers",
        str(int(getattr(args, "llamacpp_n_gpu_layers", 0))),
    ]

    if slot_save_path is not None:
        command.extend(["--slot-save-path", slot_save_path])

    return command


def _json_request(base_url, path, payload=None, method=None, timeout=30.0):
    url = f"{base_url}{path}"
    data = None
    headers = {"Accept": "application/json"}
    req_method = method or ("POST" if payload is not None else "GET")
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method=req_method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server request failed with HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"llama-server request failed: {exc}") from exc

    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"llama-server returned invalid JSON for {path}: {body[:200]!r}") from exc


def _extract_model_metadata(models_payload):
    data = []
    if isinstance(models_payload, dict):
        raw_data = models_payload.get("data")
        if isinstance(raw_data, list):
            data = raw_data

    if not data:
        raise RuntimeError("llama-server /v1/models did not return any loaded model metadata")

    first = data[0]
    meta = first.get("meta") if isinstance(first, dict) else None
    if not isinstance(meta, dict):
        meta = {}

    return {
        "model_id": first.get("id") if isinstance(first, dict) else None,
        "n_vocab": int(meta.get("n_vocab") or 0),
        "n_params": int(meta.get("n_params") or 0),
        "size": int(meta.get("size") or 0),
    }


class LlamaCppServerProcess:
    """Managed local llama-server process with readiness polling."""

    def __init__(self, args, parallelism, startup_timeout=60.0, poll_interval=0.25):
        self.args = args
        self.parallelism = parallelism
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval
        self._slot_save_path = tempfile.mkdtemp(prefix="llamacpp-slots-")
        self.command = build_llamacpp_server_command(
            args,
            parallelism,
            slot_save_path=self._slot_save_path,
        )
        self.base_url = f"http://{getattr(args, 'llamacpp_host', '127.0.0.1')}:{getattr(args, 'llamacpp_port', 8080)}"
        self.process = None
        self._log_file = None

    def start(self):
        if self.process is not None:
            return self

        self._log_file = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited before becoming healthy: "
                    f"{self._read_log_tail()}"
                )
            if self._is_healthy():
                return self
            time.sleep(self.poll_interval)

        self.stop()
        raise RuntimeError(
            "Timed out waiting for llama-server to become healthy: "
            f"{self._read_log_tail()}"
        )

    def _is_healthy(self):
        try:
            _json_request(self.base_url, "/health", timeout=2.0)
            return True
        except RuntimeError:
            return False

    def _read_log_tail(self, max_chars=2000):
        if self._log_file is None:
            return "<no process log available>"
        self._log_file.flush()
        self._log_file.seek(0)
        content = self._log_file.read()
        if not content:
            return "<no process log available>"
        return content[-max_chars:]

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if getattr(self, "_slot_save_path", None) is not None:
            shutil.rmtree(self._slot_save_path, ignore_errors=True)
            self._slot_save_path = None


class LlamaCppTokenizer:
    """Tokenizer wrapper that uses llama-server /tokenize and /detokenize."""

    def __init__(self, args, server_process=None):
        self.args = args
        self._owns_server = server_process is None
        self.server_process = server_process or LlamaCppServerProcess(args, parallelism=1)
        self.server_process.start()
        self.base_url = self.server_process.base_url
        metadata = _extract_model_metadata(_json_request(self.base_url, "/v1/models", timeout=10.0))
        self.vocab_size = metadata["n_vocab"]
        if self.vocab_size <= 0:
            raise RuntimeError("llama-server /v1/models did not expose n_vocab metadata")

    def encode(self, text, truncation=False, max_length=None):
        payload = {
            "content": text,
            "add_special": False,
            "with_pieces": False,
        }
        response = _json_request(self.base_url, "/tokenize", payload=payload, timeout=30.0)
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"Unexpected llama-server /tokenize payload: {response!r}")
        token_ids = [int(token_id) for token_id in tokens]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def decode(self, token_ids):
        return self.detokenize(token_ids)

    def detokenize(self, token_ids):
        response = _json_request(
            self.base_url,
            "/detokenize",
            payload={"tokens": [int(token_id) for token_id in token_ids]},
            timeout=30.0,
        )
        content = response.get("content")
        if content is None:
            content = response.get("text")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected llama-server /detokenize payload: {response!r}")
        return content

    def cleanup(self):
        if self._owns_server and self.server_process is not None:
            self.server_process.stop()
            self.server_process = None


def extract_step_probabilities(response_payload, expected_token_ids):
    """Parse the first-step probability payload ordered by expected token ids."""
    expected = [int(token_id) for token_id in expected_token_ids]
    if not expected:
        return []

    steps = response_payload.get("completion_probabilities")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("llama-server completion response did not include completion_probabilities")

    step = steps[0]
    if not isinstance(step, dict):
        raise RuntimeError(f"Unexpected llama-server probability step payload: {step!r}")

    candidates = []
    for key in ("top_probs", "top_logprobs", "probs"):
        value = step.get(key)
        if isinstance(value, list):
            candidates = value
            break

    if not candidates and isinstance(step.get("probs"), dict):
        candidates = step["probs"].get("top_probs") or step["probs"].get("top_logprobs") or []

    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"Unexpected llama-server probability payload: {response_payload!r}")

    values_by_id = {}
    for item in candidates:
        if not isinstance(item, dict):
            continue
        token_id = item.get("id")
        if token_id is None:
            continue
        if "prob" in item and item["prob"] is not None:
            prob = float(item["prob"])
        elif "logprob" in item and item["logprob"] is not None:
            prob = math.exp(float(item["logprob"]))
        else:
            continue
        values_by_id[int(token_id)] = prob

    missing = [token_id for token_id in expected if token_id not in values_by_id]
    if missing:
        raise RuntimeError(
            "llama-server response did not include probabilities for all requested reduced tokens; "
            f"missing {len(missing)} token ids"
        )

    return [values_by_id[token_id] for token_id in expected]


class LlamaCppTokenPredictor:
    """Token predictor backed by a managed local llama-server instance."""

    def __init__(self, args, bitmap_data):
        supported, reason = probe_llamacpp_ac_support(args)
        if not supported:
            raise ValueError(f"llama.cpp backend is not available: {reason}")
        if getattr(args, "lora_path", None) is not None:
            raise ValueError("llama.cpp backend does not support LoRA adapters in this repo.")

        self.args = args
        self.server_process = LlamaCppServerProcess(args, parallelism=max(1, getattr(args, "batch_size", 1)))
        self.server_process.start()
        self.base_url = self.server_process.base_url
        self.tokenizer = LlamaCppTokenizer(args, server_process=self.server_process)

        metadata = _extract_model_metadata(_json_request(self.base_url, "/v1/models", timeout=10.0))
        self.base_params = metadata["n_params"]
        self.base_size_mb = metadata["size"] / (1024 ** 2) if metadata["size"] else 0.0
        self.adapter_params = 0
        self.adapter_size_mb = 0.0

        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.reduce_tokens = args.reduce_tokens
        self._slot_prompt_lengths = {}
        self._disallowed_bias = self._build_disallowed_bias()

    def _build_disallowed_bias(self):
        allowed = set(int(token_id) for token_id in self.tokens_list)
        bias = {}
        for token_id in range(self.tokenizer.vocab_size):
            if token_id not in allowed:
                bias[str(token_id)] = False
        return bias

    def _should_reset_slot(self, slot_id, prompt_tokens, enable_kv_cache):
        if not enable_kv_cache:
            return True
        previous_length = self._slot_prompt_lengths.get(slot_id)
        if previous_length is None:
            return False
        return len(prompt_tokens) < previous_length

    def _reset_slot(self, slot_id):
        path = f"/slots/{int(slot_id)}?{parse.urlencode({'action': 'erase'})}"
        _json_request(self.base_url, path, payload={}, timeout=10.0)
        self._slot_prompt_lengths.pop(slot_id, None)

    def _prepare_slot(self, slot_id, prompt_tokens, enable_kv_cache):
        reset_slot = self._should_reset_slot(slot_id, prompt_tokens, enable_kv_cache)
        if reset_slot:
            self._reset_slot(slot_id)
        self._slot_prompt_lengths[slot_id] = len(prompt_tokens)
        return enable_kv_cache and not reset_slot

    def _request_completion(self, prompt_tokens, slot_id, cache_prompt):
        payload = {
            "prompt": [int(token_id) for token_id in prompt_tokens],
            "n_predict": 1,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "cache_prompt": cache_prompt,
            "id_slot": int(slot_id),
            "n_probs": len(self.tokens_list),
            "min_keep": len(self.tokens_list),
            "post_sampling_probs": True,
            "return_tokens": True,
            "stream": False,
            "logit_bias": self._disallowed_bias,
        }
        return _json_request(self.base_url, "/completion", payload=payload, timeout=120.0)

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        rows = []
        for slot_id, prompt_tokens in enumerate(prompts):
            cache_prompt = self._prepare_slot(slot_id, prompt_tokens, enable_kv_cache)
            response_payload = self._request_completion(prompt_tokens, slot_id, cache_prompt)
            rows.append(extract_step_probabilities(response_payload, self.tokens_list))

        probs = torch.tensor(rows, dtype=torch.float32)
        if self.args.encoding == "AC":
            return self.tokens_list, probs, 0.0, 0.0
        if self.args.encoding in ("bitpacked", "huffman"):
            return self.tokens_list, probs.clamp_min(torch.finfo(torch.float32).tiny).log(), 0.0, 0.0
        raise NotImplementedError(f"Encoding method '{self.args.encoding}' is not implemented.")

    def detokenize(self, token_ids):
        return self.tokenizer.detokenize(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]

    def cleanup(self):
        if hasattr(self, "server_process") and self.server_process is not None:
            self.server_process.stop()
            self.server_process = None

    def __del__(self):
        self.cleanup()
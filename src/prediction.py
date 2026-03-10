"""
Token preparation and next-token prediction helpers for compression experiments.

This module provides:
- TokenDataPreparer: loads raw text, tokenizes, and optionally reduces the vocabulary
  to the set of tokens observed in the input (for bitmap-based masking).
- TokenPredictor: runs batched, one-step LLM inference with optional KV caching and
  returns logits or probabilities depending on the encoding scheme.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import os
import torch
import math
from pyroaring import BitMap
import time
from peft import PeftModel
#from transformers.convert_slow_tokenizers_checkpoints_to_fast import args

class TokenDataPreparer:
    def __init__(self, args):
        """
        Initialize the data preparer and tokenize the input.

        Args:
            args (argparse.Namespace): Experiment configuration. Expected fields:
                input_path (str | None): Path to the input text file.
                text_input (str | None): Raw text input (exclusive with input_path).
                model_name (str): HuggingFace model name (tokenizer source).
                first_n_tokens (int | None): Optional token limit.
                reduce_tokens (bool): Whether to reduce vocab to observed tokens.
        """
        if args.input_path is None and args.text_input is None:
            raise ValueError("Either input_path or text_input must be provided.")
        if args.input_path and args.text_input:
            raise ValueError("Only one of input_path or text_input can be provided.")

        # Load tokenizer (from cache or download) for consistent tokenization.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=".cache")

        # Load and tokenize input data.
        if args.input_path:
            self.data = self._get_data_from_file(args.input_path)
        else:
            self.data = args.text_input

        self.args = args

        # Tokenize the text (optionally truncating to first_n_tokens).
        print("Starting tokenization...")
        start_time = time.time()

        if args.first_n_tokens is not None:
            # Reduce the input text to approximately first_n_tokens by splitting on spaces
            truncated_data = " ".join(self.data.split(" ", self.args.first_n_tokens)[:self.args.first_n_tokens])

            # Tokenize the truncated data with truncation to ensure we don't exceed the token limit.
            self.data_tokens = self.tokenizer.encode(truncated_data, truncation=True, max_length=self.args.first_n_tokens)

            if len(self.data_tokens) < self.args.first_n_tokens:
                self.args.first_n_tokens = len(self.data_tokens)
                print(f"Reducing first_n_tokens to {self.args.first_n_tokens}, since the input data has fewer tokens.")
            assert len(self.data_tokens) == self.args.first_n_tokens, f"Tokenization produced {len(self.data_tokens)} tokens, expected {self.args.first_n_tokens}."
        else:
            self.data_tokens = self.tokenizer.encode(self.data, truncation=False)
        print(f"Tokenization complete in {(time.time() - start_time):.2f}s. Total number of tokens: {len(self.data_tokens)}")

        self.reduce_tokens = self.args.reduce_tokens

        # Optionally reduce the vocabulary to only tokens observed in the data.
        if self.reduce_tokens:
            # Original behavior: mask based on the first_n_tokens
            self.tokens_list = sorted(list(set(self.data_tokens)))
        else:
            # No reduction
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        print(f"Total distinct tokens: {len(self.tokens_list)}")

    def _get_data_from_file(self, input_path):
        """
        Load data from a given text file.

        Args:
            input_path (str): File path to the input text.

        Returns:
            str: Contents of the file as a string.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Data file not found: {input_path}")
        with open(input_path, 'r') as f:
            return f.read()
        
    def get_data_tokens(self):
        """
        Get the full list of tokens from the loaded data.

        Returns:
            list[int]: List of token IDs from the data.
        """
        return self.data_tokens
    
    def get_bitmap(self):
        """
        Get the bitmap representation of the reduced token set.

        Returns:
            BitMap: A BitMap object representing the reduced token set.
        """

        # Create a BitMap from the tokens list
        bitmap = BitMap(self.tokens_list)
        binary_data = bitmap.serialize()
        return binary_data
    
    def get_args(self):
        """
        Get the arguments used for this data preparation.

        Returns:
            Namespace: The arguments used for this data preparation.
        """
        return self.args

class TokenPredictor:
    def __init__(self, args, bitmap_data):
        """
        Initialize the predictor and load the model.

        Args:
            args (argparse.Namespace): Experiment configuration. Expected fields:
                model_name (str): HuggingFace model name.
                engine (str): Backend engine ("transformer" supported).
                reduce_tokens (bool): Whether to use a reduced token list.
                encoding (str): "AC", "bitpacked", or "huffman".
            bitmap_data (bytes | None): Serialized roaring bitmap of allowed tokens.
        """

        # Load tokenizer and model (from cache or download).
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=".cache")
        self.args = args
        self.device = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS).")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, using CPU.")
        
        print(f"Loading model {args.model_name} on device {self.device}...")
        dtype = "auto"
        
        if args.engine == "transformer":
            if args.is_seq2seq:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, 
                                                              cache_dir=".cache", 
                                                              torch_dtype=dtype)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                             cache_dir=".cache", 
                                                             torch_dtype=dtype)

            self.base_params = self.count_parameters(self.model)[0]
            self.base_size_mb = self.estimate_model_size_mb(self.model)[0]
            self.adapter_params, self.adapter_size_mb = 0, 0

            if args.lora_path is not None:
                self.model = PeftModel.from_pretrained(self.model, args.lora_path, device_map="auto")
                self.adapter_params = self.count_parameters(self.model)[0] - self.base_params
                self.adapter_size_mb = self.estimate_model_size_mb(self.model)[0] - self.base_size_mb
                print(f"Loaded LoRA adapter from {args.lora_path}")

            print(f"Model {args.model_name} loaded with dtype {self.model.dtype}.")

            self.model.eval()
            # Move model to device
            self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported engine: {args.engine}")

        # If bitmap_data is provided, reconstruct the reduced token list.
        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        # Cache indices for fast vocab reduction via index_select.
        self.index_tensor = torch.tensor(self.tokens_list, dtype=torch.long, device=self.device)
        self.reduce_tokens = args.reduce_tokens

    def _update_token_mask(self):
        """
        Updates the index tensor and bitmap for the current self.tokens_list.
        """
        # Build a boolean bitmap and index tensor for fast filtering.
        self.index_tensor = torch.tensor(
            self.tokens_list, dtype=torch.long, device=self.device
        )
        vocab_size = self.tokenizer.vocab_size
        self.token_bitmap = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        self.token_bitmap[self.tokens_list] = True

    def _get_distinct_tokens(self):
        """
        Get distinct tokens from the input data.

        Returns:
            list[int]: Sorted list of distinct token IDs.
        """
        return self.tokens_list

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """
        Run one-step batched inference and return token scores along with timing data.

        This method supports two backends, controlled by ``self.args.engine``:

        - ``"transformer"``: uses a local HuggingFace-style model (``self.model``).
        - If ``enable_kv_cache=False``:
            * Runs the full prompt on every call (no caching).
            * Clears any existing KV cache.
        - If ``enable_kv_cache=True``:
            * Maintains an internal KV cache (``self._past_kv``) and cached context length.
            * If the new prompt is shorter than the cached context, the cache is rebuilt
            from the full prompt.
            * Otherwise, only the last token of each prompt is fed to the model with
            the existing ``past_key_values`` for incremental decoding.

        - ``prompts`` (list of token ID lists) are decoded to strings.
        - Collects per-request logits into a single tensor of shape
            ``(batch_size, vocab_size_or_reduced)``.

        Args:
            prompts (List[List[int]]):
                Batched tokenized prompts. Each element is a list of token IDs for
                one sequence; all sequences in the batch must have the same length.
            enable_kv_cache (bool, optional):
                Whether to use and maintain KV cache for incremental decoding when
                ``self.args.engine == "transformer"``. 

        Returns:
            Tuple[
                List[int],
                torch.Tensor,
                float,
                float
            ]:
                A 4-tuple:
                - ``tokens_list`` (List[int]):
                The list of token IDs corresponding to the (possibly reduced)
                vocabulary used in this run. If ``self.reduce_tokens`` is True,
                this is the reduced token set.
                - ``scores`` (torch.Tensor):
                If ``self.args.encoding == "AC"``:
                    Probability tensor (after softmax) on CPU,
                    shape ``(batch_size, vocab_size_or_reduced)``.
                If ``self.args.encoding in {"bitpacked", "huffman"}``:
                    Raw logits tensor (no softmax), kept on the current device,
                    shape ``(batch_size, vocab_size_or_reduced)``.
                - ``data_copy_time`` (float):
                Total time in seconds spent on data transfer between host and device
                within this call (e.g., moving tensors to GPU or back to CPU).
                - ``softmax_time`` (float):
                Time in seconds spent computing the softmax over logits
                (non-zero only when ``encoding == "AC"``).

            Notes:
            - For arithmetic coding (``encoding="AC"``), the method returns
            probabilities on CPU to be consumed by the arithmetic compressor.
            - For rank-based schemes (``"bitpacked"`` or ``"huffman"``), callers
            are expected to compute ranks directly from the returned logits.
        """

        # Lazily initialize KV-cache state.
        if not hasattr(self, "_past_kv"):
            self._past_kv = None
            self._cached_context_len = 0

        data_copy_time = 0
        with torch.inference_mode():
            if self.args.engine == "transformer":
                # Use local HF-style model for inference
                if not enable_kv_cache:
                    # If not using cache, run the model on the full prompt every time.
                    #t0_data_copy = time.perf_counter()
                    #input_ids = torch.tensor(prompts, device=self.device)
                    #data_copy_time += time.perf_counter() - t0_data_copy

                    #outputs = self.model(input_ids, use_cache=enable_kv_cache)

                    # Ensure cache is cleared when not in use.
                    self._past_kv = None 
                    self._cached_context_len = 0
                    t0_data_copy = time.perf_counter()

                    input_ids = torch.tensor(prompts, device=self.device)
                    data_copy_time += time.perf_counter() - t0_data_copy

                    if self.args.is_seq2seq:
                        # if self.model.config.decoder_start_token_id does not exist, default to 0
                        if self.model.config.decoder_start_token_id is None:
                            self.model.config.decoder_start_token_id = 0

                        decoder_start = self.model.config.decoder_start_token_id

                        decoder_input_ids = torch.full(
                            (input_ids.shape[0], 1),
                            decoder_start,
                            dtype=torch.long,
                            device=self.device)

                        outputs = self.model(
                            input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            use_cache=False)
                    else:
                        outputs = self.model(input_ids, use_cache=enable_kv_cache)

                    self._past_kv = None
                    self._cached_context_len = input_ids.shape[1] #0

                else:
                    # Check if the cache needs to be reset. This happens if the external
                    # context management has shortened the prompt.
                    # Reset if external context management shortened the prompt.
                    reset_cache = len(prompts[0]) < self._cached_context_len

                    if self._past_kv is None or reset_cache:
                        # Rebuild the cache from the full prompt (first step or reset).
                        t0_data_copy = time.perf_counter()
                        input_ids = torch.tensor(prompts, device=self.device)
                        data_copy_time += time.perf_counter() - t0_data_copy
                        outputs = self.model(input_ids, use_cache=True)
                        self._past_kv = outputs.past_key_values
                        self._cached_context_len = 0
                    else:
                        # Incremental step: process only the last token with cache.
                        delta = [row[-1:] for row in prompts]
                        t0_data_copy = time.perf_counter()
                        delta = torch.tensor(delta, device=self.device, dtype=torch.long)
                        data_copy_time += time.perf_counter() - t0_data_copy

                        outputs = self.model(delta, past_key_values=self._past_kv, use_cache=True)
                        self._past_kv = outputs.past_key_values
                        self._cached_context_len += 1
                logits = outputs.logits[:, -1, :]
            else:
                raise ValueError(f"Unsupported engine: {self.args.engine}")

            # print(f"logits shape: {logits.shape}")
            # Optionally reduce logits to the active token subset.
            if getattr(self, "reduce_tokens", False):
                logits = logits.index_select(1, self.index_tensor.to(logits.device))

            softmax_time = 0.0
            if self.args.encoding == "AC":
                # For arithmetic coding, convert logits to probabilities on CPU.
                t0_softmax = time.perf_counter()
                probs = torch.softmax(logits, dim=-1)
                softmax_time = time.perf_counter() - t0_softmax

                t0_data_copy = time.perf_counter()
                probs_cpu = probs.cpu()
                data_copy_time += time.perf_counter() - t0_data_copy

                return self.tokens_list, probs_cpu, data_copy_time, softmax_time
            elif self.args.encoding in ("bitpacked", "huffman"):
                # For rank-based schemes, return raw logits on the current device.
                return self.tokens_list, logits, data_copy_time, softmax_time
            else:
                raise NotImplementedError(f"Encoding method '{self.args.encoding}' is not implemented.")

    def detokenize(self, token_ids):
        """
        Convert a list of token IDs back to a string.

        Args:
            token_ids (list[int]): List of token IDs.

        Returns:
            str: Decoded string.
        """
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        """
        Get the token ID at a given index in the reduced token list.

        Args:
            token_id (int): Index in the reduced token list.

        Returns:
            int: Token ID corresponding to the given index.
        """
        return self.tokens_list[token_id]
    
    def count_parameters(self, model):
        """
        Count total and adapter parameters in the model.

        Args:
            model (torch.nn.Module): The model to analyze.

        Returns:
            Tuple[int, int]: Total parameters and adapter parameters.
        """
        total_params = sum(p.numel() for p in model.parameters())
        adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_params = 0
        trainable_params = 0

        for p in model.parameters():
            numel = p.numel()
            total_params += numel
            if p.requires_grad:
                trainable_params += numel
        return total_params, trainable_params
    
    def estimate_model_size_mb(self, model):
        model = self.model
        total_bytes = 0
        trainable_bytes = 0

        for p in model.parameters():
            bytes_ = p.numel() * p.element_size()
            total_bytes += bytes_
            if p.requires_grad:
                trainable_bytes += bytes_
        return total_bytes / (1024 ** 2), trainable_bytes / (1024 ** 2)
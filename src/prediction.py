from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import math
from pyroaring import BitMap
import time

class TokenDataPreparer:
    def __init__(self, args):
        """
        Initialize the TokenDataPreparer class.
        """
        if args.input_path is None and args.text_input is None:
            raise ValueError("Either input_path or text_input must be provided.")
        if args.input_path and args.text_input:
            raise ValueError("Only one of input_path or text_input can be provided.")

        # Load tokenizer and model from cache or download
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=".cache")

        # Load and tokenize input data
        if args.input_path:
            self.data = self._get_data_from_file(args.input_path)
        else:
            self.data = args.text_input

        self.args = args

        # Tokenize the text
        print("Starting tokenization...")
        start_time = time.time()
        if args.first_n_tokens is not None:
            truncated_data = " ".join(self.data.split(" ", self.args.first_n_tokens)[:self.args.first_n_tokens])
            self.data_tokens = self.tokenizer.encode(truncated_data, truncation=True, max_length=self.args.first_n_tokens)
            if len(self.data_tokens) < self.args.first_n_tokens:
                self.args.first_n_tokens = len(self.data_tokens)
                print(f"Reducing first_n_tokens to {self.args.first_n_tokens}, since the input data has fewer tokens.")
            assert len(self.data_tokens) == self.args.first_n_tokens, f"Tokenization produced {len(self.data_tokens)} tokens, expected {self.args.first_n_tokens}."
        else:
            self.data_tokens = self.tokenizer.encode(self.data, truncation=False)
        print(f"Tokenization complete in {(time.time() - start_time):.2f}s. Total number of tokens: {len(self.data_tokens)}")

        self.reduce_tokens = self.args.reduce_tokens

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
        Initialize the TokenPredictor class.
        """

        # Load tokenizer and model from cache or download
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

        # Pick dtype based on model name
        if "FP8" in args.model_name.upper():
            dtype = "auto"  # Use FP8 for FP8 models
        else:
            dtype = "auto"  # Let HF auto-detect dtype for non-FP8 models

        if args.engine == "transformer":
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                cache_dir=".cache",
                dtype=dtype,
                device_map="auto"
            )

            print(f"Model {args.model_name} loaded with dtype {self.model.dtype}.")

            self.model.eval()
            # Move model to device
            self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported engine: {args.engine}")

        # --- If bitmap_data is provided, reconstruct tokens_list & index_tensor ---
        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        self.index_tensor = torch.tensor(self.tokens_list, dtype=torch.long, device=self.device)
        self.reduce_tokens = args.reduce_tokens

    def _set_active_chunk(self, chunk_index):
        """
        Sets the active token mask based on the tokens in the specified chunk.
        Also returns the bitmap and its size for the current chunk.
        """
        if not self.reduce_tokens or self.chunk_size is None:
            print("Warning: set_active_chunk is only effective when reduce_tokens is True and chunk_size is set.")
            return None, 0

        start_index = chunk_index * self.chunk_size
        end_index = min(start_index + self.chunk_size, len(self.data_tokens))
        chunk_tokens = self.data_tokens[start_index:end_index]

        if not chunk_tokens:
            return None, 0

        self.tokens_list = sorted(list(set(chunk_tokens)))
        self._update_token_mask()
        
        print(f"\nActivated chunk {chunk_index}: tokens {start_index}-{end_index}. Distinct tokens: {len(self.tokens_list)}")
        
        # Get the roaring bitmap for the current chunk's token list
        bitmap = BitMap(self.tokens_list)
        binary_data = bitmap.serialize()
        size_bytes = len(binary_data) # pyroaring serialize returns bytes
        
        return binary_data, size_bytes

    def _update_token_mask(self):
        """
        Updates the index tensor and bitmap for the current self.tokens_list.
        """
        self.index_tensor = torch.tensor(
            self.tokens_list, dtype=torch.long, device=self.device
        )
        vocab_size = self.tokenizer.vocab_size
        self.token_bitmap = torch.zeros(vocab_size, dtype=torch.bool)
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

        if not hasattr(self, "_past_kv"):
            self._past_kv = None
            self._cached_context_len = 0

        data_copy_time = 0
        with torch.inference_mode():
            if self.args.engine == "transformer":
                # Use local HF-style model for inference
                if not enable_kv_cache:
                    # If not using cache, run the model on the full prompt every time.
                    t0_data_copy = time.perf_counter()
                    input_ids = torch.tensor(prompts, device=self.device)
                    data_copy_time += time.perf_counter() - t0_data_copy
                    outputs = self.model(input_ids, use_cache=enable_kv_cache)
                    # Ensure cache is cleared when not in use.
                    self._past_kv = None
                    self._cached_context_len = 0
                else:
                    # Check if the cache needs to be reset. This happens if the external
                    # context management has shortened the prompt.
                    reset_cache = len(prompts[0]) < self._cached_context_len

                    if self._past_kv is None or reset_cache:
                        # Rebuild the cache from the full prompt.
                        t0_data_copy = time.perf_counter()
                        input_ids = torch.tensor(prompts, device=self.device)
                        data_copy_time += time.perf_counter() - t0_data_copy
                        outputs = self.model(input_ids, use_cache=True)
                        self._past_kv = outputs.past_key_values
                        self._cached_context_len = 0
                    else:
                        # Incremental step: process only the last token using the existing cache.
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
            if getattr(self, "reduce_tokens", False):
                logits = logits.index_select(1, self.index_tensor.to(logits.device))

            softmax_time = 0.0
            if self.args.encoding == "AC":
                # For arithmetic coding, convert logits to probabilities on CPU
                t0_softmax = time.perf_counter()
                probs = torch.softmax(logits, dim=-1)
                softmax_time = time.perf_counter() - t0_softmax

                t0_data_copy = time.perf_counter()
                probs_cpu = probs.cpu()
                data_copy_time += time.perf_counter() - t0_data_copy

                return self.tokens_list, probs_cpu, data_copy_time, softmax_time
            elif self.args.encoding in ("bitpacked", "huffman"):
                # For rank-based schemes, return raw logits on the current device
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

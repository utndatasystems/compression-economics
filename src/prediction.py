"""
Token preparation and next-token prediction helpers for compression experiments.

This module provides:
- TokenDataPreparer: loads raw text, tokenizes, and optionally reduces the vocabulary
  to the set of tokens observed in the input (for bitmap-based masking).
- TokenPredictor: runs batched, one-step LLM inference with optional KV caching and
  returns logits or probabilities depending on the encoding scheme.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, MambaForCausalLM
import os
import tarfile
import torch
from pyroaring import BitMap
import time
from peft import PeftModel


def get_token_predictor(args, bitmap_data):
    """Create the predictor implementation for the selected inference engine."""
    if args.engine == "transformer":
        return TokenPredictor(args, bitmap_data)
    if args.engine == "vllm":
        from src.vllm_prediction_v3 import VLLMTokenPredictorV3
        return VLLMTokenPredictorV3(args, bitmap_data)
    raise ValueError(f"Unsupported engine: {args.engine}")

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
        if args.is_mamba:
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        else:
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

    def _get_data_from_file(self, input_path, concat_all=True):
        """
        Load data from a given text file.

        Args:
            input_path (str): File path to the input text.
            concat_all (bool): Whether to concatenate all files in a tar archive.

        Returns:
            str: Contents of the file as a string.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Data file not found: {input_path}")
        
        if input_path.endswith(".tar"):
            # Open the tar file
            with tarfile.open(input_path, "r") as tar:
                text_members = [m for m in tar.getmembers() if m.isfile()]

                # If no files are found in the tar archive, raise an error.
                if not text_members:
                    raise ValueError("No files found in the tar archive")
                
                # If concat_all is True, concatenate the contents of all files in the tar archive.
                if concat_all:
                    contents = []
                    print(f"Concatenating {len(text_members)} files from tar archive...")
                    for member in text_members:
                        file_obj = tar.extractfile(member)
                        if file_obj is None:
                            raise ValueError(f"Failed to extract {member.name} from tar archive")
                        contents.append(file_obj.read().decode("utf-8"))
                    return "\n".join(contents)
                
                else:
                    # Read the first file inside the tar
                    print(f"Reading {text_members[0].name} from tar archive...")
                    file_obj = tar.extractfile(text_members[0])
                    if file_obj is None:
                        raise ValueError("Failed to extract file from tar archive")
                    return file_obj.read().decode("utf-8")
        else:
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

        # Load tokenizer (from cache or download) for consistent tokenization.
        if args.is_mamba:
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        else:
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
                
                self.start_token_id = self.model.config.decoder_start_token_id
                
            elif args.is_mamba:
                self.model = MambaForCausalLM.from_pretrained(args.model_name,
                                                                cache_dir=".cache",
                                                                torch_dtype=dtype)
                                                                #ignore_mismatched_sizes=True, might lead to worse performance of mamba
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
            if self.device is not None:
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
            self.tokens_list, dtype=torch.long, device=self.device)
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
    

    def _pad_input(self, prompts, output_tensor=False):
        """
        Pads input to the maximum sequence length and creates an attention mask for variable-length prompts.
        Returns:
            padded_prompts: List[List[int]]
            attention_mask: List[List[int]]
        """
        max_len = max(len(seq) for seq in prompts)

        # Prefer tokenizer pad token, fallback to 0
        pad_token = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token is None:
            pad_token = 0

        padded_prompts = []
        attention_mask = []

        for seq in prompts:
            seq_len = len(seq)
            pad_len = max_len - seq_len

            padded_prompts.append(seq + [pad_token] * pad_len)
            attention_mask.append([1] * seq_len + [0] * pad_len)

        # Converting to tensor for efficiency.
        if output_tensor:
            padded_prompts = torch.tensor(padded_prompts, device=self.device)
            attention_mask = torch.tensor(attention_mask, device=self.device)

        return padded_prompts, attention_mask
    

    def _check_rectangular(self, prompts):
        """Returns True if all sequences have the same length."""
        assert isinstance(prompts, list) and all(isinstance(seq, list) for seq in prompts), "Input prompts must be a list of lists of token IDs."

        lengths = [len(seq) for seq in prompts]
        return len(set(lengths)) == 1

    
    def _finalize_batched_scores(self, logits, data_copy_time):
        """
        Apply optional vocab reduction and return scores in the configured format.
        Follows last in run_batched_inference() after obtaining the raw logits from the model.

        Args:
            logits (torch.Tensor): Tensor of shape (batch_size, vocab_size).
            data_copy_time (float): Time already spent on host/device transfers.

        Returns:
            tuple[list[int], torch.Tensor, float, float]:
                Same return format as run_batched_inference().
        """

        assert isinstance(logits, torch.Tensor), f"Expected logits to be a torch.Tensor, got {type(logits)}"
        #print(f"Logits shape: {logits.shape}, expected (batch_size, vocab_size) where vocab_size is {self.model.config.vocab_size}")
        assert logits.dim() == 2, f"Expected logits of shape (batch_size, vocab_size), got {logits.shape}"

        if getattr(self, "reduce_tokens", False):
            logits = logits.index_select(1, self.index_tensor.to(logits.device))

        softmax_time = 0.0
        if self.args.encoding in {"AC", "PMATIC"}:
            t0_softmax = time.perf_counter()
            probs = torch.softmax(logits, dim=-1)
            softmax_time = time.perf_counter() - t0_softmax

            t0_data_copy = time.perf_counter()
            probs_cpu = probs.cpu()
            data_copy_time += time.perf_counter() - t0_data_copy

            return self.tokens_list, probs_cpu, data_copy_time, softmax_time

        if self.args.encoding in ("bitpacked", "huffman"):
            return self.tokens_list, logits, data_copy_time, softmax_time

        raise NotImplementedError(
            f"Encoding method '{self.args.encoding}' is not implemented.")
    

    def run_batched_inference_cachefree(self, prompts):
        """
        Run a full forward pass for every prompt without reusing KV cache state.
        Can handle uneven prompt lengths by padding and using attention masks.

        This is intended as a correctness baseline for comparing against the
        cached path in run_batched_inference().

        input: 
        - prompts: list of tokenized prompts (list of list of ints)

        output:
        - tokens_list: list of token IDs corresponding to the score columns in the returned tensor.
        - scores: tensor of shape (batch_size, vocab_size_or_reduced_vocab_size)
        - data_copy_time: approximate time spent moving tensors between host and device during this call.
        - softmax_time: time spent computing the softmax (only non-zero when encoding == "AC" / "PMATIC").
        """
        data_copy_time = 0.0

        with torch.inference_mode():
            if self.args.engine != "transformer":
                raise ValueError(f"Unsupported engine: {self.args.engine}")

            t0_data_copy = time.perf_counter()
            
            if self._check_rectangular(prompts):
                padded_prompts = prompts
                attention_mask = [[1] * len(seq) for seq in prompts]
            else:
                # Pad only if needed
                padded_prompts, attention_mask = self._pad_input(prompts)

            # Convert to tensor 
            input_ids = torch.tensor(padded_prompts, device=self.device)
            attention_mask = torch.tensor(attention_mask, device=self.device)

            data_copy_time += time.perf_counter() - t0_data_copy

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False)

            # Compute last valid token index per batch from attention mask to extract the correct logits.
            last_token_idx = attention_mask.sum(dim=1) - 1 # (batch_size,)

            logits = outputs.logits[
                torch.arange(input_ids.size(0), device=self.device),
                last_token_idx,
                :]

        assert logits.dim() == 2, f"Expected logits of shape (batch_size, vocab_size), got {logits.shape}"
        assert isinstance(logits, torch.Tensor), f"Expected logits to be a torch.Tensor, got {type(logits)}"

        return self._finalize_batched_scores(logits, data_copy_time)
    

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        # TODO: correct the caching behaviour and make it work with mixed length prompts
        """
        Run a single next-token prediction step for a batch of tokenized prompts.

        The method executes one forward pass and returns the model scores for the
        next token of each prompt. Depending on the configured encoding, the scores
        are returned either as probabilities or as raw logits.

        Supported behavior
        ------------------
        Currently, only self.args.engine == "transformer" is supported.

        Caching behavior
        ----------------
        When enable_kv_cache is disabled, the full prompt batch is passed to the
        model on every call and any previously stored cache is cleared.

        When enable_kv_cache is enabled, the method maintains an internal
        key/value cache in self._past_kv:

        - On the first cached call, or after a cache reset, the full prompt batch is
        forwarded through the model and a new cache is created.
        - On subsequent cached calls, only the final token of each prompt is passed
        to the model together with the stored ``past_key_values``.
        - If the incoming prompt appears shorter than the cached context length, the
        cache is rebuilt from the full prompt batch.

        Input shape
        -----------
        prompts is a list of token-id sequences, one per batch element.

        - If all prompts have equal length, they are converted directly to a tensor.
        - If prompt lengths differ, they are padded and an attention mask is passed
        to the model.

        Output format
        -------------
        The returned tensor always has shape::

            (batch_size, vocab_size_or_reduced_vocab_size)

        where the second dimension is either the full vocabulary or the reduced
        vocabulary defined by self.tokens_list.

        Args:
            prompts (list[list[int]]):
                Batch of tokenized prompts. Each inner list is one prompt expressed
                as token IDs.
            enable_kv_cache (bool, optional):
                Whether to use the internal KV cache for incremental decoding.
                Defaults to True.

        Returns:
            tuple[list[int], torch.Tensor, float, float]:
                A 4-tuple containing:

                - ``tokens_list``:
                The token IDs corresponding to the score columns in the returned
                tensor. This is the reduced token list when token reduction is
                enabled, otherwise the full vocabulary index range.
                - ``scores``:
                If ``self.args.encoding == "AC" or "PMATIC"``, a probability tensor on CPU.
                If ``self.args.encoding in {"bitpacked", "huffman"}``, a logits
                tensor on the active device.
                - ``data_copy_time``:
                Approximate time spent moving tensors between host and device
                during this call.
                - ``softmax_time``:
                Time spent computing the softmax. This is only non-zero when
                ``encoding == "AC"``.

        Raises:
            ValueError:
                If ``self.args.engine`` is not supported.
            NotImplementedError:
                If ``self.args.encoding`` is not one of the implemented modes.

        Notes:
            - For arithmetic coding (``encoding="AC"``), probabilities are returned
            on CPU because downstream compression code consumes probabilities.
            - For rank-based encodings (``"bitpacked"`` and ``"huffman"``), raw
            logits are returned so the caller can rank tokens directly.
        """

        if enable_kv_cache is False:
            # run_batched_inference_cachefree instead
            return self.run_batched_inference_cachefree(prompts)
        
        # Initialize cache state on first use.
        if not hasattr(self, "_past_kv"):
            self._past_kv = None
            self._cached_context_len = 0

        data_copy_time = 0

        with torch.inference_mode():
            if self.args.engine == "transformer":
                if not enable_kv_cache:
                    return self.run_batched_inference_cachefree(prompts)

                else:
                    # If the prompt is now shorter than the cached context, the
                    # previously stored cache can no longer be reused safely.
                    reset_cache = len(prompts[0]) < self._cached_context_len

                    if self._past_kv is None or reset_cache:
                        # Build or rebuild the cache from the full prompt batch.
                        t0_data_copy = time.perf_counter()

                        # Variable-length prompts are padded and accompanied by an
                        # attention mask so padding tokens are ignored.
                        if len(set(len(seq) for seq in prompts)) != 1:
                            padded_prompts, attention_mask = self._pad_input(prompts)

                            input_ids = torch.tensor(padded_prompts, device=self.device)
                            attention_mask = torch.tensor(attention_mask, device=self.device)

                            data_copy_time += time.perf_counter() - t0_data_copy
                            outputs = self.model(
                                input_ids,
                                use_cache=True,
                                attention_mask=attention_mask,)
                        else:
                            # Equal-length prompts can be forwarded directly.
                            input_ids = torch.tensor(prompts, device=self.device)
                            data_copy_time += time.perf_counter() - t0_data_copy
                            outputs = self.model(input_ids, use_cache=True)

                        self._past_kv = outputs.past_key_values
                        self._cached_context_len = input_ids.shape[1]
                    else:
                        # Reuse the existing cache by feeding only the newly added
                        # final token of each prompt.
                        delta = [row[-1:] for row in prompts]

                        t0_data_copy = time.perf_counter()
                        delta = torch.tensor(delta, device=self.device, dtype=torch.long)
                        data_copy_time += time.perf_counter() - t0_data_copy

                        outputs = self.model(
                            delta,
                            past_key_values=self._past_kv,
                            use_cache=True,
                        )
                        self._past_kv = outputs.past_key_values
                        self._cached_context_len += delta.shape[1]

                # Use only the scores for the final position in each sequence.
                logits = outputs.logits[:, -1, :]
            else:
                raise ValueError(f"Unsupported engine: {self.args.engine}")

        return self._finalize_batched_scores(logits, data_copy_time)
            

    def generate_draft(self, prompt, k=None, enable_kv_cache=False, full_draft=False):                 
        #ToDo: later implement kv cashing for repeated calls on iterative prompts
        """
        Generate k draft tokens autoregressively for a batch of equal-length tokenized prompts.

        Args:
            prompts: List[List[int]] of shape (batch_size, prompt_len)
            k: Number of draft steps. Defaults to self.args.spec_k or 1.
            enable_kv_cache: Reserved for future use.
            full_draft: If True, also return the extended prompts.

        Returns:
            draft_tokens: List[List[int]] of shape (batch_size, k)
            draft_scores: torch.Tensor of shape (k, batch_size, vocab_size)
            data_copy_time: float
            softmax_time: float
            current_prompts: only if full_draft=True
        """

        if k is None:
            k = getattr(self.args, "spec_k", 1)
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")

        assert isinstance(prompt, list) and all(isinstance(row, list) for row in prompt), "Input prompt must be a list of lists of token IDs."
        
        # If k=1, we can just call the existing inference method once
        if k == 1:
            tokens_list, scores, data_copy_time, softmax_time = self.run_batched_inference_cachefree(
                [row[:] for row in prompt])
            next_idx = torch.argmax(scores, dim=-1)
            draft_tokens = [[tokens_list[idx.item()]] for idx in next_idx]
            draft_scores = scores.unsqueeze(0).detach().cpu()   # shape (1, B, V)

            if full_draft:
                current_prompt = [row + [tok[0]] for row, tok in zip(prompt, draft_tokens)]
                return draft_tokens, draft_scores, data_copy_time, softmax_time, current_prompt

            # assert correct types and shapes of outputs
            assert isinstance(draft_tokens, list) and all(isinstance(row, list) for row in draft_tokens), "Draft tokens must be a list of lists of token IDs."
            assert draft_scores.shape == (1, len(prompt), len(self.tokens_list)), f"Expected draft_scores shape (1, {len(prompt)}, {len(self.tokens_list)}), got {draft_scores.shape}"

            return draft_tokens, draft_scores, data_copy_time, softmax_time
    
        elif k > 1:
            #TODO: 

            """ assert all lists in list are of equal length
            #if len(prompt) > 0:
                prompt_length = len(prompt[0])
                assert all(len(row) == prompt_length for row in prompt), "All prompts must be of equal length." """ 
            
            B, V = len(prompt), len(self.tokens_list) # number of batches and vocab size (or reduced vocab size)
            draft_tokens = [[] for _ in range(B)] # make it list of lists to store tokens for each batch element
            draft_scores = torch.zeros((k, B, V), device=self.device) # pre-allocate tensor for scores with shape (k, batch_size, vocab_size_or_reduced_vocab_size)
            
            current_prompt = [row[:] for row in prompt]

            total_data_copy_time, total_softmax_time = 0.0, 0.0

            for ki in range(k):
                # Run inference for the current prompt and get scores for the next token.
                tokens_list, scores, data_copy_time, softmax_time = self.run_batched_inference_cachefree([row[:] for row in current_prompt],)
                assert scores.shape == (B, V), f"Expected scores shape ({B}, {V}), got {scores.shape}" # torch.Size([2, 50257]) 

                # Accumulate timing metrics across steps.
                total_data_copy_time += data_copy_time
                total_softmax_time += softmax_time

                # Select the next token based on the scores
                if isinstance(scores, torch.Tensor):
                    next_idx = torch.argmax(scores, dim=-1) # get the index of the highest scoring token
                    assert next_idx.shape == (B,), f"Expected next_idx shape ({B},), got {next_idx.shape}"
                else: 
                    raise TypeError(f"Expected torch.Tensor for scores, got {type(scores)}")

                # Get the next tokens for each batch element, all batches share tokens_list
                next_tokens = [tokens_list[next] for next in next_idx.flatten().tolist()] # convert to list of ints
                assert len(next_tokens) == len(prompt), f"Expected next_tokens length {len(prompt)}, got {len(next_tokens)}"

                current_prompt = [row + [next_token] for row, next_token in zip(current_prompt, next_tokens)]

                for i, next_token in enumerate(next_tokens):
                    draft_tokens[i].append(next_token)
                
                draft_scores[ki] = scores.to(self.device)  # move to cpu for easier handling later

            # Wrap draft tokens in one additional layer for each k step, so that the output is a list of lists of lists: batch_size x k x tokens_per_step
            draft_tokens = [[draft_tokens[i][j] for j in range(k)] for i in range(len(prompt))]
            
            assert len(draft_tokens) == len(prompt), f"Expected draft_tokens length {len(prompt)}, got {len(draft_tokens)}"
            assert draft_scores.shape == (k, len(prompt), len(self.tokens_list)), f"Expected draft_scores shape ({k}, {len(prompt)}, {len(self.tokens_list)}), got {draft_scores.shape}"             #assert that shape is k, B, V 

            if full_draft:
                return draft_tokens, draft_scores, total_data_copy_time, total_softmax_time, current_prompt
            else:
                return draft_tokens, draft_scores, total_data_copy_time, total_softmax_time


    def greedy_next_token(self, prompt, enable_kv_cache=True):
        """
        Predict one next token for a single prompt.

        Args:
            prompt: List[int]
        Returns:
            next_token: int
            scores: torch.Tensor of shape (vocab_size_or_reduced_vocab_size,)
            data_copy_time: float
            softmax_time: float
        """
        tokens_list, scores, data_copy_time, softmax_time = self.run_batched_inference(
            [prompt],
            enable_kv_cache=enable_kv_cache,)

        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for scores, got {type(scores)}")

        next_idx = torch.argmax(scores[0]).item()
        next_token = tokens_list[next_idx]

        return next_token, scores[0], data_copy_time, softmax_time


    def accept_reject_loop(self, base_prompts, draft_tokens):
        """
        Verify drafted tokens against the verifier model. Each batch element is processed independently in a greedy accept-reject loop.

        Args:
            base_prompts: List[List[int]] of shape (batch_size, prompt_len)
                The prompts BEFORE draft generation started.
            draft_tokens: List[List[int]] of shape (batch_size, k)
                The drafted tokens for each batch element.

        Returns:
            accepted_tokens: List[List[int]]
                Longest accepted prefix for each batch item.
            accepted_lengths: List[int]
                Number of accepted draft tokens per batch item.
            updated_prompts: List[List[int]]
                Prompt after applying accepted prefix and, if rejection occurs,
                appending the verifier token at the rejection position.
            rejected: List[bool]
                True if that batch item had a rejection.
            total_data_copy_time: float
            total_softmax_time: float
        """
        if not isinstance(base_prompts, list) or not all(isinstance(p, list) for p in base_prompts):
            raise TypeError("base_prompts must be a list of token-id lists")

        if not isinstance(draft_tokens, list) or not all(isinstance(p, list) for p in draft_tokens):
            raise TypeError("draft_tokens must be a list of token-id lists")

        if len(base_prompts) != len(draft_tokens):
            raise ValueError(
                f"Batch size mismatch: got {len(base_prompts)} prompts and {len(draft_tokens)} draft rows"
            )

        batch_size = len(base_prompts)

        accepted_tokens = [[] for _ in range(batch_size)]
        accepted_lengths = [0 for _ in range(batch_size)]
        updated_prompts = [p[:] for p in base_prompts]
        rejected = [False for _ in range(batch_size)]

        total_data_copy_time = 0.0
        total_softmax_time = 0.0

        for i in range(batch_size):
            prompt_i = base_prompts[i][:]
            draft_i = draft_tokens[i]

            # Important: verifier cache is per sequence here, so reset for each batch item.
            self.reset_kv_cache()

            for j, drafted_tok in enumerate(draft_i):
                pred_tok, _, data_copy_time, softmax_time = self.greedy_next_token(
                    prompt_i,
                    enable_kv_cache=True,
                )
                total_data_copy_time += data_copy_time
                total_softmax_time += softmax_time

                if pred_tok == drafted_tok:
                    # Accept this drafted token
                    accepted_tokens[i].append(drafted_tok)
                    accepted_lengths[i] += 1
                    prompt_i.append(drafted_tok)
                else:
                    # Reject at first mismatch:
                    # append verifier token instead, then stop for this batch item
                    rejected[i] = True
                    prompt_i.append(pred_tok)
                    break

            updated_prompts[i] = prompt_i

        return (
            accepted_tokens,
            accepted_lengths,
            updated_prompts,
            rejected,
            total_data_copy_time,
            total_softmax_time,
        )

    def speculative_decode(self, prompts, max_new_tokens, k=None):
        """
        Simple speculative decoding loop with restart-on-rejection.

        Args:
            prompts: List[List[int]]
            max_new_tokens: int, maximum number of newly generated tokens to append to each prompt
            k: draft length per cycle

        Returns:
            final_prompts: List[List[int]]
            generated_tokens: List[List[int]]   # only newly generated tokens
        """
        if k is None:
            k = getattr(self.args, "spec_k", 1)
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        batch_size = len(prompts)
        current_prompts = [p[:] for p in prompts]
        original_lengths = [len(p) for p in prompts]

        generated_tokens = [[] for _ in range(batch_size)]

        while True:
            # Stop when every batch element has enough newly generated tokens
            done = all(len(g) >= max_new_tokens for g in generated_tokens)
            if done:
                break

            # Optionally shrink k for sequences close to completion
            remaining = [max_new_tokens - len(g) for g in generated_tokens]
            current_k = min(k, max(remaining))

            # 1. Draft
            draft_tokens, _, _, _ = self.generate_draft(
                current_prompts,
                k=current_k,
                enable_kv_cache=False,
                full_draft=False,
            )

            # 2. Verify + restart-on-reject
            (
                accepted_tokens,
                accepted_lengths,
                updated_prompts,
                rejected,
                _,
                _,
            ) = self.accept_reject_loop(current_prompts, draft_tokens)

            current_prompts = updated_prompts

            # 3. Refresh generated_tokens from prompt deltas
            for i in range(batch_size):
                new_tokens = current_prompts[i][original_lengths[i]:]
                generated_tokens[i] = new_tokens[:max_new_tokens]

        # Trim final prompts in case we overshot
        final_prompts = []
        for i in range(batch_size):
            final_prompt = prompts[i] + generated_tokens[i]
            final_prompts.append(final_prompt)

        return final_prompts, generated_tokens


    def reset_kv_cache(self):
        # Helper method to reset the KV cache state, e.g. when the prompt context is shortened.
        self._past_kv = None
        self._cached_context_len = 0


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
    



    
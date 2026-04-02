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
    
    def pad_input(self, prompts):
        """Pads input to length of maximum sequence and 
           creates an attention mask for the model to ingore the padded input"""
        
        max_len = max(len(seq) for seq in prompts)

        # choose padding value (e.g. 0 or tokenizer.pad_token_id)
        pad_token = 0 #self.tokenizer.pad_token_id

        # pad all sequences
        padded_prompts = [
            seq + [pad_token] * (max_len - len(seq))
            for seq in prompts]
        
        max_len = max(len(seq) for seq in prompts)
        attention_mask = [[1] * len(seq) + [0] * (max_len - len(seq))
                          for seq in prompts]
        
        return padded_prompts, attention_mask

    def run_batched_inference(self, prompts, enable_kv_cache=True):
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
                If ``self.args.encoding == "AC"``, a probability tensor on CPU.
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

        # Initialize cache state on first use.
        if not hasattr(self, "_past_kv"):
            self._past_kv = None
            self._cached_context_len = 0

        data_copy_time = 0
        with torch.inference_mode():
            if self.args.engine == "transformer":
                if not enable_kv_cache:
                    # Disable caching for this call and clear any previously stored state.
                    self._past_kv = None
                    self._cached_context_len = 0

                    # Materialize the full prompt batch on the target device.
                    t0_data_copy = time.perf_counter()
                    input_ids = torch.tensor(prompts, device=self.device)
                    data_copy_time += time.perf_counter() - t0_data_copy

                    if self.args.is_seq2seq:
                        # Seq2seq models need an explicit decoder start token for
                        # one-step decoding. Fall back to 0 if the config does not
                        # provide one.
                        if self.model.config.decoder_start_token_id is None:
                            self.model.config.decoder_start_token_id = 0

                        decoder_start = self.model.config.decoder_start_token_id
                        decoder_input_ids = torch.full(
                            (input_ids.shape[0], 1),
                            decoder_start,
                            dtype=torch.long,
                            device=self.device,
                        )

                        outputs = self.model(
                            input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            use_cache=False,
                        )
                    else:
                        # Causal LM path without caching: run the full prompt batch.
                        outputs = self.model(input_ids, use_cache=False)

                    # No cache is retained in the no-cache path.
                    self._past_kv = None
                    self._cached_context_len = input_ids.shape[1]

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
                            padded_prompts, attention_mask = self.pad_input(prompts)

                            input_ids = torch.tensor(padded_prompts, device=self.device)
                            attention_mask = torch.tensor(attention_mask, device=self.device)

                            data_copy_time += time.perf_counter() - t0_data_copy
                            outputs = self.model(
                                input_ids,
                                use_cache=True,
                                attention_mask=attention_mask,
                            )
                        else:
                            # Equal-length prompts can be forwarded directly.
                            input_ids = torch.tensor(prompts, device=self.device)
                            data_copy_time += time.perf_counter() - t0_data_copy
                            outputs = self.model(input_ids, use_cache=True)

                        self._past_kv = outputs.past_key_values
                        self._cached_context_len = 0
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
                        self._cached_context_len += 1

                # Use only the scores for the final position in each sequence.
                logits = outputs.logits[:, -1, :]
            else:
                raise ValueError(f"Unsupported engine: {self.args.engine}")

            # Restrict the vocabulary to the active token subset when token
            # reduction is enabled.
            if getattr(self, "reduce_tokens", False):
                logits = logits.index_select(1, self.index_tensor.to(logits.device))

            softmax_time = 0.0
            if self.args.encoding == "AC":
                # Arithmetic coding expects probabilities rather than raw logits.
                t0_softmax = time.perf_counter()
                probs = torch.softmax(logits, dim=-1)
                softmax_time = time.perf_counter() - t0_softmax

                # Move probabilities to CPU for downstream consumption.
                t0_data_copy = time.perf_counter()
                probs_cpu = probs.cpu()
                data_copy_time += time.perf_counter() - t0_data_copy

                return self.tokens_list, probs_cpu, data_copy_time, softmax_time

            elif self.args.encoding in ("bitpacked", "huffman"):
                # Rank-based encodings operate directly on logits.
                return self.tokens_list, logits, data_copy_time, softmax_time

            else:
                raise NotImplementedError(
                    f"Encoding method '{self.args.encoding}' is not implemented."
                )
            
        
    def generate_draft(self, prompt, k=None, enable_kv_cache=True):
        """
        Generate k draft tokens autoregressively for a single prompt.

        Returns a tuple similar in spirit to run_batched_inference():
            (   draft_tokens,       # List[int], length k
                draft_scores,       # torch.Tensor, shape (k, vocab_size_or_reduced)
                data_copy_time,     # float
                softmax_time,       # float
            )
        """

        if k is None:
            k = getattr(self.args, "spec_k", 1)

        draft_tokens = []
        scores_per_step = []
        current_prompt = prompt.copy()

        total_data_copy_time = 0.0
        total_softmax_time = 0.0

        for _ in range(k):
            tokens_list, scores, data_copy_time, softmax_time = self.run_batched_inference(
                [current_prompt.copy()],    
                enable_kv_cache=enable_kv_cache,)

            total_data_copy_time += data_copy_time
            total_softmax_time += softmax_time

            # scores has shape (1, vocab_size_or_reduced)
            step_scores = scores[0]
            scores_per_step.append(step_scores)

            if isinstance(step_scores, torch.Tensor):
                next_idx = torch.argmax(step_scores).item()
            else:
                raise TypeError(f"Expected torch.Tensor for scores, got {type(step_scores)}")

            next_token = tokens_list[next_idx]
            draft_tokens.append(int(next_token))
            current_prompt.append(int(next_token))

        if len(scores_per_step) > 0:
            draft_scores = torch.stack(scores_per_step, dim=0)
        else:
            if self.args.encoding == "AC":
                draft_scores = torch.empty((0, len(self.tokens_list)), dtype=torch.float32)
            else:
                draft_scores = torch.empty((0, len(self.tokens_list)), device=self.device)

        return draft_tokens, draft_scores, total_data_copy_time, total_softmax_time


    def generate_draft_old(self, prompt, k=None):
        """
        Generate a sequence of draft tokens for a given prompt, iteratively producing k tokens.

        Args:
            prompt (list[int]): Token IDs for the input prompt.
                k (int, optional): Number of speculative tokens to generate. Defaults to self.args.spec_k.

        Returns:
            list[int]: List of k generated token IDs.
        """

        if k is None:
            k = getattr(self.args, "spec_k", 1)

        draft_tokens = []
        current_prompt = prompt.copy()  # do not modify original prompt

        for _ in range(k):
            tokens_list, scores, _, _ = self.run_batched_inference([current_prompt], enable_kv_cache=True)
            scores = scores[0]  # remove batch dimension
            #TODO: check if removal of batch dimension is correct

            # Select next token
            if isinstance(scores, torch.Tensor):
                next_idx = torch.argmax(scores).item()
                next_token = tokens_list[next_idx]
            else:
                next_token = tokens_list[0]  # fallback

            draft_tokens.append(next_token)
            current_prompt.append(next_token)  # extend prompt for next step

        return draft_tokens

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
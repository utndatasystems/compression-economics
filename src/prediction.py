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
from pathlib import Path
import tarfile
import torch
from pyroaring import BitMap
import time
from peft import PeftModel

from src.models import NGramPredictor
from src.predictors import load_ngram_predictor, save_ngram_predictor, train_ngram_predictor

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

        encode_kwargs = {"add_special_tokens": False, "truncation": False}
        # Tokenize only the requested prefix. Encoding the entire input and
        # slicing afterwards is needlessly expensive for large corpora and can
        # emit a model-context warning even though the model never sees that
        # full sequence.
        if args.first_n_tokens is not None:
            encode_kwargs.update(
                truncation=True,
                max_length=args.first_n_tokens,
            )
        self.data_tokens = self.tokenizer.encode(self.data, **encode_kwargs)
        if args.first_n_tokens is not None:
            # Keep the limit authoritative for tokenizers that do not implement
            # the optional truncation keyword (including lightweight test ones).
            self.data_tokens = self.data_tokens[:args.first_n_tokens]
            if len(self.data_tokens) < args.first_n_tokens:
                self.args.first_n_tokens = len(self.data_tokens)
                print(
                    "Reducing first_n_tokens to "
                    f"{self.args.first_n_tokens}, since the input data "
                    "has fewer tokens."
                )
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
            # Preserve CR, LF, and CRLF exactly. Universal-newline
            # translation can otherwise change a lossless token sequence.
            with open(input_path, "r", encoding="utf-8", newline="") as f:
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
    

class NGramTokenPredictor:
    """Token bigram adapter implementing the legacy predictor interface."""

    def __init__(self, args, bitmap_data):
        if args.encoding not in {"AC", "ANS"}:
            raise ValueError("the ngram engine supports AC or ANS coding")
        if bitmap_data is None:
            raise ValueError("the ngram engine requires a global token bitmap")
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=".cache")
        self.tokens_list = list(BitMap.deserialize(bitmap_data))
        self._dense_ids = {token: index for index, token in enumerate(self.tokens_list)}
        self.base_params = self.adapter_params = 0
        self.base_size_mb = self.adapter_size_mb = 0.0
        checkpoint = Path(args.ngram_model_path)
        if args.mode == "compress":
            training_path = getattr(args, "ngram_training_path", None)
            if not training_path:
                raise ValueError("--ngram-training-path is required when compressing with --engine ngram")
            with open(training_path, "r", encoding="utf-8", newline="") as handle:
                training_text = handle.read()
            source_ids = self.tokenizer.encode(training_text, add_special_tokens=False)
            training_ids = [self._dense_ids[token] for token in source_ids if token in self._dense_ids]
            if len(training_ids) < 2:
                raise ValueError("n-gram training data has fewer than two tokens in the stream alphabet")
            self.model = NGramPredictor(len(self.tokens_list), order=args.ngram_order)
            train_ngram_predictor(self.model, training_ids)
            args.ngram_model_sha256 = save_ngram_predictor(
                checkpoint, self.model, tokenizer_name=args.model_name, token_ids=self.tokens_list
            )
        else:
            self.model = load_ngram_predictor(
                checkpoint, tokenizer_name=args.model_name, token_ids=self.tokens_list,
                expected_sha256=getattr(args, "ngram_model_sha256", None),
            )

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """Score prompt final tokens in global-bitmap order; no KV cache is used."""
        started = time.perf_counter()
        try:
            contexts = [[self._dense_ids[token] for token in prompt] for prompt in prompts]
        except KeyError as error:
            raise ValueError("prompt contains a token outside the stream bitmap") from error
        self.last_model_input_tokens = sum(
            min(len(context), self.model.context_length) for context in contexts
        )
        logits = self.model.logits(contexts)
        softmax_started = time.perf_counter()
        probabilities = torch.softmax(logits, dim=-1)
        return self.tokens_list, probabilities, time.perf_counter() - started, time.perf_counter() - softmax_started

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_by_id(self, token_id):
        return self.tokens_list[token_id]

    def reset_kv_cache(self):
        return None


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
        self.reset_kv_cache()

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
        if self.args.encoding in {"AC", "ANS", "PMATIC"}:
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
        Run a full forward pass without reusing KV cache state.
        Uneven prompts are scored independently without padding.

        This is intended as a correctness baseline for comparing against the
        cached path in run_batched_inference().

        input: 
        - prompts: list of tokenized prompts (list of list of ints)

        output:
        - tokens_list: list of token IDs corresponding to the score columns in the returned tensor.
        - scores: tensor of shape (batch_size, vocab_size_or_reduced_vocab_size)
        - data_copy_time: approximate time spent moving tensors between host and device during this call.
        - softmax_time: time spent computing the softmax (only non-zero when encoding in {"AC", "ANS"} / "ANS" / "PMATIC").
        """
        # A cache-free call ends the previous cached sequence.
        self.reset_kv_cache()
        if self.args.engine != "transformer":
            raise ValueError(f"Unsupported engine: {self.args.engine}")
        if not prompts or any(not row for row in prompts):
            raise ValueError("prompts must contain non-empty token sequences")

        data_copy_time = 0.0
        self.last_model_input_tokens = sum(map(len, prompts))
        with torch.inference_mode():
            if self._check_rectangular(prompts):
                started = time.perf_counter()
                input_ids = torch.tensor(prompts, dtype=torch.long, device=self.device)
                data_copy_time += time.perf_counter() - started
                outputs = self.model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits[:, -1, :]
            else:
                # Match the cached path: each uneven row has its own unpadded
                # positions and its own final-token logits.
                rows = []
                for prompt in prompts:
                    started = time.perf_counter()
                    input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
                    data_copy_time += time.perf_counter() - started
                    outputs = self.model(input_ids=input_ids, use_cache=False)
                    rows.append(outputs.logits[0, -1, :])
                logits = torch.stack(rows)

        assert logits.dim() == 2, f"Expected logits of shape (batch_size, vocab_size), got {logits.shape}"
        assert isinstance(logits, torch.Tensor), f"Expected logits to be a torch.Tensor, got {type(logits)}"

        return self._finalize_batched_scores(logits, data_copy_time)
    

    def run_batched_inference(self, prompts, enable_kv_cache=True):
        """Score prompts, reusing only caches built from their exact prefixes.

        Rectangular batches share a single cache. Uneven batches use independent
        per-row caches so padding never affects logits or cached positions.
        Cache-free calls discard all previous cache state.
        """
        if not enable_kv_cache:
            return self.run_batched_inference_cachefree(prompts)
        if self.args.engine != "transformer":
            raise ValueError(f"Unsupported engine: {self.args.engine}")
        if not prompts or any(not row for row in prompts):
            raise ValueError("prompts must contain non-empty token sequences")

        current = [tuple(row) for row in prompts]
        previous = getattr(self, "_cached_prompts", [])
        rectangular = len({len(row) for row in current}) == 1
        data_copy_time = 0.0
        model_input_tokens = 0

        def extends(row, prefix):
            return len(row) > len(prefix) and row[:len(prefix)] == prefix

        try:
            with torch.inference_mode():
                if rectangular:
                    reuse = (
                        getattr(self, "_past_kv", None) is not None
                        and len(previous) == len(current)
                        and all(extends(row, prefix) for row, prefix in zip(current, previous))
                    )
                    inputs = (
                        [row[len(prefix):] for row, prefix in zip(current, previous)]
                        if reuse else current
                    )
                    started = time.perf_counter()
                    input_ids = torch.tensor(inputs, dtype=torch.long, device=self.device)
                    data_copy_time += time.perf_counter() - started
                    model_input_tokens = int(input_ids.numel())
                    kwargs = {"past_key_values": self._past_kv} if reuse else {}
                    outputs = self.model(input_ids=input_ids, use_cache=True, **kwargs)
                    logits = outputs.logits[:, -1, :]
                    self._past_kv = outputs.past_key_values
                    self._row_past_kv = None
                else:
                    # Per-row inference avoids model-specific padding behavior.
                    old_rows = getattr(self, "_row_past_kv", None)
                    row_caches = []
                    row_logits = []
                    for index, row in enumerate(current):
                        reuse = (
                            old_rows is not None
                            and index < len(old_rows)
                            and old_rows[index] is not None
                            and index < len(previous)
                            and extends(row, previous[index])
                        )
                        input_row = row[len(previous[index]):] if reuse else row
                        started = time.perf_counter()
                        input_ids = torch.tensor([input_row], dtype=torch.long, device=self.device)
                        data_copy_time += time.perf_counter() - started
                        model_input_tokens += int(input_ids.numel())
                        kwargs = {"past_key_values": old_rows[index]} if reuse else {}
                        outputs = self.model(input_ids=input_ids, use_cache=True, **kwargs)
                        row_caches.append(outputs.past_key_values)
                        row_logits.append(outputs.logits[0, -1, :])
                    logits = torch.stack(row_logits)
                    self._row_past_kv = row_caches
                    self._past_kv = None

            self._cached_prompts = current
            self._cached_context_len = max(map(len, current))
            self.last_model_input_tokens = model_input_tokens
        except Exception:
            self.reset_kv_cache()
            raise

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
        """Discard batched and per-row caches with their source prefixes."""
        self._past_kv = None
        self._row_past_kv = None
        self._cached_prompts = []
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
    

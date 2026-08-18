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
        if not self.reduce_tokens:
            raise ValueError("Bitmap only available when reduce_tokens is True.")

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
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=".cache")
        self.model.eval()  # Set model to evaluation mode

        # --- If bitmap_data is provided, reconstruct tokens_list & index_tensor ---
        if bitmap_data is not None:
            bitmap = BitMap.deserialize(bitmap_data)
            self.tokens_list = list(bitmap)
        else:
            self.tokens_list = list(range(self.tokenizer.vocab_size))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.index_tensor = torch.tensor(self.tokens_list, dtype=torch.long, device=device)
        self.reduce_tokens = args.reduce_tokens

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')
            print("Model moved to GPU.")
        else:
            print("GPU not available, using CPU.")

        # pin the 
        self.probs_cpu = torch.empty(
            (args.batch_size, len(self.tokens_list)),
            dtype=torch.float32,
            pin_memory=True
        )

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
            self.tokens_list, dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu'
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
        # TODO: Add documentation for this method

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not hasattr(self, "_past_kv"):
            self._past_kv = None
            self._cached_context_len = 0

        with torch.inference_mode():
            if not enable_kv_cache:
                # If not using cache, run the model on the full prompt every time.
                input_ids = torch.tensor(prompts, device=device)
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
                    input_ids = torch.tensor(prompts, device=device)
                    outputs = self.model(input_ids, use_cache=True)
                    self._past_kv = outputs.past_key_values
                    self._cached_context_len = 0
                else:
                    # Incremental step: process only the last token using the existing cache.
                    delta = [row[-1:] for row in prompts]
                    delta = torch.tensor(delta, device=device, dtype=torch.long)

                    outputs = self.model(delta, past_key_values=self._past_kv, use_cache=True)
                    self._past_kv = outputs.past_key_values
                    self._cached_context_len += 1

            logits = outputs.logits[:, -1, :]
            if getattr(self, "reduce_tokens", False):
                logits = logits.index_select(1, self.index_tensor.to(logits.device))
            probs = torch.softmax(logits, dim=-1)

            self.probs_cpu.copy_(probs, non_blocking=True)
            torch.cuda.synchronize()

        return self.tokens_list, self.probs_cpu

    def get_token_info(self, prompt_tokens):
        """
        Given a list of prompt tokens, return the list of candidate token IDs and their probabilities.

        Args:
            prompt_tokens (list[int]): Tokenized prompt input.

        Returns:
            tuple: (List of candidate token IDs, list of corresponding probabilities)
        """
        input_ids = torch.tensor([prompt_tokens])
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        with torch.inference_mode():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token

            logits = logits[0]  # Shape: (vocab_size,)

            if self.reduce_tokens:
                # Select only logits for reduced token set
                selected_logits = torch.index_select(logits, dim=0, index=self.index_tensor)
            else:
                selected_logits = logits

            # Convert logits to probabilities using softmax
            probs = torch.nn.functional.softmax(selected_logits, dim=0)

        return self.tokens_list, probs.cpu()

    def get_token_info_cache(self, prompt_tokens, use_kv_cache=True):
        """
        Returns candidate token IDs and their probabilities, using KV caching for efficiency.

        This function handles the model's key-value cache to speed up inference.
        The calling function is responsible for managing the context window (slicing).
        This function detects if the prompt has been shortened, which implies a cache reset.

        Args:
            prompt_tokens (list[int]): The list of token IDs for the prompt.
            use_kv_cache (bool): If True, enables the KV cache mechanism.

        Returns:
            tuple: A tuple containing:
                - list[int]: The list of candidate token IDs.
                - list[float]: The corresponding probabilities for each candidate token.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize cache state if it doesn't exist.
        if not hasattr(self, "_past_kv"):
            self._past_kv = None
            self._cached_context_len = 0

        if not use_kv_cache:
            # If not using cache, run the model on the full prompt every time.
            input_ids = torch.tensor([prompt_tokens], device=device)
            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=False)
            # Ensure cache is cleared when not in use.
            self._past_kv = None
            self._cached_context_len = 0
        else:
            # Check if the cache needs to be reset. This happens if the external
            # context management has shortened the prompt.
            reset_cache = len(prompt_tokens) < self._cached_context_len

            if self._past_kv is None or reset_cache:
                # Rebuild the cache from the full prompt.
                input_ids = torch.tensor([prompt_tokens], device=device)
                with torch.inference_mode():
                    outputs = self.model(input_ids, use_cache=True)
                    self._past_kv = outputs.past_key_values
                    self._cached_context_len = len(prompt_tokens)
            else:
                # Incremental step: process only the last token using the existing cache.
                new_token_id = prompt_tokens[-1]
                input_ids = torch.tensor([[new_token_id]], device=device)
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids,
                        past_key_values=self._past_kv,
                        use_cache=True
                    )
                    self._past_kv = outputs.past_key_values
                    self._cached_context_len += 1

        with torch.inference_mode():
            # Extract logits for the next token prediction.
            logits = outputs.logits[:, -1, :]
            logits = logits[0]

            # Filter logits based on the allowed token mask if token reduction is enabled.
            if self.reduce_tokens:
                selected_logits = torch.index_select(logits, dim=0, index=self.index_tensor)
            else:
                selected_logits = logits

            # Convert logits to probabilities.
            probs = torch.nn.functional.softmax(selected_logits, dim=0)
        return self.tokens_list, probs

    def get_full_token_info(self, prompt_tokens):
        """
        Given a list of prompt tokens, return the list of all token IDs and their probabilities
        (full vocabulary), regardless of the reduce_tokens setting.

        Args:
            prompt_tokens (list[int]): Tokenized prompt input.

        Returns:
            tuple: (List of all token IDs, list of corresponding probabilities)
        """
        input_ids = torch.tensor([prompt_tokens])
        if torch.cuda.is_available():
            input_ids = input_ids.to('cuda')

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for the last token

        logits = logits[0]  # Shape: (vocab_size,)

        # Convert logits to probabilities using softmax for the full vocabulary
        probs = torch.nn.functional.softmax(logits, dim=0).tolist()

        return list(range(self.tokenizer.vocab_size)), probs

    def get_ids_probs(self, prompt_tokens, next_token):
        """
        Get the index and probability of the next token from the list of predicted tokens.

        Args:
            prompt_tokens (list[int]): Tokenized input prompt.
            next_token (int): Token ID to look for in the prediction.

        Returns:
            tuple: (Index of the next_token in the token list, probability score)
        """
        token_ids, probs = self.get_token_info(prompt_tokens)
        next_index = token_ids.index(next_token)
        return next_index, probs

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

    def _get_bitmap(self, compression='none', ranking=None):
        """
        Get the bitmap and size with optional compression.

        Args:
            compression (str): Type of compression ('none', 'rle', 'sparse').

        Returns:
            tuple: (bitmap_representation, size_in_bytes)
        """
        if not self.reduce_tokens:
            raise ValueError("Bitmap only available when reduce_tokens is True.")

        if compression == 'none':
            size_bytes = (len(self.token_bitmap) + 7) // 8
            return self.token_bitmap.clone(), size_bytes

        elif compression == 'rle':
            rle = []
            current_bit = self.token_bitmap[0].item()
            count = 1
            count_max = 0
            for bit in self.token_bitmap[1:]:
                bit = bit.item()
                if bit == current_bit:
                    count += 1
                else:
                    if count > count_max:
                        count_max = count
                    rle.append((current_bit, count))
                    current_bit = bit
                    count = 1
            if count > count_max:
                count_max = count
            rle.append((current_bit, count))
            
            size_bits = len(rle) * (1 + math.log2(count_max))  # 1 bit for value, log2 for count
            size_bytes = (size_bits + 7) // 8
            return rle, size_bytes

        elif compression == 'sparse':
            indices = self.tokens_list
            size_bytes = len(indices) * 4  # 4 bytes per int index
            return indices, size_bytes
        
        elif compression == 'roaring':
            bitmap = BitMap(self.tokens_list)
            binary_data = bitmap.serialize()
            # The result of serialize() is a bytes object, so its length is the size in bytes.
            size_bytes = len(binary_data)
            # Verification step
            deserialized_bitmap = BitMap.deserialize(binary_data)
            assert self.tokens_list == list(deserialized_bitmap)
            return binary_data, size_bytes
        
        elif compression == 'roaring_ranking':
            assert ranking is not None
            bitmap = BitMap(ranking)
            binary_data = bitmap.serialize()
            # The result of serialize() is a bytes object, so its length is the size in bytes.
            size_bytes = len(binary_data)
            # Verification step
            deserialized_bitmap = BitMap.deserialize(binary_data)
            assert ranking == list(deserialized_bitmap)
            return binary_data, size_bytes

        else:
            raise ValueError(f"Unsupported compression type: {compression}")

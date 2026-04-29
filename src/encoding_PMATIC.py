from string import printable
import numpy as np
import random
import math
from collections import Counter
import heapq
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import zstandard as zstd

@dataclass
class HuffmanNode:
    symbol: Optional[int]
    left: Optional["HuffmanNode"]
    right: Optional["HuffmanNode"]

def build_huffman_code(rank_list: List[int]) -> Dict[int, str]:
    """
    Given a rank_list, e.g., [3,5,3,1,5,...]
    Returns a dict: { symbol -> bitstring code }
    """
    if not rank_list:
        return {}

    freq = Counter(rank_list)

    # Special case: if there is only one symbol, assign it the code "0"
    if len(freq) == 1:
        only_symbol = next(iter(freq))
        return {only_symbol: "0"}

    # Min-heap: (freq, counter, node)
    heap: List[Any] = []
    counter = 0
    for sym, f in freq.items():
        node = HuffmanNode(symbol=sym, left=None, right=None)
        heap.append((f, counter, node))
        counter += 1

    heapq.heapify(heap)

    # Repeatedly merge the two nodes with the smallest frequencies
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        parent = HuffmanNode(symbol=None, left=n1, right=n2)
        heapq.heappush(heap, (f1 + f2, counter, parent))
        counter += 1

    # The last remaining node is the root
    [(_, _, root)] = heap

    # DFS generate codebook
    codebook: Dict[int, str] = {}

    def dfs(node: HuffmanNode, prefix: str):
        if node.symbol is not None:
            # leaf node
            codebook[node.symbol] = prefix or "0"
            return
        if node.left is not None:
            dfs(node.left, prefix + "0")
        if node.right is not None:
            dfs(node.right, prefix + "1")

    dfs(root, "")
    return codebook

def huffman_encode(rank_list: List[int], codebook: Dict[int, str]) -> str:
    """
    Use the codebook to convert the rank_list into a continuous bit string ('0100111...')
    """
    # This is O(n) and much faster than concatenating strings one by one
    bit_chunks = [codebook[sym] for sym in rank_list]
    bit_string = "".join(bit_chunks)
    return bit_string

def build_tree_from_codebook(codebook: Dict[int, str]) -> HuffmanNode:
    """
    Given a codebook {symbol: bitstring}, reconstruct the Huffman tree.
    """
    root = HuffmanNode(symbol=None, left=None, right=None)
    for sym, code in codebook.items():
        node = root
        for bit in code:
            if bit == "0":
                if node.left is None:
                    node.left = HuffmanNode(symbol=None, left=None, right=None)
                node = node.left
            else:  # bit == "1"
                if node.right is None:
                    node.right = HuffmanNode(symbol=None, left=None, right=None)
                node = node.right
        node.symbol = sym
    return root


def huffman_decode(bit_string: str, codebook: Dict[int, str]) -> List[int]:
    """
    Use bit_string + codebook to decode and return rank_list
    """
    root = build_tree_from_codebook(codebook)
    result: List[int] = []
    node = root

    for bit in bit_string:
        if bit == "0":
            node = node.left
        else:
            node = node.right

        # When reaching a leaf node, output a symbol
        if node.symbol is not None:
            result.append(node.symbol)
            node = root

    return result

class ArithmeticCoderBase(object):
    # Constructs an arithmetic coder, which initializes the code range.
    def __init__(self, statesize):
        #if statesize < 1:
            #raise ValueError("State size out of range")
        # -- Configuration fields --
        # Number of bits for the 'low' and 'high' state variables. Must be at least 1.
        # - Larger values are generally better - they allow a larger maximum frequency total (MAX_TOTAL),
        #   and they reduce the approximation error inherent in adapting fractions to integers;
        #   both effects reduce the data encoding loss and asymptotically approach the efficiency
        #   of arithmetic coding using exact fractions.
        # - But larger state sizes increase the computation time for integer arithmetic,
        #   and compression gains beyond ~30 bits essentially zero in real-world applications.
        # - Python has native bigint arithmetic, so there is no upper limit to the state size.
        #   For Java and C++ where using native machine-sized integers makes the most sense,
        #   they have a recommended value of STATE_SIZE=32 as the most versatile setting.
        self.STATE_SIZE = statesize
        # Maximum range (high+1-low) during coding (trivial), which is 2^STATE_SIZE = 1000...000.
        self.MAX_RANGE = 1 << self.STATE_SIZE
        # Minimum range (high+1-low) during coding (non-trivial), which is 0010...010.
        self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
        # Maximum allowed total from a frequency table at all times during coding. This differs from Java
        # and C++ because Python's native bigint avoids constraining the size of intermediate computations.
        self.MAX_TOTAL = self.MIN_RANGE
        # Bit mask of STATE_SIZE ones, which is 0111...111.
        self.MASK = self.MAX_RANGE - 1
        # The top bit at width STATE_SIZE, which is 0100...000.
        self.TOP_MASK = self.MAX_RANGE >> 1
        # The second highest bit at width STATE_SIZE, which is 0010...000. This is zero when STATE_SIZE=1.
        self.SECOND_MASK = self.TOP_MASK >> 1

        # -- State fields --
        # Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
        self.low = 0
        # High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.
        self.high = self.MASK
#         print("STATE_SIZE  : ",self.STATE_SIZE)
#         print("MAX_RANGE   : ",bin(self.MAX_RANGE))
#         print("MIN_RANGE   : ",bin(self.MIN_RANGE))
#         print("MAX_TOTAL   : ",bin(self.MAX_TOTAL))
#         print("MASK        : ",bin(self.MASK))
#         print("TOP_MASK    : ",bin(self.TOP_MASK))
#         print("SECOND_MASK : ",bin(self.SECOND_MASK))
#         print("low         : ",bin(self.low))
#         print("high        : ",bin(self.high))


    # Updates the code range (low and high) of this arithmetic coder as a result
    # of processing the given symbol with the given frequency table.
    # Invariants that are true before and after encoding/decoding each symbol:
    # - 0 <= low <= code <= high < 2^STATE_SIZE. ('code' exists only in the decoder.)
    #   Therefore these variables are unsigned integers of STATE_SIZE bits.
    # - (low < 1/2 * 2^STATE_SIZE) && (high >= 1/2 * 2^STATE_SIZE).
    #   In other words, they are in different halves of the full range.
    # - (low < 1/4 * 2^STATE_SIZE) || (high >= 3/4 * 2^STATE_SIZE).
    #   In other words, they are not both in the middle two quarters.
    # - Let range = high - low + 1, then MAX_RANGE/4 < MIN_RANGE <= range
    #   <= MAX_RANGE = 2^STATE_SIZE. These invariants for 'range' essentially
    #   dictate the maximum total that the incoming frequency table can have.
    def update(self,  cumul, symbol):
        # State check
        low = self.low
        high = self.high
        #if low >= high or (low & self.MASK) != low or (high & self.MASK) != high:
            #raise AssertionError("Low or high out of range")
        range = high - low + 1
        #if not (self.MIN_RANGE <= range <= self.MAX_RANGE):
            #raise AssertionError("Range out of range")

        # Frequency table values check
        total = cumul[-1].item()
        symlow = cumul[symbol].item()
        symhigh = cumul[symbol+1].item()
        #if symlow == symhigh:
            #raise ValueError("Symbol has zero frequency")
        #if total > self.MAX_TOTAL:
            #raise ValueError("Cannot code symbol because total is too large")

        # Update range
        newlow  = low + symlow  * range // total
        newhigh = low + symhigh * range // total - 1
        self.low = newlow
        self.high = newhigh
        # While the highest bits are equal
#         print("New loop")
#         print(bin(self.low),"; ",bin(self.high))
#         print((self.low ^ self.high) & self.TOP_MASK)
        while ((self.low ^ self.high) & self.TOP_MASK) == 0:
            self.shift()
#             print("After shift:",bin(self.low),"; ",bin(self.high))
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) & self.MASK) | 1
#             print(bin(self.low),"; ",bin(self.high))

        # While the second highest bit of low is 1 and the second highest bit of high is 0
#         print(self.low & ~self.high & self.SECOND_MASK)
            
        while (self.low & ~self.high & self.SECOND_MASK) != 0:
            self.underflow()
#             print("After underflow",bin(self.low),"; ",bin(self.high))
            self.low = (self.low << 1) & (self.MASK >> 1)
            self.high = ((self.high << 1) & (self.MASK >> 1)) | self.TOP_MASK | 1
#             print(bin(self.low),"; ",bin(self.high))


    # Called to handle the situation when the top bit of 'low' and 'high' are equal.
    def shift(self):
        raise NotImplementedError()


    # Called to handle the situation when low=01(...) and high=10(...).
    def underflow(self):
        raise NotImplementedError()



# Encodes symbols and writes to an arithmetic-coded bit stream.
class ArithmeticEncoder(ArithmeticCoderBase):

    # Constructs an arithmetic coding encoder based on the given bit output stream.
    def __init__(self, statesize, bitout):
        super(ArithmeticEncoder, self).__init__(statesize)
        # The underlying bit output stream.
        self.output = bitout
        # Number of saved underflow bits. This value can grow without bound.
        self.num_underflow = 0


    # Encodes the given symbol based on the given frequency table.
    # This updates this arithmetic coder's state and may write out some bits.
    def write(self, cumul, symbol):
    #		if not isinstance(freqs, CheckedFrequencyTable):
    #			freqs = CheckedFrequencyTable(freqs)
        self.update(cumul, symbol)


    # Terminates the arithmetic coding by flushing any buffered bits, so that the output can be decoded properly.
    # It is important that this method must be called at the end of the each encoding process.
    # Note that this method merely writes data to the underlying output stream but does not close it.
    def finish(self):
        self.output.write(1)


    def shift(self):
        bit = self.low >> (self.STATE_SIZE - 1)
        self.output.write(bit)

        # Write out the saved underflow bits
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
        self.num_underflow = 0


    def underflow(self):
        self.num_underflow += 1



# Reads from an arithmetic-coded bit stream and decodes symbols.
class ArithmeticDecoder(ArithmeticCoderBase):

    # Constructs an arithmetic coding decoder based on the
    # given bit input stream, and fills the code bits.
    def __init__(self, statesize, bitin):
        super(ArithmeticDecoder, self).__init__(statesize)
        # The underlying bit input stream.
        self.input = bitin
        # The current raw code bits being buffered, which is always in the range [low, high].
        self.code = 0
        for _ in range(self.STATE_SIZE):
            self.code = self.code << 1 | self.read_code_bit()


    def read(self, cumul, alphabet_size):
        """Decodes the next symbol based on the given frequency table and returns it.
            Also updates this arithmetic coder's state and may read in some bits."""
    #		if not isinstance(freqs, CheckedFrequencyTable):
    #			freqs = CheckedFrequencyTable(freqs)

        # Translate from coding range scale to frequency table scale
        total = cumul[-1].item()
    #		if total > self.MAX_TOTAL:
    #			raise ValueError("Cannot decode symbol because total is too large")
        range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range
    #		assert value * range // total <= offset
    #		assert 0 <= value < total

        # A kind of binary search. Find highest symbol such that freqs.get_low(symbol) <= value.
        start = 0
        end = alphabet_size
        while end - start > 1:
            middle = (start + end) >> 1
            if cumul[middle] > value:
                end = middle
            else:
                start = middle
    #		assert start + 1 == end

        symbol = start
    #		assert freqs.get_low(symbol) * range // total <= offset < freqs.get_high(symbol) * range // total
        self.update(cumul, symbol)
    #		if not (self.low <= self.code <= self.high):
    #			raise AssertionError("Code out of range")
        return symbol


    def shift(self):
        self.code = ((self.code << 1) & self.MASK) | self.read_code_bit()


    def underflow(self):
        self.code = (self.code & self.TOP_MASK) | ((self.code << 1) & (self.MASK >> 1)) | self.read_code_bit()


    # Returns the next bit (0 or 1) from the input stream. The end
    # of stream is treated as an infinite number of trailing zeros.
    def read_code_bit(self):
        temp = self.input.read()
        if temp == -1:
            temp = 0
        return temp


# ------------------------------------------------------------------
# Minimal bit-stream helpers
# ------------------------------------------------------------------
class BitOutputStream:
    """Collects single bits written by the encoder."""
    def __init__(self):
        self.bits = []

    def write(self, bit: int):
        self.bits.append(bit & 1)  # store only the LSB

    def get_bits(self):
        return self.bits


class BitInputStream:
    """Feeds bits (and infinite trailing zeros) to the decoder."""
    def __init__(self, bits):
        self.bits = bits
        self.index = 0

    def read(self):
        if self.index < len(self.bits):
            b = self.bits[self.index]
            self.index += 1
            return b
        else:                     # spec: return –1 → decoder treats as 0
            return -1


# ------------------------------------------------------------------
# Utility – convert probabilities → integer cumulative vector
# ------------------------------------------------------------------


def build_cumul(prob_vec: np.ndarray, total: int = 262144) -> np.ndarray:
    """
    Turn a probability vector into a cumulative-frequency array for arithmetic coding.
    Ensures every symbol ≥1 and freq.sum() == total.
    """
    alphabet_size = prob_vec.size

    # Step 1: allocate counts proportional to prob_vec
    freq = (prob_vec * (total - alphabet_size)).astype(np.int64)

    # Step 2: ensure every symbol ≥1
    freq += 1

    # Step 3: compute diff and adjust
    diff = total - freq.sum()
    if diff != 0:
        # adjust the symbol with largest probability first
        # np.argsort(-prob_vec) gives descending probabilities
        indices = np.argsort(-prob_vec)
        idx = 0
        while diff != 0:
            i = indices[idx % alphabet_size]  # wrap around if needed
            if diff > 0:
                freq[i] += 1
                diff -= 1
            else:  # diff < 0
                if freq[i] > 1:
                    freq[i] -= 1
                    diff += 1
            idx += 1

    # Safety check
    assert freq.sum() == total, f"freq.sum={freq.sum()} != total={total}"
    assert np.all(freq >= 1)

    # Build cumulative array
    cumul = np.empty(alphabet_size + 1, dtype=np.int64)
    cumul[0] = 0
    cumul[1:] = np.cumsum(freq)
    return cumul

class LLMCompressor:
    def __init__(self):
        self.bitout = BitOutputStream()
        self.encoder = ArithmeticEncoder(32, self.bitout)
        self.cross_entropy_sum = 0.0
        self.token_count = 0

    def next_token(self, correct_token_idx, probs):
        # Compute cross-entropy loss for this token
        prob = probs[correct_token_idx]
        if prob > 0:
            self.cross_entropy_sum += -math.log2(prob)
        else:
            self.cross_entropy_sum += float('inf')  # handle numerical underflow

        self.token_count += 1
        self.encoder.write(build_cumul(probs), correct_token_idx)

    def compress(self, encoding="AC", rank_list=None):
        if encoding == "AC":
            self.encoder.finish()
            return self.bitout.get_bits()
        elif encoding == "bitpacked":
            assert rank_list is not None, "rank_list must be provided for bitpacked encoding"
            # bitpack encoding depend on max rank
            max_rank = max(rank_list) if rank_list else 0
            num_bits = max_rank.bit_length()

            # print(f"max_rank = {max_rank}, num_bits = {num_bits}")

            bit_chunks = [format(rank, f'0{num_bits}b') for rank in rank_list]
            bit_string = "".join(bit_chunks)
            # print(f"length of bit string: {len(bit_string)}")
            return bit_string
        elif encoding == "huffman":
            assert rank_list is not None, "rank_list must be provided for huffman encoding"
            codebook = build_huffman_code(rank_list)
            bit_string = huffman_encode(rank_list, codebook)
            print("encoded bit length:", len(bit_string))
            return bit_string, codebook
        elif encoding == "zstd":
            assert rank_list is not None, "rank_list must be provided for zstd encoding"
            ranks_arr = np.asarray(rank_list, dtype=np.uint32)
            raw_bytes = ranks_arr.tobytes()
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(raw_bytes)
            bit_string = "".join(f"{b:08b}" for b in compressed)
            return bit_string
        else:
            raise NotImplementedError(f"Encoding method '{encoding}' is not implemented.")

    def get_cross_entropy(self):
        return self.cross_entropy_sum


class LLMDecompressor:
    def __init__(self, code):
        self.decoder = ArithmeticDecoder(32, BitInputStream(code))

    def decompress(self, probs: np.ndarray) -> int: # Returns one single token at a time (returns the index of the token)
        cumul = build_cumul(probs)
        return self.decoder.read(cumul, len(probs))
    

# ------------------------------------------------------------------
# PMATIC utilities
# ------------------------------------------------------------------

def binary_cumul(p1: float, total: int = 262144) -> np.ndarray:
    """
    Cumulative table for Bernoulli bit:
      symbol 0 has prob 1-p1
      symbol 1 has prob p1
    """
    p1 = float(np.clip(p1, 1e-12, 1.0 - 1e-12))
    return build_cumul(np.array([1.0 - p1, p1], dtype=np.float64), total=total)


def token_to_bits(token_idx: int, bit_width: int) -> List[int]:
    return [(token_idx >> shift) & 1 for shift in range(bit_width - 1, -1, -1)]


def bits_to_token(bits: List[int]) -> int:
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x

def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def pad_probs_to_power_of_two(probs: np.ndarray, alphabet_size: int) -> np.ndarray:
    """
    Pads probs to a power-of-two PMATIC alphabet.

    Real symbols keep their original probabilities.
    Dummy symbols get probability 0.
    """
    probs = np.asarray(probs, dtype=np.float64)

    if probs.size != alphabet_size:
        raise ValueError(f"Expected probs of size {alphabet_size}, got {probs.size}")

    padded_size = next_power_of_two(alphabet_size)

    padded = np.zeros(padded_size, dtype=np.float64)
    padded[:alphabet_size] = probs

    s = padded.sum()
    if s <= 0:
        raise ValueError("Probability vector must have positive total mass")

    return padded / s


def conditional_bit_prob(
    probs: np.ndarray,
    prefix_bits: List[int],
    bit_width: int,
) -> float:
    """
    Computes P[next bit = 1 | prefix].

    Works safely with padded zero-probability dummy symbols.
    """
    probs = np.asarray(probs, dtype=np.float64)

    j = len(prefix_bits)
    total_mass = 0.0
    one_mass = 0.0

    for tok, p in enumerate(probs):
        if p == 0:
            continue

        bits = token_to_bits(tok, bit_width)

        if bits[:j] == prefix_bits:
            total_mass += p
            if bits[j] == 1:
                one_mass += p

    if total_mass <= 0:
        # This path should be unreachable if the arithmetic stream is valid.
        # Return neutral probability to avoid crashing before the true issue appears.
        return 0.5

    return one_mass / total_mass


def pmatic_quantize_encoder(p: float, delta: float, r: float):
    """
    Returns:
      helper_bit, quantized_probability

    Bins have width 2r.
    If p is safely inside a bin, helper=0 and use bin center.
    If p is near a bin boundary, helper=1 and use the nearest boundary.
    """
    if not (r > 2 * delta):
        raise ValueError("PMATIC requires r > 2 * delta")

    p = float(np.clip(p, 0.0, 1.0))
    width = 2.0 * r
    m = round(1.0 / width)

    if abs(m * width - 1.0) > 1e-9:
        raise ValueError("PMATIC expects r = 1/(2m) for integer m")

    # Find bin k in 0-based indexing.
    k = min(int(p / width), m - 1)
    left = k * width
    right = (k + 1) * width

    # PMATIC delta-interior.
    if k == 0:
        in_delta_interior = p <= right - delta
    elif k == m - 1:
        in_delta_interior = p >= left + delta
    else:
        in_delta_interior = (p >= left + delta) and (p <= right - delta)

    if in_delta_interior:
        center = left + r
        return 0, center

    # Near boundary: use nearest internal boundary.
    boundaries = [width * kk for kk in range(1, m)]
    boundary = min(boundaries, key=lambda b: abs(b - p))
    return 1, boundary


def pmatic_quantize_decoder(q: float, helper_bit: int, delta: float, r: float):
    """
    Decoder-side quantization.
    If helper=0, use center of q's bin.
    If helper=1, use nearest internal boundary.
    """
    if not (r > 2 * delta):
        raise ValueError("PMATIC requires r > 2 * delta")

    q = float(np.clip(q, 0.0, 1.0))
    width = 2.0 * r
    m = round(1.0 / width)

    if helper_bit == 0:
        k = min(int(q / width), m - 1)
        return k * width + r

    boundaries = [width * kk for kk in range(1, m)]
    return min(boundaries, key=lambda b: abs(b - q))


# ------------------------------------------------------------------
# PMATIC compressor / decompressor
# ------------------------------------------------------------------

class PMATICCompressor:
    def __init__(
        self,
        alphabet_size: int,
        delta: float = 1e-3,
        r: float = 0.05,
        statesize: int = 32,
        total: int = 262144,
    ):
        self.alphabet_size = alphabet_size
        self.padded_alphabet_size = next_power_of_two(alphabet_size)
        self.bit_width = math.ceil(math.log2(self.padded_alphabet_size))

        self.delta = delta
        self.r = r
        self.total = total

        self.bitout = BitOutputStream()
        self.encoder = ArithmeticEncoder(statesize, self.bitout)

        self.cross_entropy_sum = 0.0
        self.token_count = 0
        self.helper_ones = 0
        self.helper_count = 0

    def next_token(self, correct_token_idx: int, probs: np.ndarray):
        if not (0 <= correct_token_idx < self.alphabet_size):
            raise ValueError(
                f"Token {correct_token_idx} outside real vocabulary size "
                f"{self.alphabet_size}"
            )

        probs = pad_probs_to_power_of_two(probs, self.alphabet_size)

        p_tok = probs[correct_token_idx]
        self.cross_entropy_sum += -math.log2(max(p_tok, 1e-300))
        self.token_count += 1

        token_bits = token_to_bits(correct_token_idx, self.bit_width)
        prefix = []

        helper_p1 = self.delta / self.r

        for bit in token_bits:
            p_bit = conditional_bit_prob(probs, prefix, self.bit_width)

            helper_bit, qprob = pmatic_quantize_encoder(
                p_bit,
                delta=self.delta,
                r=self.r,
            )

            self.encoder.write(binary_cumul(helper_p1, self.total), helper_bit)
            self.encoder.write(binary_cumul(qprob, self.total), bit)

            self.helper_count += 1
            self.helper_ones += helper_bit

            prefix.append(bit)

    def compress(self):
        self.encoder.finish()
        return self.bitout.get_bits()

    def get_cross_entropy(self):
        return self.cross_entropy_sum

    def helper_one_fraction(self):
        return 0.0 if self.helper_count == 0 else self.helper_ones / self.helper_count


class PMATICDecompressor:
    def __init__(
        self,
        code,
        alphabet_size: int,
        delta: float = 1e-3,
        r: float = 0.05,
        statesize: int = 32,
        total: int = 262144,
    ):
        self.alphabet_size = alphabet_size
        self.padded_alphabet_size = next_power_of_two(alphabet_size)
        self.bit_width = math.ceil(math.log2(self.padded_alphabet_size))

        self.delta = delta
        self.r = r
        self.total = total

        self.decoder = ArithmeticDecoder(statesize, BitInputStream(code))

    def decompress(self, probs: np.ndarray) -> int:
        probs = pad_probs_to_power_of_two(probs, self.alphabet_size)

        prefix = []
        helper_p1 = self.delta / self.r

        for _ in range(self.bit_width):
            helper_bit = self.decoder.read(
                binary_cumul(helper_p1, self.total),
                2,
            )

            p_bit = conditional_bit_prob(probs, prefix, self.bit_width)

            qprob = pmatic_quantize_decoder(
                p_bit,
                helper_bit=helper_bit,
                delta=self.delta,
                r=self.r,
            )

            bit = self.decoder.read(
                binary_cumul(qprob, self.total),
                2,
            )

            prefix.append(bit)

        token_idx = bits_to_token(prefix)

        if token_idx >= self.alphabet_size:
            raise ValueError(
                f"Decoded dummy token {token_idx}. This means the induced conditional probability mismatch exceeded PMATIC tolerance, or the bitstream is corrupted."
            )

        return token_idx
    

def max_conditional_mismatch(p, q, alphabet_size):
    padded_p = pad_probs_to_power_of_two(p, alphabet_size)
    padded_q = pad_probs_to_power_of_two(q, alphabet_size)

    padded_size = next_power_of_two(alphabet_size)
    bit_width = math.ceil(math.log2(padded_size))

    max_diff = 0.0

    prefixes = [[]]
    for depth in range(bit_width):
        new_prefixes = []
        for prefix in prefixes:
            pp = conditional_bit_prob(padded_p, prefix, bit_width)
            qq = conditional_bit_prob(padded_q, prefix, bit_width)
            max_diff = max(max_diff, abs(pp - qq))

            new_prefixes.append(prefix + [0])
            new_prefixes.append(prefix + [1])

        prefixes = new_prefixes

    return max_diff


def make_safe_decoder_probs(p, alphabet_size, delta, initial_scale=1e-6):
    """
    Creates perturbed decoder probabilities whose induced conditional bit
    probabilities stay within delta.
    """
    scale = initial_scale

    for _ in range(30):
        noise = np.random.normal(scale=scale, size=alphabet_size)
        q = p + noise
        q = np.clip(q, 1e-12, None)
        q /= q.sum()

        if max_conditional_mismatch(p, q, alphabet_size) < delta:
            return q

        scale *= 0.5

    # Fallback: exact probabilities
    return p.copy()


def choose_pmatic_r(delta: float, safety: float = 1.01) -> float:
    """
    Choose r adaptively for PMATIC.

    Requirements:
      r > 2*delta
      r = 1/(2m), m integer

    Uses the smallest valid r up to the discrete 1/(2m) grid.
    """
    if not (0 < delta < 0.5):
        raise ValueError("delta must be in (0, 0.5)")

    target = 2.0 * delta * safety

    # Need 1/(2m) > target  =>  m < 1/(2*target)
    m_max = math.floor((1.0 / (2.0 * target)) - 1e-12)

    if m_max < 1:
        raise ValueError(
            f"delta={delta} is too large; cannot choose r=1/(2m) with r > 2*delta"
        )

    r = 1.0 / (2.0 * m_max)

    if not (r > 2.0 * delta):
        raise AssertionError(f"Internal error: r={r} does not satisfy r > 2*delta")

    return r

def PMATIC_test(decoder_case="safe-perturbed", delta=1e-3):

    decoder_case = decoder_case.lower()
    if decoder_case not in {"exact", "perturbed", "safe-perturbed"}:
        raise ValueError(
            "decoder_case must be one of: exact, perturbed, safe-perturbed"
        )
    np.random.seed(0)
    random.seed(0)

    # ----------------------------
    # Config
    # ----------------------------
    vocab_size = 50
    sequence_length = 200

    # PMATIC parameters (must satisfy r > 2*delta and r = 1/(2m))
    delta = delta
    r = choose_pmatic_r(delta) 
    #r = 0.05  # = 1/(2*10)

    # ----------------------------
    # Generate synthetic data
    # ----------------------------
    tokens = np.random.randint(0, vocab_size, size=sequence_length)

    def random_probs(vocab_size):
        x = np.random.rand(vocab_size)
        return x / x.sum()

    # Encoder sees "true" probabilities
    encoder_probs = [random_probs(vocab_size) for _ in range(sequence_length)]

    # Case 0 - Decoder sees slightly perturbed probabilities (simulate mismatch!)
    if decoder_case == "exact":
        decoder_probs = [p.copy() for p in encoder_probs]
    elif decoder_case == "perturbed":
        decoder_probs = []
        for p in encoder_probs:
            noise = np.random.normal(scale=1e-5, size=vocab_size)
            noisy = p + noise
            noisy = np.clip(noisy, 1e-9, None)
            noisy /= noisy.sum()
            decoder_probs.append(noisy)
    elif decoder_case == "safe-perturbed":
        decoder_probs = [make_safe_decoder_probs(p, vocab_size, delta) for p in encoder_probs]

    # Check PMATIC mismatch
    worst = max(
        max_conditional_mismatch(p, q, vocab_size)
        for p, q in zip(encoder_probs, decoder_probs))

    print("Worst conditional PMATIC mismatch:", worst)
    print("Delta:", delta)

    # ----------------------------
    # Compression
    # ----------------------------
    comp = PMATICCompressor(
        alphabet_size=vocab_size,
        delta=delta,
        r=r,
    )

    for token, probs in zip(tokens, encoder_probs):
        comp.next_token(token, probs)

    code = comp.compress()

    # ----------------------------
    # Decompression
    # ----------------------------
    dec = PMATICDecompressor(
        code,
        alphabet_size=vocab_size,
        delta=delta,
        r=r,
    )

    decoded = []
    for probs in decoder_probs:
        decoded.append(dec.decompress(probs))

    decoded = np.array(decoded)

    # ----------------------------
    # Evaluation
    # ----------------------------
    success = np.array_equal(tokens, decoded)

    print("\n=== PMATIC Test ===")
    print("Success:", success)

    if not success:
        mismatches = np.where(tokens != decoded)[0]
        print("First mismatch at index:", mismatches[0])
        print("Original:", tokens[mismatches[0]])
        print("Decoded :", decoded[mismatches[0]])

    # Bitrate
    total_bits = len(code)
    bpt = total_bits / sequence_length

    print("\n--- Compression stats ---")
    print("Total bits:", total_bits)
    print("Bits per token:", bpt)

    print("\n--- Model stats ---")
    print("Cross-entropy (encoder):", comp.get_cross_entropy() / sequence_length)

    print("\n--- PMATIC stats ---")
    print("Helper bit fraction (1s):", comp.helper_one_fraction())
    print("Helper bits per token:", comp.helper_count / sequence_length)

    # Optional sanity check: compare with ideal entropy
    entropy = 0.0
    for tok, probs in zip(tokens, encoder_probs):
        entropy += -math.log2(max(probs[tok], 1e-300))
    entropy /= sequence_length

    print("\n--- Reference ---")
    print("Empirical entropy:", entropy)


def experimient_setting_PMATIC_paper():
    # This is the experimental setting used in the PMATIC paper.

    # Setting 1 
    delta = 0.001
    r = 0.05 # should be approx 0.047
    # tokenizer alphabet size = 128256 tokens 
    # Each token requires a length-17 bitstring representation since ⌈log(128, 256)⌉ = 17.
    # context window = 512 with reset every 256 token via truncation

    # Setting 2 
    delta = 0.00001
    r = 0.005 



if __name__ == "__main__":
    PMATIC_test(decoder_case="safe-perturbed", delta=0.01)
    #PMATIC_test(decoder_case="exact", delta = 0.01)
    PMATIC_test(decoder_case="perturbed", delta=0.01)
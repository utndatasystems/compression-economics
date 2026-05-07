import numpy as np
import math
from collections import Counter
import heapq
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Sequence 
import zstandard as zstd

from src.encoding_utils import *


# =============================================================================
# Huffman coding for rank lists
# =============================================================================

@dataclass
class HuffmanNode:
    """A node in a binary Huffman tree."""

    symbol: Optional[int] = None
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.symbol is not None

def build_huffman_code(symbols: Sequence[int]) -> Dict[int, str]:
    """Build a Huffman codebook for a sequence of integer symbols.
    Input: 

    Given a rank_list, e.g., [3,5,3,1,5,...]

    Returns:
        A dictionary mapping each symbol to a bitstring {symbol -> bitstring code }, e.g. {7: "010"}.
    """

    if not symbols:
        return {}

    freq = Counter(symbols)

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

    def walk(node: HuffmanNode, prefix: str) -> None:
        if node.is_leaf:
            codebook[node.symbol] = prefix or "0"  # type: ignore[index]
            return
        if node.left is not None:
            walk(node.left, prefix + "0")
        if node.right is not None:
            walk(node.right, prefix + "1")

    walk(root, "")
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

# ------------------------------------------------------------------
# Minimal bit-stream helpers
# ------------------------------------------------------------------
class BitOutputStream:
    """Collects single bits, written by the encoder, in memory."""
    def __init__(self):
        self._bits: List[int] = []

    def write(self, bit: int):
        self._bits.append(bit & 1)  # store only the LSB

    def get_bits(self)-> List[int]:
        return self._bits.copy()


class BitInputStream:
    """Read bits from memory. After EOF, return -1 as required by the coder."""

    def __init__(self, bits: Sequence[int]) -> None:
        self._bits = bits
        self._index = 0

    def read(self) -> int:
        if self._index >= len(self._bits):
            return -1

        bit = self._bits[self._index]
        self._index += 1
        return bit
    

# =============================================================================
# Arithmetic coding core
# =============================================================================

class ArithmeticCoderBase(object):
    """
    Constructs an arithmetic coder, which initializes the code range.
    """

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
        self.MAX_RANGE = 1 << self.STATE_SIZE                       # Maximum range (high+1-low) during coding (trivial), which is 2^STATE_SIZE = 1000...000.
        self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2                  # Minimum range (high+1-low) during coding (non-trivial), which is 0010...010.
        self.MAX_TOTAL = self.MIN_RANGE                             # Maximum allowed total from a frequency table at all times during coding. This differs from Java and C++ because Python's native bigint avoids constraining the size of intermediate computations.
        self.MASK = self.MAX_RANGE - 1                              # Bit mask of STATE_SIZE ones, which is 0111...111.
        self.TOP_MASK = self.MAX_RANGE >> 1                         # The top bit at width STATE_SIZE, which is 0100...000.
        self.SECOND_MASK = self.TOP_MASK >> 1                       # The second highest bit at width STATE_SIZE, which is 0010...000. This is zero when STATE_SIZE=1.

        # -- State fields --
        self.low = 0                                                # Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
        self.high = self.MASK                                       # High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.

        print_on = False
        if print_on:
            print("STATE_SIZE  : ",self.STATE_SIZE)
            print("MAX_RANGE   : ",bin(self.MAX_RANGE))
            print("MIN_RANGE   : ",bin(self.MIN_RANGE))
            print("MAX_TOTAL   : ",bin(self.MAX_TOTAL))
            print("MASK        : ",bin(self.MASK))
            print("TOP_MASK    : ",bin(self.TOP_MASK))
            print("SECOND_MASK : ",bin(self.SECOND_MASK))
            print("low         : ",bin(self.low))
            print("high        : ",bin(self.high))


    def update(self, cumul, symbol):
        """
        Update the coding interval (low and high) after processing one symbol with the given frequency table.
        
        Invariants that are true before and after encoding/decoding each symbol:
        - 0 <= low <= code <= high < 2^STATE_SIZE. ('code' exists only in the decoder.)
        Therefore these variables are unsigned integers of STATE_SIZE bits.
        - (low < 1/2 * 2^STATE_SIZE) && (high >= 1/2 * 2^STATE_SIZE).
        In other words, they are in different halves of the full range.
        - (low < 1/4 * 2^STATE_SIZE) || (high >= 3/4 * 2^STATE_SIZE).
        In other words, they are not both in the middle two quarters.
        - Let range = high - low + 1, then MAX_RANGE/4 < MIN_RANGE <= range
        <= MAX_RANGE = 2^STATE_SIZE. These invariants for 'range' essentially
        dictate the maximum total that the incoming frequency table can have.
        """
        
        # State check
        low, high = self.low, self.high

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

    def shift(self):
        """Called to handle the situation when the top bit of 'low' and 'high' are equal."""
        raise NotImplementedError()

    def underflow(self):
        """Called to handle the situation when low=01(...) and high=10(...)."""
        raise NotImplementedError()



class ArithmeticEncoder(ArithmeticCoderBase):
    """Encodes symbols and writes to an arithmetic-coded bit stream."""
   
    def __init__(self, statesize, bitout):
        """ Constructs an arithmetic coding encoder based on the given bit output stream."""
        super(ArithmeticEncoder, self).__init__(statesize)
        # The underlying bit output stream.
        self.output = bitout
        # Number of saved underflow bits. This value can grow without bound.
        self.num_underflow = 0

    def write(self, cumul, symbol):
        """Encodes the given symbol based on the given frequency table. This updates this arithmetic coder's state and may write some bits."""
    #		if not isinstance(freqs, CheckedFrequencyTable):
    #			freqs = CheckedFrequencyTable(freqs)
        self.update(cumul, symbol)


    def finish(self):
        """
        Terminates the arithmetic coding by flushing any buffered bits, so that the output can be decoded properly.
        It is important that this method must be called at the end of the each encoding process.
        Note that this method merely writes data to the underlying output stream but does not close it. 
        """
        self.output.write(1)


    def shift(self):
        bit = self.low >> (self.STATE_SIZE - 1)
        self.output.write(bit)

        # Write the saved underflow bits
        for _ in range(self.num_underflow):
            self.output.write(bit ^ 1)
        self.num_underflow = 0


    def underflow(self):
        self.num_underflow += 1


class ArithmeticDecoder(ArithmeticCoderBase):
    """Reads from an arithmetic-coded bit stream and decodes symbols."""

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


    def read(self, cumulative: np.ndarray, alphabet_size: int) -> int:
        """
        Decodes the next symbol based on the given frequency table and returns it.
        Also updates this arithmetic coder's state and may read in some bits.
        """
        total = int(cumulative[-1])
        current_range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // current_range

        # `searchsorted(..., side="right") - 1` finds the symbol whose interval
        # contains `value`.
        symbol = int(np.searchsorted(cumulative, value, side="right") - 1)
        self.update(cumulative, symbol)
        return symbol
    

    def shift(self):
        self.code = ((self.code << 1) & self.MASK) | self.read_code_bit()

    def underflow(self):
        self.code = (self.code & self.TOP_MASK) | ((self.code << 1) & (self.MASK >> 1)) | self.read_code_bit()

    def read_code_bit(self):
        """Returns the next bit (0 or 1) from the input stream. The end of stream is treated as an infinite number of trailing zeros."""
        temp = self.input.read()
        if temp == -1:
            temp = 0
        return temp
    

# =============================================================================
# LLM-facing compressor/decompressor
# =============================================================================

class LLMCompressor:
    def __init__(
        self,
        *,
        algorithm: str = "AC",
        alphabet_size: Optional[int] = None,
        delta: float = 1e-3,
        r: Optional[float] = None,
        statesize: int = 32,
        total: int = 262144):

        self.bitout = BitOutputStream()
        self.encoder = ArithmeticEncoder(statesize, self.bitout)
        self.cross_entropy_sum = 0.0
        self.token_count = 0

        self.algorithm = algorithm.upper()
        self.total = total

        self.helper_ones = 0
        self.helper_count = 0

        if self.algorithm == "PMATIC":
            if alphabet_size is None:
                raise ValueError("alphabet_size must be provided for PMATIC")

            self.alphabet_size = alphabet_size
            self.padded_alphabet_size = next_power_of_two(alphabet_size)
            self.bit_width = math.ceil(math.log2(self.padded_alphabet_size))

            self.delta = delta
            self.r = choose_pmatic_r(delta) if r is None else r

        elif self.algorithm == "AC":
            self.alphabet_size = alphabet_size
            self.delta = delta
            self.r = r

        else:
            raise ValueError("algorithm must be one of: AC, PMATIC")


    def next_token(self, correct_token_idx: int, probs: np.ndarray):
        # Switch between different encoding algorithms
        if self.algorithm == "AC":
            self._next_token_ac(correct_token_idx, probs)
        elif self.algorithm == "PMATIC":
            self._next_token_pmatic(correct_token_idx, probs)

    def _next_token_ac(self, correct_token_idx: int, probs: np.ndarray):
        # AC next token prediction 
        prob = probs[correct_token_idx]
        self.cross_entropy_sum += -math.log2(max(prob, 1e-300))
        self.token_count += 1

        self.encoder.write(build_cumul(probs, total=self.total), correct_token_idx)

    def _next_token_pmatic(self, correct_token_idx: int, probs: np.ndarray):
        # PMATIC next token prediction
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
        prefix: List[int] = []

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

        
    def compress(self, encoding: Optional[str] = None, rank_list=None):
        encoding_name = self.algorithm if encoding is None else encoding.upper()

        if encoding_name in {"AC", "PMATIC"}:
            if encoding_name != self.algorithm:
                raise ValueError(
                    f"compress encoding '{encoding}' does not match compressor "
                    f"algorithm '{self.algorithm}'"
                )
            self.encoder.finish()
            return self.bitout.get_bits()

        elif encoding in {"bitpacked", "BITPACKED"}:
            assert rank_list is not None, "rank_list must be provided for bitpacked encoding"
            max_rank = max(rank_list) if rank_list else 0
            num_bits = max_rank.bit_length()
            return "".join(format(rank, f"0{num_bits}b") for rank in rank_list)

        elif encoding in {"huffman", "HUFFMAN"}:
            assert rank_list is not None, "rank_list must be provided for huffman encoding"
            codebook = build_huffman_code(rank_list)
            bit_string = huffman_encode(rank_list, codebook)
            return bit_string, codebook

        elif encoding in {"zstd", "ZSTD"}:
            assert rank_list is not None, "rank_list must be provided for zstd encoding"
            ranks_arr = np.asarray(rank_list, dtype=np.uint32)
            raw_bytes = ranks_arr.tobytes()
            cctx = zstd.ZstdCompressor()
            compressed = cctx.compress(raw_bytes)
            return "".join(f"{b:08b}" for b in compressed)

        else:
            raise NotImplementedError(f"Encoding method '{encoding}' is not implemented.")
    
    def get_cross_entropy(self):
        return self.cross_entropy_sum

    def helper_one_fraction(self):
        return 0.0 if self.helper_count == 0 else self.helper_ones / self.helper_count


class LLMDecompressor:
    def __init__(
        self,
        code,
        *,
        algorithm: str = "AC",
        alphabet_size: Optional[int] = None,
        delta: float = 1e-3,
        r: Optional[float] = None,
        statesize: int = 32,
        total: int = 262144,
    ):
        self.algorithm = algorithm.upper()
        self.total = total

        self.decoder = ArithmeticDecoder(statesize, BitInputStream(code))

        if self.algorithm == "PMATIC":
            if alphabet_size is None:
                raise ValueError("alphabet_size must be provided for PMATIC")

            self.alphabet_size = alphabet_size
            self.padded_alphabet_size = next_power_of_two(alphabet_size)
            self.bit_width = math.ceil(math.log2(self.padded_alphabet_size))

            self.delta = delta
            self.r = choose_pmatic_r(delta) if r is None else r

        elif self.algorithm == "AC":
            self.alphabet_size = alphabet_size
            self.delta = delta
            self.r = r

        else:
            raise ValueError("algorithm must be one of: AC, PMATIC")

    def decompress(self, probs: np.ndarray) -> int:
        if self.algorithm == "AC":
            return self._decompress_ac(probs)
        elif self.algorithm == "PMATIC":
            return self._decompress_pmatic(probs)

        raise AssertionError("unreachable")

    def _decompress_ac(self, probs: np.ndarray) -> int:
        cumul = build_cumul(probs, total=self.total)
        return self.decoder.read(cumul, len(probs))

    def _decompress_pmatic(self, probs: np.ndarray) -> int:
        probs = pad_probs_to_power_of_two(probs, self.alphabet_size)

        prefix: List[int] = []
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
                f"Decoded dummy token {token_idx}. "
                "This means the induced conditional probability mismatch exceeded "
                "PMATIC tolerance, or the bitstream is corrupted."
            )

        return token_idx
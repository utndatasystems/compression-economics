"""The old coding imports must point to the implementations in src.coding."""

import src.encoding as old_encoding
import src.encoding_utils as old_utils
import src.multistream_ac as old_multistream
import src.parallel_ac as old_parallel
from src.coding import encoding, encoding_utils, multistream_ac, parallel_ac


def test_compatibility_imports_resolve_to_coding_package():
    assert old_encoding.LLMCompressor is encoding.LLMCompressor
    assert old_utils.build_cumul is encoding_utils.build_cumul
    assert old_multistream.MultistreamACEncoder is multistream_ac.MultistreamACEncoder
    assert old_parallel.prepare_parallel_encoder is parallel_ac.prepare_parallel_encoder
    assert old_multistream._HEADER is multistream_ac._HEADER

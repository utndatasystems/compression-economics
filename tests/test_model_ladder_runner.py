from pathlib import Path

import pytest

from scripts.plain_text_compression.evaluate_model_ladder import common_utf8_prefix, split_contiguous


def test_common_utf8_prefix_never_returns_partial_codepoint(tmp_path: Path):
    path = tmp_path / "unicode.txt"
    path.write_text("ab€cd", encoding="utf-8")
    text, raw = common_utf8_prefix(path, 4)
    assert text == "ab"
    assert raw == b"ab"


def test_split_contiguous_preserves_order_and_balances_batches():
    assert split_contiguous(list(range(10)), 3) == [
        list(range(4)), list(range(4, 7)), list(range(7, 10))
    ]


@pytest.mark.parametrize("batch_size", [0, 5])
def test_split_contiguous_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch size"):
        split_contiguous([1, 2, 3, 4], batch_size)

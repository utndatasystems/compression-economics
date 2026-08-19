from types import SimpleNamespace

from src.prediction import TokenDataPreparer
from src.utils import check_mismatch
from scripts.generate_adversarial import ascii_byte_token_ids


def test_predictor_preserves_text_line_endings(tmp_path):
    input_path = tmp_path / "adversarial.txt"
    input_path.write_bytes(b"a\rb\r\nc\n")

    preparer = TokenDataPreparer.__new__(TokenDataPreparer)

    assert preparer._get_data_from_file(str(input_path)) == "a\rb\r\nc\n"


def test_mismatch_check_distinguishes_carriage_return(tmp_path):
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_bytes(b"a\rb")
    output_path.write_bytes(b"a\nb")

    assert check_mismatch(input_path, output_path) is False


class FakeCharacterTokenizer:
    vocab_size = 256

    def encode(self, text, **_kwargs):
        return list(text.encode("utf-8"))


def test_token_limit_slices_without_rewriting_text(tmp_path, monkeypatch):
    import src.prediction as prediction_module

    input_path = tmp_path / "adversarial.txt"
    input_path.write_bytes(bytes([97, 32, 98, 13, 10, 99]))
    monkeypatch.setattr(
        prediction_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeCharacterTokenizer(),
    )
    args = SimpleNamespace(
        input_path=str(input_path),
        text_input=None,
        is_mamba=False,
        model_name="fake",
        first_n_tokens=4,
        reduce_tokens=False,
    )

    preparer = TokenDataPreparer(args)

    assert preparer.data_tokens == [97, 32, 98, 13]


class FakeAsciiTokenizer:
    vocab_size = 128

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token_id) for token_id in token_ids)

    def encode(self, text, **_kwargs):
        return [ord(character) for character in text]


def test_ascii_generation_alphabets_are_exactly_one_byte():
    tokenizer = FakeAsciiTokenizer()

    assert ascii_byte_token_ids(
        tokenizer, printable_only=False
    ) == list(range(128))
    assert ascii_byte_token_ids(
        tokenizer, printable_only=True
    ) == list(range(32, 127))

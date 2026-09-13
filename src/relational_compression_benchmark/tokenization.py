"""Lossless byte tokenization and compact fixed-width token-ID packing.

The model tokenizer consumes Unicode rather than arbitrary bytes. Each block is
therefore mapped bijectively through Latin-1 before tokenization and mapped back
after decoding. No special tokens are inserted or removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


BYTE_MAPPING = "latin-1-v1"
DEFAULT_TOKENIZER_CACHE = Path(".cache")
_REQUIRED_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
)


class TokenizerBackend(Protocol):
    """Minimal Hugging Face tokenizer surface used by the benchmark."""

    def __len__(self) -> int: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Convert text into token IDs."""
        ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        """Convert token IDs back into text."""
        ...


def token_id_width(vocabulary_size: int) -> int:
    """Return the smallest whole-byte width that covers a vocabulary."""
    if vocabulary_size < 1:
        raise ValueError("vocabulary_size must be positive")
    return max(1, ((vocabulary_size - 1).bit_length() + 7) // 8)


def pack_token_ids(token_ids: Sequence[int], width: int) -> bytes:
    """Pack unsigned token IDs into stable big-endian fixed-width integers."""
    if width < 1:
        raise ValueError("token ID width must be positive")
    limit = 1 << (8 * width)
    if width == 1:
        if any(not 0 <= token_id < limit for token_id in token_ids):
            raise ValueError("token ID does not fit in one byte")
        return bytes(token_ids)

    output = bytearray(len(token_ids) * width)
    for index, token_id in enumerate(token_ids):
        if not 0 <= token_id < limit:
            raise ValueError(f"token ID {token_id} does not fit in {width} bytes")
        offset = index * width
        if width == 3:
            output[offset] = token_id >> 16
            output[offset + 1] = token_id >> 8 & 0xFF
            output[offset + 2] = token_id & 0xFF
        else:
            output[offset : offset + width] = token_id.to_bytes(width, "big")
    return bytes(output)


def unpack_token_ids(data: bytes, width: int) -> list[int]:
    """Decode big-endian fixed-width token IDs and reject partial values."""
    if width < 1:
        raise ValueError("token ID width must be positive")
    if len(data) % width:
        raise ValueError("packed token payload ends with a partial token ID")
    if width == 1:
        return list(data)
    if width == 3:
        return [
            data[offset] << 16
            | data[offset + 1] << 8
            | data[offset + 2]
            for offset in range(0, len(data), 3)
        ]
    return [
        int.from_bytes(data[offset : offset + width], "big")
        for offset in range(0, len(data), width)
    ]


@dataclass(frozen=True)
class TokenizerAdapter:
    """Pinned tokenizer plus its byte mapping, packing, and storage charge."""

    backend: TokenizerBackend
    name: str
    revision: str
    vocabulary_size: int
    id_width: int
    asset_bytes: int

    @classmethod
    def from_backend(
        cls,
        backend: TokenizerBackend,
        *,
        name: str,
        revision: str,
        asset_bytes: int = 0,
    ) -> "TokenizerAdapter":
        """Wrap a tokenizer backend, deriving its fixed token-ID width."""
        vocabulary_size = len(backend)
        return cls(
            backend=backend,
            name=name,
            revision=revision,
            vocabulary_size=vocabulary_size,
            id_width=token_id_width(vocabulary_size),
            asset_bytes=asset_bytes,
        )

    @property
    def descriptor(self) -> dict[str, str | int]:
        """Return the tokenizer identity persisted in token archives."""
        return {
            "byte_mapping": BYTE_MAPPING,
            "name": self.name,
            "revision": self.revision,
            "token_id_width": self.id_width,
            "vocabulary_size": self.vocabulary_size,
        }

    def encode_bytes(self, data: bytes) -> list[int]:
        """Map arbitrary bytes to token IDs without inserting special tokens."""
        token_ids = self.backend.encode(
            data.decode("latin-1"), add_special_tokens=False
        )
        if any(
            not 0 <= token_id < self.vocabulary_size for token_id in token_ids
        ):
            raise ValueError("tokenizer emitted an ID outside its declared vocabulary")
        return token_ids

    def decode_bytes(self, token_ids: Sequence[int]) -> bytes:
        """Recover exact bytes from token IDs or reject a lossy decode."""
        if any(
            not 0 <= token_id < self.vocabulary_size for token_id in token_ids
        ):
            raise ValueError("archive contains an out-of-vocabulary token ID")
        text = self.backend.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        try:
            return text.encode("latin-1")
        except UnicodeEncodeError as error:
            raise ValueError(
                "tokenizer decode violated the Latin-1 byte mapping"
            ) from error

    def validate_descriptor(self, descriptor: dict[str, Any]) -> None:
        """Ensure an archive was created by this exact tokenizer contract."""
        if descriptor != self.descriptor:
            raise ValueError("tokenizer does not match the archive descriptor")


def _cached_tokenizer_asset_bytes(
    name: str, revision: str, cache_dir: Path
) -> int:
    """Sum pinned tokenizer files available in the local Hugging Face cache."""
    from huggingface_hub import try_to_load_from_cache

    paths: set[Path] = set()
    missing = []
    for filename in _REQUIRED_TOKENIZER_ASSETS:
        cached = try_to_load_from_cache(
            name,
            filename,
            cache_dir=cache_dir,
            revision=revision,
        )
        if isinstance(cached, str):
            paths.add(Path(cached).resolve())
        else:
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            f"Missing tokenizer assets {missing} for "
            f"{name}@{revision} in {cache_dir}"
        )
    return sum(path.stat().st_size for path in paths)


def load_tokenizer(
    name: str,
    revision: str,
    *,
    cache_dir: Path = DEFAULT_TOKENIZER_CACHE,
) -> TokenizerAdapter:
    """Load a pinned tokenizer locally, never initiating a download."""
    from transformers import AutoTokenizer

    try:
        backend = AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except OSError as error:
        raise FileNotFoundError(
            f"Tokenizer {name}@{revision} is not cached. Run "
            "main.py cidr prepare-tokenizer explicitly first."
        ) from error
    return TokenizerAdapter.from_backend(
        backend,
        name=name,
        revision=revision,
        asset_bytes=_cached_tokenizer_asset_bytes(name, revision, cache_dir),
    )

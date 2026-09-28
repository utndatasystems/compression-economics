"""Reference MSAP format: paired forward/backward MSAC streams.

The existing arithmetic coder emits fixed termination bits. This format shares
the two terminal *bytes* when their fixed significant bits are compatible; it
does not choose alternative final arithmetic codes from the coder interval.
"""
from __future__ import annotations

import struct

from src.coding.multistream_ac import MultistreamACDecoder, MultistreamACEncoder

_HEADER = struct.Struct(">4sBBHII")
_STREAM = struct.Struct(">QQI")
_MAGIC = b"MSAP"
_SHARED = 1 << 63
_REVERSE_BITS = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


def _mask(bit_count: int) -> int:
    return (0xFF << (8 - ((bit_count - 1) % 8 + 1))) & 0xFF


def _pack_pair(forward: bytes, forward_bits: int, backward: bytes, backward_bits: int):
    """Return physical pair bytes and whether the final byte is shared."""
    reversed_bytes = backward.translate(_REVERSE_BITS)[::-1]
    forward_mask = _mask(forward_bits)
    backward_mask = _REVERSE_BITS[_mask(backward_bits)]
    if (forward[-1] ^ reversed_bytes[0]) & forward_mask & backward_mask:
        return forward + reversed_bytes, False
    terminal = (forward[-1] & forward_mask) | (reversed_bytes[0] & backward_mask)
    return forward[:-1] + bytes((terminal,)) + reversed_bytes[1:], True


def _unpack_pair(data: bytes, forward_size: int, backward_size: int,
                 forward_bits: int, backward_bits: int, shared: bool):
    forward = bytearray(data[:forward_size])
    backward = bytearray(data[forward_size - int(shared):])
    if len(forward) != forward_size or len(backward) != backward_size:
        raise ValueError("truncated paired payload")
    backward = bytearray(backward[::-1].translate(_REVERSE_BITS))
    if shared:
        forward[-1] &= _mask(forward_bits)
        backward[-1] &= _mask(backward_bits)
    return bytes(forward), bytes(backward)


def pack_standard_archive(archive: bytes) -> bytes:
    """Convert a validated MSAC v1/v2 archive without changing coded symbols."""
    standard = MultistreamACDecoder(archive)
    magic, version, state_bits, flags, total, count = _HEADER.unpack_from(archive)
    payload_offset = standard.framing_bytes
    streams = []
    descriptors = []
    for stream_id, bit_count in enumerate(standard.bit_counts):
        size = (bit_count + 7) // 8
        streams.append(archive[payload_offset:payload_offset + size])
        payload_offset += size
        descriptors.append(list(_STREAM.unpack_from(archive, _HEADER.size + stream_id * _STREAM.size)))
    payloads = []
    for stream_id in range(0, count, 2):
        if stream_id + 1 == count:
            payloads.append(streams[stream_id])
            continue
        pair, shared = _pack_pair(streams[stream_id], standard.bit_counts[stream_id],
                                  streams[stream_id + 1], standard.bit_counts[stream_id + 1])
        payloads.append(pair)
        if shared:
            descriptors[stream_id][1] |= _SHARED
    return b"".join((
        _HEADER.pack(_MAGIC, 1, state_bits, flags, total, count),
        *(_STREAM.pack(*entry) for entry in descriptors),
        *payloads,
    ))


def unpack_paired_archive(archive: bytes) -> bytes:
    """Restore an ordinary MSAC archive for the existing checked decoder."""
    if len(archive) < _HEADER.size:
        raise ValueError("truncated MSAP header")
    magic, version, state_bits, flags, total, count = _HEADER.unpack_from(archive)
    if magic != _MAGIC or version != 1 or flags not in (0, 1) or count < 1:
        raise ValueError("invalid MSAP header")
    directory_end = _HEADER.size + count * _STREAM.size
    if len(archive) < directory_end:
        raise ValueError("truncated MSAP directory")
    entries = [list(_STREAM.unpack_from(archive, _HEADER.size + i * _STREAM.size))
               for i in range(count)]
    offset = directory_end
    streams = []
    for stream_id in range(0, count, 2):
        first = entries[stream_id]
        shared = bool(first[1] & _SHARED)
        first[1] &= ~_SHARED
        if first[1] < 1 or (stream_id + 1 == count and shared):
            raise ValueError("invalid MSAP stream descriptor")
        if stream_id + 1 == count:
            size = (first[1] + 7) // 8
            streams.append(archive[offset:offset + size])
            offset += size
            continue
        second = entries[stream_id + 1]
        if second[1] < 1 or second[1] & _SHARED:
            raise ValueError("invalid MSAP stream descriptor")
        first_size = (first[1] + 7) // 8
        second_size = (second[1] + 7) // 8
        size = first_size + second_size - int(shared)
        if offset + size > len(archive):
            raise ValueError("truncated MSAP paired payload")
        streams.extend(_unpack_pair(archive[offset:offset + size],
                                    first_size, second_size, first[1], second[1], shared))
        offset += size
    if offset != len(archive):
        raise ValueError("trailing MSAP bytes")
    standard = b"".join((
        _HEADER.pack(b"MSAC", 2 if flags else 1, state_bits, flags, total, count),
        *(_STREAM.pack(*entry) for entry in entries),
        *streams,
    ))
    MultistreamACDecoder(standard)  # CRC, padding, and directory validation
    return standard


class PairedACEncoder:
    """MSAC encoder with a paired-byte output format."""

    def __init__(self, *args, **kwargs):
        self._encoder = MultistreamACEncoder(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._encoder, name)

    def encode(self, *args, **kwargs):
        return self._encoder.encode(*args, **kwargs)

    def encode_interval(self, *args, **kwargs):
        return self._encoder.encode_interval(*args, **kwargs)

    def finish(self) -> bytes:
        return pack_standard_archive(self._encoder.finish())


class PairedACDecoder:
    """Decode MSAP by restoring each stream's logical byte order."""

    def __init__(self, archive: bytes, **kwargs):
        self._standard = unpack_paired_archive(archive)
        self._decoder = MultistreamACDecoder(self._standard, **kwargs)
        self.shared_pairs = sum(
            bool(_STREAM.unpack_from(archive, _HEADER.size + i * _STREAM.size)[1] & _SHARED)
            for i in range(0, self._decoder.stream_count, 2)
        )

    def __getattr__(self, name):
        return getattr(self._decoder, name)

    def decode(self, *args, **kwargs):
        return self._decoder.decode(*args, **kwargs)

    def assert_complete(self):
        return self._decoder.assert_complete()

"""Compatibility import for :mod:`src.coding.multistream_ac`."""

from src.coding.multistream_ac import *  # noqa: F401,F403
from src.coding import multistream_ac as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

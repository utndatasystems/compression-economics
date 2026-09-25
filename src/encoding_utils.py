"""Compatibility import for :mod:`src.coding.encoding_utils`."""

from src.coding.encoding_utils import *  # noqa: F401,F403
from src.coding import encoding_utils as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

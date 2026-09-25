"""Compatibility import for :mod:`src.coding.encoding`."""

from src.coding.encoding import *  # noqa: F401,F403
from src.coding import encoding as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

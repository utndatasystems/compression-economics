"""Compatibility import for :mod:`src.coding.parallel_ac`."""

from src.coding.parallel_ac import *  # noqa: F401,F403
from src.coding import parallel_ac as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

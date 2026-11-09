"""Geometry on the wrapped field."""

from __future__ import annotations

import numpy as np

from .config import FIELD


def wrap(pos: np.ndarray) -> np.ndarray:
    return pos - FIELD * np.floor(pos / FIELD)


def shortest(delta: np.ndarray) -> np.ndarray:
    """The shortest way across the wrapped field for a raw position difference."""
    return delta - FIELD * np.rint(delta / FIELD)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def length(delta: np.ndarray) -> np.ndarray:
    return np.hypot(delta[..., 0], delta[..., 1])

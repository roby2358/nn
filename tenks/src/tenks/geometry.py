"""Vector helpers over the last axis of (..., 2) arrays."""

from __future__ import annotations

import numpy as np


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def length(delta: np.ndarray) -> np.ndarray:
    return np.hypot(delta[..., 0], delta[..., 1])

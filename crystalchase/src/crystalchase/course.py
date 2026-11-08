"""A course is a seeded start pose plus a crystal sequence. Same seed, same course."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import COURSE_LEN, FIELD_H, FIELD_W, MIN_HOP, SPAWN_PAD, TAU


@dataclass(frozen=True)
class Course:
    seed: int
    start: np.ndarray      # (2,) position
    heading: float
    crystals: np.ndarray   # (COURSE_LEN, 2)


def _spot(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(SPAWN_PAD, FIELD_W - SPAWN_PAD), rng.uniform(SPAWN_PAD, FIELD_H - SPAWN_PAD)])


def _hop_from(rng: np.random.Generator, anchors: list[np.ndarray]) -> np.ndarray:
    while True:
        p = _spot(rng)
        if all(np.hypot(*(p - a)) >= MIN_HOP for a in anchors):
            return p


def make_course(seed: int) -> Course:
    rng = np.random.default_rng(seed)
    start = _spot(rng)
    crystals = [_hop_from(rng, [start])]
    for _ in range(COURSE_LEN - 1):
        crystals.append(_hop_from(rng, [crystals[-1]]))
    return Course(seed, start, float(rng.uniform(0, TAU)), np.array(crystals))


def make_ring_course(seed: int, spots: int) -> Course:
    """`spots` crystals that close into a ring: flight wraps from the last back to the
    first, and that hop keeps the MIN_HOP spacing too."""
    rng = np.random.default_rng(seed)
    start = _spot(rng)
    crystals = [_hop_from(rng, [start])]
    for _ in range(spots - 2):
        crystals.append(_hop_from(rng, [crystals[-1]]))
    crystals.append(_hop_from(rng, [crystals[-1], crystals[0]]))
    return Course(seed, start, float(rng.uniform(0, TAU)), np.array(crystals))

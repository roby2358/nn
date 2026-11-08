"""A scenario is everything random about a round, drawn up front from one seed, so every
ship on a scenario meets exactly the same rocks and only its flying can differ."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DRIFT_TABLE, FIELD, LEFT_EDGE_ODDS, TAU, WAVE_MAX, WAVE_TABLE


@dataclass(frozen=True)
class Scenario:
    seed: int
    heading: float
    spots: np.ndarray    # (WAVE_TABLE, WAVE_MAX, 2) where each wave's rocks appear
    drifts: np.ndarray   # (DRIFT_TABLE, 2) angle, and speed as a fraction of the size's range


def make_scenario(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    spots = rng.random((WAVE_TABLE, WAVE_MAX, 2)) * FIELD
    on_left_edge = rng.random((WAVE_TABLE, WAVE_MAX)) < LEFT_EDGE_ODDS
    spots[..., 0] = np.where(on_left_edge, 0.0, spots[..., 0])
    spots[..., 1] = np.where(on_left_edge, spots[..., 1], 0.0)
    drifts = np.stack([rng.uniform(0.0, TAU, DRIFT_TABLE), rng.random(DRIFT_TABLE)], axis=1)
    return Scenario(seed, float(rng.uniform(0.0, TAU)), spots, drifts)

"""A scenario is the starting layout of one battle, drawn up front from one seed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FIELD_H, FIELD_W, SPAWN_GAP, SPAWN_HALF_BAND, SPAWN_LINE, SPAWN_MARGIN, TAU, TANKS, TEAM_SIZE
from .geometry import length


@dataclass(frozen=True)
class Scenario:
    seed: int
    spots: np.ndarray      # (TANKS, 2)
    headings: np.ndarray   # (TANKS,)


def _place_team(rng: np.random.Generator, line: float) -> np.ndarray:
    """TEAM_SIZE spots near a vertical line, no two closer than SPAWN_GAP."""
    placed: list[np.ndarray] = []
    while len(placed) < TEAM_SIZE:
        spot = np.array([line + rng.uniform(-SPAWN_HALF_BAND, SPAWN_HALF_BAND),
                         rng.uniform(SPAWN_MARGIN, FIELD_H - SPAWN_MARGIN)])
        if all(length(spot - other) >= SPAWN_GAP for other in placed):
            placed.append(spot)
    return np.array(placed)


def make_scenario(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    spots = np.vstack([_place_team(rng, SPAWN_LINE), _place_team(rng, FIELD_W - SPAWN_LINE)])
    return Scenario(seed, spots, rng.uniform(0.0, TAU, TANKS))

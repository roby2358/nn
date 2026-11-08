"""Pilots turn observations into actions. Each one flies a batch of ships:
act(obs (K, 6)) -> actions (K,). A Crew stitches several pilots into one Flock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import (BOT_AIM_TOLERANCE, BOT_ARRIVE_SPEED, BOT_BRAKE_SHARE, BOT_DEADBAND, BOT_MAX_SPEED,
                     BOT_MIN_ERROR, DIST_SCALE, N_ACTIONS, N_HIDDEN, N_INPUTS, SPEED_SCALE, THRUST)
from .flock import encode_action

SHAPES = [(N_INPUTS, N_HIDDEN), (N_HIDDEN,), (N_HIDDEN, N_ACTIONS), (N_ACTIONS,)]
GENOME = sum(int(np.prod(s)) for s in SHAPES)


class Pilot(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray: ...


def _fresh(rng: np.random.Generator, n: int, shape: tuple) -> np.ndarray:
    """Weight matrices scaled by 1/sqrt(fan-in); biases start at zero."""
    if len(shape) == 1:
        return np.zeros((n, shape[0]))
    return rng.standard_normal((n, shape[0] * shape[1])) / np.sqrt(shape[0])


def random_genomes(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, GENOME) fresh nets."""
    return np.concatenate([_fresh(rng, n, s) for s in SHAPES], axis=1)


class Brains:
    """A batch of nets, one per ship: 6 inputs -> 16 tanh -> 6 logits, greedy argmax.
    Weights are one flat genome per row, so evolution can treat them as vectors."""

    def __init__(self, genomes: np.ndarray) -> None:
        self.genomes = genomes
        n, cut = len(genomes), 0
        self.layers = []
        for s in SHAPES:
            size = int(np.prod(s))
            self.layers.append(genomes[:, cut:cut + size].reshape((n, *s)))
            cut += size

    def act(self, obs: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self.layers
        h = np.tanh(np.matmul(obs[:, None, :], W1)[:, 0] + b1)
        logits = np.matmul(h[:, None, :], W2)[:, 0] + b2
        return np.argmax(logits, axis=1)

    def save(self, path: Path) -> None:
        np.savez(path, champion=self.genomes[0])

    @classmethod
    def load(cls, path: Path) -> "Brains":
        with np.load(path) as f:
            g = f["champion"]
        if g.shape != (GENOME,):
            raise ValueError(f"{path} holds a {g.shape} genome; this build expects ({GENOME},)")
        return cls(g[None, :])


def load_pilot(path: Path) -> Pilot:
    """Load either a standard evolved genome or a layered calculated racer."""
    with np.load(path) as f:
        layered = "neural_layers" in f
    if layered:
        from .encoded_bot import EncodedBot
        return EncodedBot.load(path)
    return Brains.load(path)


class AutoPilot:
    """Hand-coded baseline that sees exactly what the nets see. It picks the velocity it
    wants (toward the crystal, slow enough to brake into it), then points along the
    velocity *error* and burns when roughly aligned: the classic thrust-vector chase."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        sinb, cosb, dist, vf, vs = obs[:, 0], obs[:, 1], obs[:, 2] * DIST_SCALE, obs[:, 3], obs[:, 4]
        want = np.minimum(BOT_MAX_SPEED, np.sqrt(BOT_ARRIVE_SPEED ** 2 + 2 * THRUST * BOT_BRAKE_SHARE * dist))
        ex, ey = cosb * want - vf * SPEED_SCALE, sinb * want - vs * SPEED_SCALE
        aim = np.arctan2(ey, ex)
        turn = np.where(np.abs(aim) < BOT_DEADBAND, 0, np.sign(aim)).astype(np.int64)
        thrust = ((np.abs(aim) < BOT_AIM_TOLERANCE) & (np.hypot(ex, ey) > BOT_MIN_ERROR)).astype(np.int64)
        return encode_action(turn, thrust)


class HumanPilot:
    """Keyboard ship. The UI sets turn and thrust from the arrow keys every frame."""

    def __init__(self) -> None:
        self.turn = 0
        self.thrust = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.full(len(obs), encode_action(self.turn, self.thrust), dtype=np.int64)


@dataclass(frozen=True)
class Seat:
    pilot: Pilot
    ships: slice


class Crew:
    """Several pilots flying consecutive blocks of ships in one Flock."""

    def __init__(self, seats: list[Seat]) -> None:
        self.seats = seats

    def act(self, obs: np.ndarray) -> np.ndarray:
        out = np.zeros(len(obs), dtype=np.int64)
        for seat in self.seats:
            out[seat.ships] = seat.pilot.act(obs[seat.ships])
        return out

"""Pilots turn observations into commands: act(obs (K, N_INPUTS)) -> commands (K, 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .config import CMD_FIRE, CMD_THRUST, CMD_TURN, N_HIDDEN, N_INPUTS, N_OUTPUTS

SHAPES = [(N_INPUTS, N_HIDDEN), (N_HIDDEN,), (N_HIDDEN, N_OUTPUTS), (N_OUTPUTS,)]
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
    """A batch of nets, one per ship: inputs -> tanh hidden -> turn and moods (tanh), thrust and fire
    (squashed to 0..1). Each row of `genomes` is one flat net, so evolution can treat it as a vector."""

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
        out = np.tanh(np.matmul(h[:, None, :], W2)[:, 0] + b2)
        out[:, [CMD_THRUST, CMD_FIRE]] = (out[:, [CMD_THRUST, CMD_FIRE]] + 1.0) / 2.0
        return out

    def save(self, path: Path) -> None:
        np.savez(path, champion=self.genomes[0])

    @classmethod
    def load(cls, path: Path) -> "Brains":
        with np.load(path) as f:
            g = f["champion"]
        if g.shape != (GENOME,):
            raise ValueError(f"{path} holds a {g.shape} genome; this build expects ({GENOME},)")
        return cls(g[None, :])


class HumanPilot:
    """Keyboard ship. The UI sets turn, thrust and fire from the keys every frame."""

    def __init__(self) -> None:
        self.turn = 0.0
        self.thrust = 0.0
        self.fire = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        command = np.zeros(N_OUTPUTS)
        command[[CMD_TURN, CMD_THRUST, CMD_FIRE]] = self.turn, self.thrust, self.fire
        return np.tile(command, (len(obs), 1))

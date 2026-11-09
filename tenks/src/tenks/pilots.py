"""Nets turn observations into commands: act(obs (K, N_INPUTS)) -> commands (K, N_OUTPUTS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CMD_SHOOT, N_HIDDEN, N_INPUTS, N_OUTPUTS

SHAPES = [(N_INPUTS, N_HIDDEN), (N_HIDDEN,), (N_HIDDEN, N_OUTPUTS), (N_OUTPUTS,)]
GENOME = sum(int(np.prod(s)) for s in SHAPES)


def _fresh(rng: np.random.Generator, n: int, shape: tuple) -> np.ndarray:
    """Weight matrices scaled by 1/sqrt(fan-in); biases start at zero."""
    if len(shape) == 1:
        return np.zeros((n, shape[0]))
    return rng.standard_normal((n, shape[0] * shape[1])) / np.sqrt(shape[0])


def random_genomes(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, GENOME) fresh nets."""
    return np.concatenate([_fresh(rng, n, s) for s in SHAPES], axis=1)


class Brains:
    """A batch of nets, one per tank: inputs -> tanh hidden -> treads in -1..1 and shoot in 0..1.
    Each row of `genomes` is one flat net, so evolution can treat it as a vector."""

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
        out[:, CMD_SHOOT] = (out[:, CMD_SHOOT] + 1.0) / 2.0
        return out


class HumanDriver:
    """The keyboard's one tank. The UI writes `command` (left, right, shoot) every frame."""

    def __init__(self) -> None:
        self.command = np.zeros(N_OUTPUTS)


class Commandeered:
    """Nets drive every tank except one, which does what the human says."""

    def __init__(self, brains: Brains, human: HumanDriver, tank: int) -> None:
        self.brains = brains
        self.human = human
        self.tank = tank

    def act(self, obs: np.ndarray) -> np.ndarray:
        out = self.brains.act(obs)
        out[self.tank] = self.human.command
        return out


def save_champion(path: Path, genome: np.ndarray) -> None:
    np.savez(path, champion=genome)


def load_roster(path: Path) -> np.ndarray:
    """(n, GENOME) from a swarm file (its whole population) or a weights file (its one champion)."""
    with np.load(path) as f:
        roster = f["population"].copy() if "population" in f else f["champion"][None].copy()
    if roster.shape[1:] != (GENOME,):
        raise ValueError(f"{path} holds {roster.shape[1:]} genomes; this build expects ({GENOME},)")
    return roster


def seat(roster: np.ndarray, count: int) -> np.ndarray:
    """(count, GENOME): the first `count` nets of a roster, starting over if it runs out."""
    return roster[np.arange(count) % len(roster)]

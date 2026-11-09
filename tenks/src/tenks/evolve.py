"""Neuroevolution with one net per tank. Each generation the population is dealt at random into
battles, a few times over; the top nets by mean fitness survive unchanged and the rest are
mutated (sometimes crossed) copies of them."""

from __future__ import annotations

import sys
import time
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from .config import (CROSSOVER_GENE_ODDS, CROSSOVER_RATE, DEALS_PER_GEN, ELITES, MUTATION_SIGMAS, POP_SIZE, RUN_STEPS,
                     TANKS)
from .pilots import Brains, load_roster, random_genomes, save_champion
from .scenario import Scenario, make_scenario
from .worlds import Worlds

SWARM = Path("swarm.npz")       # the whole population plus the log history
WEIGHTS = Path("weights.npz")   # the leader alone


@dataclass(frozen=True)
class GenStats:
    gen: int
    best: float
    mean: float
    best_kills: float
    kills: float
    friendly_kills: float
    near_misses: float
    shots: float
    survived: float
    decided: float


def fight(worlds: Worlds, brains: Brains) -> Worlds:
    while not worlds.done:
        worlds.step(brains.act(worlds.observe()))
    return worlds


def deal(rng: np.random.Generator, population: int, deals: int) -> np.ndarray:
    """(battles, TANKS) of population indices. Each deal seats every net exactly once."""
    return np.vstack([rng.permutation(population).reshape(-1, TANKS) for _ in range(deals)])


def evaluate(genomes: np.ndarray, seats: np.ndarray, scenarios: list[Scenario],
             gen: int) -> tuple[np.ndarray, GenStats]:
    """Each net's score is its mean over the battles it was seated in."""
    worlds = fight(Worlds(scenarios, RUN_STEPS), Brains(genomes[seats.ravel()]))
    battles = np.bincount(seats.ravel(), minlength=len(genomes))

    def per_net(values: np.ndarray) -> np.ndarray:
        return np.bincount(seats.ravel(), values.ravel(), minlength=len(genomes)) / battles

    fitness, kills = per_net(worlds.fitness()), per_net(worlds.kills)
    best = int(np.argmax(fitness))
    stats = GenStats(gen, float(fitness[best]), float(fitness.mean()), float(kills[best]), float(kills.mean()),
                     float(worlds.friendly_kills.mean()), float(worlds.near_misses.mean()), float(worlds.fired.mean()),
                     float(worlds.survived.mean()), float(worlds.over.mean()))
    return fitness, stats


def reproduce(rng: np.random.Generator, genomes: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """Elites first (best at row 0), then children of elites: a rank-weighted parent,
    sometimes uniformly crossed with a second elite, plus Gaussian noise at one of a few scales."""
    elites = genomes[np.argsort(-fitness)[:ELITES]]
    weights = 1.0 / np.arange(1, ELITES + 1)
    weights /= weights.sum()
    n = len(genomes) - ELITES
    mums = elites[rng.choice(ELITES, n, p=weights)]
    dads = elites[rng.choice(ELITES, n, p=weights)]
    crossed = (rng.random(n) < CROSSOVER_RATE)[:, None] & (rng.random(mums.shape) < CROSSOVER_GENE_ODDS)
    sigmas = rng.choice(MUTATION_SIGMAS, n)[:, None]
    children = np.where(crossed, dads, mums) + sigmas * rng.standard_normal(mums.shape)
    return np.vstack([elites, children])


class Evolver:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.genomes = random_genomes(self.rng, POP_SIZE)
        self.history: list[GenStats] = []

    @property
    def gen(self) -> int:
        return len(self.history)

    def step(self) -> GenStats:
        seats = deal(self.rng, len(self.genomes), DEALS_PER_GEN)
        scenarios = [make_scenario(int(s)) for s in self.rng.integers(0, 2 ** 31, len(seats))]
        fitness, stats = evaluate(self.genomes, seats, scenarios, self.gen + 1)
        self.genomes = reproduce(self.rng, self.genomes, fitness)
        self.history.append(stats)
        return stats

    def save(self, swarm: Path, weights: Path) -> None:
        rows = np.array([astuple(s) for s in self.history]).reshape(-1, len(fields(GenStats)))
        np.savez(swarm, population=self.genomes, history=rows)
        save_champion(weights, self.genomes[0])

    def load(self, swarm: Path) -> None:
        self.genomes = load_roster(swarm)
        with np.load(swarm) as f:
            rows = f["history"]
        self.history = [GenStats(int(r[0]), *map(float, r[1:])) for r in rows]


def main() -> None:
    """Headless evolution: `just evolve <gens>`. Resumes from swarm.npz if present."""
    gens = int(sys.argv[1])
    ev = Evolver(int(time.time()))
    if SWARM.exists():
        ev.load(SWARM)
        print(f"Resuming from {SWARM} at generation {ev.gen}.")
    t0 = time.perf_counter()
    for _ in range(gens):
        s = ev.step()
        print(f"gen {s.gen:5d}  best {s.best:6.2f} ({s.best_kills:4.2f} kills)  mean {s.mean:6.2f}  "
              f"per tank: kills {s.kills:4.2f} friendly {s.friendly_kills:4.2f} near {s.near_misses:4.2f} "
              f"shots {s.shots:4.1f} alive {s.survived:4.1f}s  decided {s.decided:4.0%}  "
              f"{time.perf_counter() - t0:6.1f}s", flush=True)
    ev.save(SWARM, WEIGHTS)
    print(f"saved {SWARM} and {WEIGHTS}")


if __name__ == "__main__":
    main()

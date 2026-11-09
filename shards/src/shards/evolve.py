"""Neuroevolution. Each generation the whole population flies the same freshly seeded
scenarios; the top nets survive unchanged and the rest are mutated (sometimes crossed) copies."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from .config import (CROSSOVER_GENE_ODDS, CROSSOVER_RATE, ELITES, MUTATION_SIGMAS, POP_SIZE, RUN_STEPS, SAVE_EVERY,
                     SCENARIOS_PER_GEN)
from .pilots import Brains, Pilot, random_genomes
from .scenario import Scenario, make_scenario
from .worlds import Worlds

SWARM = Path("swarm.npz")       # the whole population plus the chart history
WEIGHTS = Path("weights.npz")   # the leader alone; on load it takes the swarm's top spot


@dataclass(frozen=True)
class GenStats:
    gen: int
    best: float
    mean: float
    best_hits: float
    best_seconds: float
    best_accuracy: float
    death_rate: float


@dataclass(frozen=True)
class LoadReport:
    swarm_found: bool
    leader_found: bool
    gen: int

    def describe(self) -> str:
        swarm = f"swarm at generation {self.gen}" if self.swarm_found else f"no {SWARM}, so a random swarm"
        leader = f"leader from {WEIGHTS}" if self.leader_found else f"no {WEIGHTS}, so the swarm's own leader"
        return f"Loaded {swarm}; {leader}."


def accuracy(hits: np.ndarray, fired: np.ndarray) -> np.ndarray:
    return hits / np.maximum(fired, 1)


def fly(worlds: Worlds, pilot: Pilot) -> Worlds:
    while not worlds.done:
        worlds.step(pilot.act(worlds.observe()))
    return worlds


def evaluate(genomes: np.ndarray, scenarios: list[Scenario], gen: int) -> tuple[np.ndarray, GenStats]:
    """Worlds [s*P, (s+1)*P) are the population on scenario s."""
    p, s = len(genomes), len(scenarios)
    worlds = fly(Worlds(scenarios, np.repeat(np.arange(s), p), RUN_STEPS), Brains(np.tile(genomes, (s, 1))))

    def per_net(values: np.ndarray) -> np.ndarray:
        return values.reshape(s, p).mean(0)

    fitness, hits, fired = per_net(worlds.fitness()), per_net(worlds.hits), per_net(worlds.fired)
    best = int(np.argmax(fitness))
    stats = GenStats(gen, float(fitness[best]), float(fitness.mean()), float(hits[best]),
                     float(per_net(worlds.survived)[best]), float(accuracy(hits, fired)[best]),
                     float((~worlds.alive).mean()))
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
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.genomes = random_genomes(self.rng, POP_SIZE)
            self.history: list[GenStats] = []

    @property
    def gen(self) -> int:
        return len(self.history)

    def champion(self) -> Brains:
        with self.lock:
            return Brains(self.genomes[:1].copy())

    def step(self) -> GenStats:
        with self.lock:
            genomes, gen = self.genomes.copy(), self.gen + 1
        scenarios = [make_scenario(int(s)) for s in self.rng.integers(0, 2 ** 31, SCENARIOS_PER_GEN)]
        fitness, stats = evaluate(genomes, scenarios, gen)
        nxt = reproduce(self.rng, genomes, fitness)
        with self.lock:
            self.genomes = nxt
            self.history.append(stats)
        return stats

    def save(self, swarm: Path, weights: Path) -> None:
        with self.lock:
            genomes, history = self.genomes.copy(), list(self.history)
        rows = np.array([astuple(s) for s in history]).reshape(-1, len(fields(GenStats)))
        np.savez(swarm, population=genomes, history=rows)
        Brains(genomes[:1]).save(weights)

    def load(self, swarm: Path, weights: Path) -> LoadReport:
        """The swarm from `swarm` (random, with no history, if it's missing), then the leader
        from `weights` written over the top spot (left alone if that file is missing)."""
        genomes, history = (self._read_swarm(swarm) if swarm.exists()
                            else (random_genomes(self.rng, POP_SIZE), []))
        if weights.exists():
            genomes[0] = Brains.load(weights).genomes[0]
        with self.lock:
            self.genomes = genomes
            self.history = history
        return LoadReport(swarm.exists(), weights.exists(), len(history))

    def _read_swarm(self, path: Path) -> tuple[np.ndarray, list[GenStats]]:
        with np.load(path) as f:
            genomes, rows = f["population"].copy(), f["history"]
        if genomes.shape[1:] != self.genomes.shape[1:]:
            raise ValueError(f"{path} holds {genomes.shape[1]}-weight nets; this build uses {self.genomes.shape[1]}")
        return genomes, [GenStats(int(r[0]), *map(float, r[1:])) for r in rows]


class BackgroundEvolver(threading.Thread):
    """Runs evolver.step() off the UI thread. start_run(n) evolves at most n generations,
    halt() stops early, shutdown() ends the thread."""

    def __init__(self, evolver: Evolver) -> None:
        super().__init__(daemon=True)
        self.evolver = evolver
        self._go = threading.Event()
        self._quit = threading.Event()
        self.remaining = 0

    @property
    def running(self) -> bool:
        return self._go.is_set()

    def start_run(self, gens: int) -> None:
        self.remaining = gens
        self._go.set()

    def halt(self) -> None:
        self._go.clear()

    def shutdown(self) -> None:
        self.halt()
        self._quit.set()

    def run(self) -> None:
        while not self._quit.is_set():
            if not self._go.wait(timeout=0.1):
                continue
            if self.remaining <= 0:
                self._go.clear()
                continue
            self.evolver.step()
            self.remaining -= 1


def main() -> None:
    """Headless evolution: `just evolve [gens]`. Resumes from swarm.npz / weights.npz if present,
    and saves them every SAVE_EVERY generations and at the end."""
    gens = int(sys.argv[1])
    ev = Evolver(int(time.time()))
    if SWARM.exists() or WEIGHTS.exists():
        print(ev.load(SWARM, WEIGHTS).describe())
    t0 = time.perf_counter()
    for _ in range(gens):
        s = ev.step()
        print(f"gen {s.gen:5d}  best {s.best:6.2f} ({s.best_hits:5.1f} hits, {s.best_seconds:4.1f}s, "
              f"acc {s.best_accuracy:4.0%})  mean {s.mean:6.2f}  died {s.death_rate:4.0%}  "
              f"{time.perf_counter() - t0:6.1f}s", flush=True)
        if s.gen % SAVE_EVERY == 0:
            ev.save(SWARM, WEIGHTS)
            print(f"saved {SWARM} and {WEIGHTS}", flush=True)
    ev.save(SWARM, WEIGHTS)
    print(f"saved {SWARM} and {WEIGHTS}")


if __name__ == "__main__":
    main()

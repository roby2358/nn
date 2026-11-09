"""Neuroevolution with one net per tank. Each generation the population is dealt at random into
battles, a few times over; the top nets by mean fitness survive unchanged and the rest are
mutated (sometimes crossed) copies of them."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from .config import (CROSSOVER_GENE_ODDS, CROSSOVER_RATE, DEALS_PER_GEN, ELITES, MUTATION_SIGMAS, POP_SIZE, RUN_STEPS,
                     TANKS)
from .pilots import Brains, load_roster, random_genomes, save_champion, seat
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


@dataclass(frozen=True)
class LoadReport:
    swarm_found: bool
    leader_found: bool
    gen: int

    def describe(self) -> str:
        if self.leader_found and not self.swarm_found:
            return f"No {SWARM}, so every tank gets the leader from {WEIGHTS}."
        swarm = f"swarm at generation {self.gen}" if self.swarm_found else f"no {SWARM}, so a random swarm"
        leader = f"leader from {WEIGHTS}" if self.leader_found else f"no {WEIGHTS}, so the swarm's own leader"
        return f"Loaded {swarm}; {leader}."


class Evolver:
    """The population and its log. Safe to step on one thread while another reads, resets or loads:
    the arrays are replaced and never changed in place, and a generation that was under way
    when the population got replaced is thrown away."""

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

    def leaders(self) -> np.ndarray:
        """(TANKS, GENOME): enough of the top of the population to fill a battle, best first."""
        with self.lock:
            return seat(self.genomes, TANKS)

    def step(self) -> GenStats:
        with self.lock:
            genomes, history = self.genomes, self.history
            seats = deal(self.rng, len(genomes), DEALS_PER_GEN)
            scenarios = [make_scenario(int(s)) for s in self.rng.integers(0, 2 ** 31, len(seats))]
        fitness, stats = evaluate(genomes, seats, scenarios, len(history) + 1)
        with self.lock:
            if self.history is history:
                self.genomes = reproduce(self.rng, genomes, fitness)
                self.history.append(stats)
        return stats

    def save(self, swarm: Path, weights: Path) -> None:
        with self.lock:
            genomes, history = self.genomes, list(self.history)
        rows = np.array([astuple(s) for s in history]).reshape(-1, len(fields(GenStats)))
        np.savez(swarm, population=genomes, history=rows)
        save_champion(weights, genomes[0])

    def load(self, swarm: Path, weights: Path) -> LoadReport:
        """Replace the population with what these files hold; see _starting_population."""
        genomes, history = self._starting_population(swarm, weights)
        with self.lock:
            self.genomes = genomes
            self.history = history
        return LoadReport(swarm.exists(), weights.exists(), len(history))

    def _starting_population(self, swarm: Path, weights: Path) -> tuple[np.ndarray, list[GenStats]]:
        """With a swarm, the swarm and its history, the `weights` leader (if any) on the top spot.
        With only a leader, the whole population is that leader. With neither, a random population."""
        if not swarm.exists():
            genomes = seat(load_roster(weights), POP_SIZE) if weights.exists() else random_genomes(self.rng, POP_SIZE)
            return genomes, []
        genomes, history = self._read_swarm(swarm)
        if weights.exists():
            genomes[0] = load_roster(weights)[0]
        return genomes, history

    def _read_swarm(self, path: Path) -> tuple[np.ndarray, list[GenStats]]:
        genomes = load_roster(path)
        with np.load(path) as f:
            rows = f["history"]
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
    """Headless evolution: `just evolve <gens>`. Resumes from swarm.npz / weights.npz if present."""
    gens = int(sys.argv[1])
    ev = Evolver(int(time.time()))
    if SWARM.exists() or WEIGHTS.exists():
        print(ev.load(SWARM, WEIGHTS).describe())
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

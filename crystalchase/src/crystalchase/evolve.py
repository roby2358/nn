"""Neuroevolution. Each generation the whole population (plus the scripted bot, as a
yardstick) flies the same freshly seeded courses solo; the top nets survive unchanged and
the rest are replaced by mutated (sometimes crossed) copies of them."""

from __future__ import annotations

import threading
import time
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np

from .config import COURSES_PER_GEN, CROSSOVER_RATE, ELITES, MUTATION_SIGMAS, POP_SIZE
from .course import Course, make_course
from .flock import Flock, solo_rule
from .pilots import AutoPilot, Brains, Crew, Seat, random_genomes

SWARM = Path("swarm.npz")       # the whole population plus the chart history
WEIGHTS = Path("weights.npz")   # the leader alone; on load it takes the swarm's top spot


@dataclass(frozen=True)
class GenStats:
    gen: int
    best: float
    mean: float
    bot: float
    crash_rate: float
    best_crystals: float


@dataclass(frozen=True)
class LoadReport:
    swarm_found: bool
    leader_found: bool
    gen: int

    def describe(self) -> str:
        swarm = f"swarm at generation {self.gen}" if self.swarm_found else f"no {SWARM}, so a random swarm"
        leader = f"leader from {WEIGHTS}" if self.leader_found else f"no {WEIGHTS}, so the swarm's own leader"
        return f"Loaded {swarm}; {leader}."


@dataclass(frozen=True)
class Generation:
    """One evaluated generation: enough to replay it exactly in the GUI."""
    stats: GenStats
    courses: list[Course]
    genomes: np.ndarray      # (P, GENOME), as evaluated
    fitness: np.ndarray      # (P,) mean over courses


def population_flight(genomes: np.ndarray, courses: list[Course]) -> tuple[Flock, Crew]:
    """Ships [c*P, (c+1)*P) are the population on course c; the last C ships are the
    bot, one per course."""
    p, c = len(genomes), len(courses)
    course_of = np.concatenate([np.repeat(np.arange(c), p), np.arange(c)])
    crew = Crew([Seat(Brains(np.tile(genomes, (c, 1))), slice(0, c * p)),
                 Seat(AutoPilot(), slice(c * p, c * p + c))])
    return Flock(courses, course_of, solo_rule), crew


def fly(flock: Flock, crew: Crew) -> Flock:
    while not flock.done:
        flock.step(crew.act(flock.observe()))
    return flock


def evaluate(genomes: np.ndarray, courses: list[Course]) -> tuple[np.ndarray, GenStats]:
    p, c = len(genomes), len(courses)
    flock = fly(*population_flight(genomes, courses))
    fit = flock.fitness()
    pop_fit = fit[:c * p].reshape(c, p).mean(0)
    best = int(np.argmax(pop_fit))
    crystals = flock.crystals[:c * p].reshape(c, p).mean(0)
    stats = GenStats(0, float(pop_fit[best]), float(pop_fit.mean()), float(fit[c * p:].mean()),
                     float((~flock.alive[:c * p]).mean()), float(crystals[best]))
    return pop_fit, stats


def reproduce(rng: np.random.Generator, genomes: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """Elites first (best at row 0), then children of elites: a rank-weighted parent,
    sometimes uniformly crossed with a second elite, plus Gaussian noise at one of a few
    scales, so some children fine-tune and some leap."""
    elites = genomes[np.argsort(-fitness)[:ELITES]]
    weights = 1.0 / np.arange(1, ELITES + 1)
    weights /= weights.sum()
    n = len(genomes) - ELITES
    mums = elites[rng.choice(ELITES, n, p=weights)]
    dads = elites[rng.choice(ELITES, n, p=weights)]
    crossed = (rng.random(n) < CROSSOVER_RATE)[:, None] & (rng.random(mums.shape) < 0.5)
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
            self.latest: Generation | None = None

    @property
    def gen(self) -> int:
        return len(self.history)

    def champion(self) -> Brains:
        with self.lock:
            return Brains(self.genomes[:1].copy())

    def step(self) -> GenStats:
        with self.lock:
            genomes, gen = self.genomes.copy(), self.gen + 1
        courses = [make_course(int(s)) for s in self.rng.integers(0, 2 ** 31, COURSES_PER_GEN)]
        fitness, stats = evaluate(genomes, courses)
        stats = GenStats(gen, *astuple(stats)[1:])
        nxt = reproduce(self.rng, genomes, fitness)
        with self.lock:
            self.genomes = nxt
            self.history.append(stats)
            self.latest = Generation(stats, courses, genomes, fitness)
        return stats

    def save(self, swarm: Path, weights: Path) -> None:
        with self.lock:
            genomes, history = self.genomes.copy(), list(self.history)
        np.savez(swarm, population=genomes, history=np.array([astuple(s) for s in history]).reshape(-1, 6))
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
            self.latest = None
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
    """Headless evolution: `just evolve [gens]`. Resumes from swarm.npz / weights.npz if present."""
    import sys

    gens = int(sys.argv[1])
    ev = Evolver(int(time.time()))
    if SWARM.exists() or WEIGHTS.exists():
        print(ev.load(SWARM, WEIGHTS).describe())
    t0 = time.perf_counter()
    for _ in range(gens):
        s = ev.step()
        print(f"gen {s.gen:5d}  best {s.best:6.2f} ({s.best_crystals:4.1f} crystals)  mean {s.mean:6.2f}  "
              f"bot {s.bot:6.2f}  crashed {s.crash_rate:4.0%}  {time.perf_counter() - t0:6.1f}s", flush=True)
    ev.save(SWARM, WEIGHTS)
    print(f"saved {SWARM} and {WEIGHTS}")


if __name__ == "__main__":
    main()

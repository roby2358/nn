"""Did the saved swarm descend from a reference net? `just cc-lineage <n> <reference.npz>`.
Compares swarm.npz and weights.npz to the reference weight by weight, then flies them all,
with the autopilot, on n courses that evolution's random seeds are unlikely to have used."""

import sys

import numpy as np

from .config import ELITES
from .course import make_course
from .evolve import SWARM, WEIGHTS, fly, population_flight
from .pilots import Brains

FIRST_COURSE = 50_000
SHOW_SLOTS = 8


def weight_change(genomes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """(N, 3): for each net, how many weights differ, their RMS change, their largest change."""
    delta = genomes - reference
    return np.column_stack([(delta != 0).sum(1), np.sqrt((delta ** 2).mean(1)), np.abs(delta).max(1)])


def flown(genomes: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(mean score per net, crash share per net, the autopilot's mean score)."""
    courses = [make_course(FIRST_COURSE + i) for i in range(n)]
    flock = fly(*population_flight(genomes, courses))
    p, fit = len(genomes), flock.fitness()
    crashed = ~flock.alive[:n * p].reshape(n, p)
    return fit[:n * p].reshape(n, p).mean(0), crashed.mean(0), float(fit[n * p:].mean())


def print_history(history: np.ndarray) -> None:
    print(f"swarm history: {len(history)} generations")
    if len(history) == 0:
        return
    for label, rows in (("first 50", history[:50]), ("last 50", history[-50:])):
        best, mean, bot, crashed = rows[:, 1:5].mean(0)
        print(f"  {label:9s} best {best:6.2f}  mean {mean:6.2f}  autopilot {bot:6.2f}  crashed {crashed:4.0%}")


def row(label: str, score: float, crashed: float, change: np.ndarray) -> str:
    return (f"  {label:12s} score {score:6.2f}  crashed {crashed:5.1%}  changed {change[0]:3.0f} weights  "
            f"rms {change[1]:8.4f}  max {change[2]:8.4f}")


def main() -> None:
    n, reference_path = int(sys.argv[1]), sys.argv[2]
    reference = Brains.load(reference_path).genomes[0]
    leader = Brains.load(WEIGHTS).genomes[0]
    with np.load(SWARM) as f:
        population, history = f["population"].copy(), f["history"].copy()
    print_history(history)
    nets = np.vstack([reference, leader, population])
    scores, crashes, bot = flown(nets, n)
    changes = weight_change(nets, reference)
    print(f"on {n} unseen courses (autopilot {bot:.2f}); changes are measured from {reference_path}:")
    print(row("reference", scores[0], crashes[0], changes[0]))
    print(row(str(WEIGHTS), scores[1], crashes[1], changes[1]))
    for slot in range(SHOW_SLOTS):
        print(row(f"slot {slot}" + (" (elite)" if slot < ELITES else ""), scores[2 + slot], crashes[2 + slot], changes[2 + slot]))
    rms = changes[2:, 1]
    print(f"swarm: mean score {scores[2:].mean():.2f}, best {scores[2:].max():.2f} (slot {int(np.argmax(scores[2:]))}), "
          f"beating reference {int((scores[2:] > scores[0]).sum())}/{len(population)}; "
          f"weight rms from reference: min {rms.min():.4f}, median {np.median(rms):.4f}, max {rms.max():.4f}")


if __name__ == "__main__":
    main()

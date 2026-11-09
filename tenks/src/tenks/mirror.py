"""A weights or swarm file against itself: `just cc-mirror <file.npz> <n>`, for how a net's battles play out."""

import sys
from pathlib import Path

import numpy as np

from .config import RUN_STEPS, TANKS
from .evolve import fight
from .pilots import Brains, load_roster, seat
from .scenario import make_scenario
from .stats import report
from .worlds import BLUE, GREEN, Worlds


def battle_mirror(roster: np.ndarray, n: int) -> Worlds:
    genomes = np.tile(seat(roster, TANKS), (n, 1))
    return fight(Worlds([make_scenario(s) for s in range(n)], RUN_STEPS), Brains(genomes))


def main() -> None:
    path, n = Path(sys.argv[1]), int(sys.argv[2])
    w = battle_mirror(load_roster(path), n)
    print(f"{path}: {n} battles against itself")
    last_death = (w.survived * ~w.alive).max(1)
    print(f"  a team wiped out {w.over.mean():4.0%}   tanks left in all {w.alive.sum(1).mean():4.1f} of {TANKS}   "
          f"last death at {last_death.mean():4.1f} s (from {last_death.min():4.1f} to {last_death.max():4.1f})")
    report("green", w, GREEN)
    report("blue", w, BLUE)


if __name__ == "__main__":
    main()

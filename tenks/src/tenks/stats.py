"""How a swarm or weights file fights random nets: `just cc-stats <file.npz> <n>`. The file's
nets take the green seats on n fixed battles and fresh random nets take the blue ones, so
unlike evolution's own numbers this yardstick doesn't move as the population improves."""

import sys
from pathlib import Path

import numpy as np

from .config import RUN_STEPS, TEAM_SIZE
from .evolve import fight
from .pilots import Brains, load_roster, random_genomes, seat
from .scenario import make_scenario
from .worlds import BLUE, GREEN, Worlds


def battle_against_random(roster: np.ndarray, n: int) -> Worlds:
    rng = np.random.default_rng(0)
    green = seat(roster, TEAM_SIZE)
    genomes = np.vstack([np.vstack([green, random_genomes(rng, TEAM_SIZE)]) for _ in range(n)])
    return fight(Worlds([make_scenario(s) for s in range(n)], RUN_STEPS), Brains(genomes))


def report(name: str, worlds: Worlds, side: np.ndarray) -> None:
    kills, shots = worlds.kills[:, side], worlds.fired[:, side]
    print(f"  {name:5s}  alive at the end {worlds.alive[:, side].sum(1).mean():4.1f} of {TEAM_SIZE}   "
          f"per tank: kills {kills.mean():4.2f}  friendly {worlds.friendly_kills[:, side].mean():4.2f}  "
          f"near {worlds.near_misses[:, side].mean():4.2f}  shots {shots.mean():4.1f}  "
          f"kills per shot {kills.sum() / max(shots.sum(), 1):4.0%}")


def main() -> None:
    path, n = Path(sys.argv[1]), int(sys.argv[2])
    w = battle_against_random(load_roster(path), n)
    margin = w.alive[:, GREEN].sum(1) - w.alive[:, BLUE].sum(1)
    print(f"{path}: {n} battles against random nets")
    print(f"  green wins {(margin > 0).mean():4.0%}   blue wins {(margin < 0).mean():4.0%}   "
          f"level {(margin == 0).mean():4.0%}   a team wiped out {w.over.mean():4.0%}")
    report("green", w, GREEN)
    report("blue", w, BLUE)


if __name__ == "__main__":
    main()

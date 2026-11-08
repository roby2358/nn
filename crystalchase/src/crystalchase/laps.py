"""Fly the saved champion solo on N fresh courses and report how each lap ended:
`just cc-laps N`. Shows crystals, score and, for crashes, when and where."""

import sys

import numpy as np

from .config import FIELD_H, FIELD_W
from .course import make_course
from .evolve import WEIGHTS
from .flock import Flock, solo_rule
from .pilots import Brains


def main() -> None:
    n = int(sys.argv[1])
    champ = Brains.load(WEIGHTS)
    courses = [make_course(seed) for seed in range(n)]
    flock = Flock(courses, np.arange(n), solo_rule)
    crashed_at = np.full(n, np.nan)
    while not flock.done:
        flock.step(champ.act(flock.observe()))
        for e in flock.events:
            if e.kind == "crash":
                crashed_at[e.ship] = flock.seconds
    for k in range(n):
        x, y = flock.pos[k]
        end = (f"crashed at {crashed_at[k]:4.1f}s near ({x:5.0f},{y:5.0f})" if not flock.alive[k]
               else "flew the full 30s")
        print(f"course {k:3d}  {flock.crystals[k]:2d} crystals  score {flock.fitness()[k]:5.2f}  {end}")
    lost = ~flock.alive
    print(f"\n{lost.sum()}/{n} laps crashed; mean crash time {np.nanmean(crashed_at):.1f}s" if lost.any()
          else f"\nno crashes in {n} laps")
    print(f"field is {FIELD_W:.0f}x{FIELD_H:.0f}; mean crystals {flock.crystals.mean():.1f}")


if __name__ == "__main__":
    main()

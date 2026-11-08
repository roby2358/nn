"""Choose sector_brain's five design constants by flying them: `just cc-sector-search`.
No training, only evaluation: a coarse grid, the leaders on more courses, the winner on
courses the search never saw, next to the autopilot."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from itertools import product  # noqa: E402

import numpy as np  # noqa: E402

from .calculated import sector_brain  # noqa: E402
from .course import Course, make_course  # noqa: E402
from .evolve import fly, population_flight  # noqa: E402

ARRIVAL_SPEEDS = (40.0, 60.0, 80.0, 100.0, 120.0)
RANGE_GAINS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.2)
DEADBANDS = (0.06, 0.1, 0.15, 0.22)
AIM_TOLERANCES = (0.5, 0.7, 0.9, 1.1, 1.4)
MIN_ERRORS = (1.0, 3.0, 5.0, 10.0)

WORKERS = 20
CHUNK = 128
SCREEN = (10_000, 100)     # (first course seed, course count)
REFINE = (20_000, 400)
HOLDOUT = (30_000, 1000)
LEADERS = 30

Design = tuple[float, float, float, float, float]


def courses(span: tuple[int, int]) -> list[Course]:
    return [make_course(span[0] + i) for i in range(span[1])]


def score_chunk(job: tuple[list[Design], tuple[int, int]]) -> np.ndarray:
    """(D, 2) mean score and crash count for each design on the span's courses."""
    designs, span = job
    genomes = np.vstack([sector_brain(*d).genomes for d in designs])
    c, d = span[1], len(designs)
    flock = fly(*population_flight(genomes, courses(span)))
    fit = flock.fitness()[:c * d].reshape(c, d).mean(0)
    crashes = (~flock.alive[:c * d]).reshape(c, d).sum(0)
    return np.column_stack([fit, crashes])


def score_all(designs: list[Design], span: tuple[int, int]) -> np.ndarray:
    jobs = [(designs[i:i + CHUNK], span) for i in range(0, len(designs), CHUNK)]
    with ProcessPoolExecutor(WORKERS) as pool:
        return np.vstack(list(pool.map(score_chunk, jobs)))


def bot_on(span: tuple[int, int]) -> tuple[float, int]:
    """The autopilot's mean score and crash count; one throwaway design rides along."""
    design = (140.0, 0.45, 0.06, 0.35, 15.0)
    flock = fly(*population_flight(sector_brain(*design).genomes, courses(span)))
    c = span[1]
    return float(flock.fitness()[c:].mean()), int((~flock.alive[c:]).sum())


def main() -> None:
    grid = list(product(ARRIVAL_SPEEDS, RANGE_GAINS, DEADBANDS, AIM_TOLERANCES, MIN_ERRORS))
    screened = score_all(grid, SCREEN)
    leaders = [grid[i] for i in np.argsort(-screened[:, 0])[:LEADERS]]
    print(f"screened {len(grid)} designs on {SCREEN[1]} courses; best {screened[:, 0].max():.2f}", flush=True)
    refined = score_all(leaders, REFINE)
    order = np.argsort(-refined[:, 0])
    print(f"on {REFINE[1]} courses (bot {bot_on(REFINE)[0]:.2f}):")
    for i in order[:8]:
        print(f"  {refined[i, 0]:6.2f}  crashes {refined[i, 1]:3.0f}  {leaders[i]}")
    winner = leaders[order[0]]
    final = score_all([winner], HOLDOUT)[0]
    bot_score, bot_crashes = bot_on(HOLDOUT)
    print(f"winner {winner}\nheld-out {HOLDOUT[1]} courses: net {final[0]:.3f} ({final[1]:.0f} crashes), "
          f"bot {bot_score:.3f} ({bot_crashes} crashes)")


if __name__ == "__main__":
    main()

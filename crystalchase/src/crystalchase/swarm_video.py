"""Film one 30-second training course as an MP4: `just cc-swarm-video <seed> <out.mp4>`.
The saved champion flies in slot 0 among random nets, the autopilot alongside, in the same
replay view that evolution shows, cropped to the flight field. The seed picks the course
and the random nets."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from .config import POP_SIZE, RUN_STEPS  # noqa: E402
from .course import make_course  # noqa: E402
from .game import FIELD, App  # noqa: E402
from .pilots import random_genomes  # noqa: E402
from .scenes import Replay  # noqa: E402
from .video import record  # noqa: E402


def mixed_swarm(champion: np.ndarray, seed: int) -> np.ndarray:
    """(POP_SIZE, GENOME): the champion in slot 0, random nets in every other slot."""
    return np.vstack([champion, random_genomes(np.random.default_rng(seed), POP_SIZE - 1)])


def main() -> None:
    seed, out = int(sys.argv[1]), sys.argv[2]
    app = App()
    genomes = mixed_swarm(app.evolver.champion().genomes, seed)
    scene = Replay("random swarm + calculated leader", make_course(seed), genomes, app.rng)
    app.scene = scene
    led =np.zeros(RUN_STEPS, dtype=bool)

    def advance(frame: int) -> None:
        scene.update(1)
        led[frame - 1] = scene.leader == 0

    record(app, pygame.Rect((0, 0), FIELD), RUN_STEPS, advance, out)
    app.bg.shutdown()
    f, fit = scene.flock, scene.flock.fitness()
    print(f"course {seed}: calculated {fit[0]:.2f} ({f.crystals[0]} crystals), best random {fit[1:POP_SIZE].max():.2f}, "
          f"autopilot {fit[POP_SIZE]:.2f}; random nets alive {int(f.alive[1:POP_SIZE].sum())}/{POP_SIZE - 1}; "
          f"calculated net highlighted {led.mean():.1%} of frames")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

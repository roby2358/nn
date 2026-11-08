"""Film the whole window as a looping MP4: `just cc-loop-video <seeds> <out.mp4>`.
The champion flies a ring of crystals; the film runs from one capture of the first crystal
to its next capture, ten captures in all. One unfilmed lap of the ring comes first, so both
captures are approached from the same crystal. Tries `seeds` rings and films the one whose
two captures match best. Uses the saved swarm.npz and weights.npz."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from .art import Starfield  # noqa: E402
from .config import DT, SHIP_R  # noqa: E402
from .course import make_ring_course  # noqa: E402
from .game import App  # noqa: E402
from .pilots import Pilot  # noqa: E402
from .scenes import FreeFlight  # noqa: E402
from .video import record  # noqa: E402

CAPTURES = 10
SPOTS = CAPTURES - 1          # the tenth capture is the first crystal again
WARM_LAPS = 1
START_IDX = WARM_LAPS * SPOTS + 1
END_IDX = START_IDX + SPOTS
MAX_STEPS = 7200
STAR_FADE = 2.0               # seconds over which the stars cross over to their opening frame
SEAM_LOOKAHEAD = 0.25         # seconds of velocity difference counted in the seam size
PARTICLE_SEED = 1


@dataclass(frozen=True)
class Take:
    seed: int
    start_step: int           # the step that makes the opening capture
    end_step: int             # the step that makes the closing capture
    seam_px: float


class LoopingStars:
    """A Starfield whose drawing at t=period equals its drawing at t=0: over the last
    `fade` seconds it crossfades to the same stars one period earlier."""

    def __init__(self, stars: Starfield, period: float, fade: float, size: tuple[int, int]) -> None:
        self.stars = stars
        self.period = period
        self.fade = fade
        self.earlier = pygame.Surface(size)

    def draw(self, screen: pygame.Surface, t: float) -> None:
        self.stars.draw(screen, t)
        blend = (t - (self.period - self.fade)) / self.fade
        if blend <= 0.0:
            return
        self.stars.draw(self.earlier, t - self.period)
        self.earlier.set_alpha(round(255 * min(blend, 1.0)))
        screen.blit(self.earlier, (0, 0))


def pose(scene: FreeFlight) -> tuple[np.ndarray, np.ndarray, float]:
    f = scene.flock
    return f.pos[0].copy(), f.vel[0].copy(), float(f.ang[0])


def seam_px(a: tuple[np.ndarray, np.ndarray, float], b: tuple[np.ndarray, np.ndarray, float]) -> float:
    """How far apart the ship looks at the two captures, in pixels."""
    turn = abs((a[2] - b[2] + math.pi) % math.tau - math.pi)
    return float(np.hypot(*(a[0] - b[0])) + SHIP_R * turn + SEAM_LOOKAHEAD * np.hypot(*(a[1] - b[1])))


def ring_scene(seed: int, title: str, pilot: Pilot, rng: np.random.Generator) -> FreeFlight:
    return FreeFlight(title, make_ring_course(seed, SPOTS), pilot, rng)


def rehearse(seed: int, pilot: Pilot) -> Take | None:
    """Fly without drawing. None if the ship crashes or never finishes."""
    scene = ring_scene(seed, "rehearsal", pilot, np.random.default_rng(PARTICLE_SEED))
    marks: dict[int, tuple[int, tuple]] = {}
    for step in range(1, MAX_STEPS + 1):
        scene.update(1)
        if not scene.flock.alive[0]:
            return None
        idx = int(scene.flock.idx[0])
        if idx in (START_IDX, END_IDX) and idx not in marks:
            marks[idx] = (step, pose(scene))
        if END_IDX in marks:
            return Take(seed, marks[START_IDX][0], marks[END_IDX][0], seam_px(marks[START_IDX][1], marks[END_IDX][1]))
    return None


def best_take(seeds: int, pilot: Pilot) -> Take:
    takes = [t for t in (rehearse(seed, pilot) for seed in range(seeds)) if t is not None]
    if not takes:
        raise ValueError(f"no clean flight on {seeds} rings")
    return min(takes, key=lambda t: t.seam_px)


def step_scene(scene: FreeFlight, burst_seeded: bool) -> None:
    """One step; a capture's burst restarts from one seed, so both captures burst alike."""
    if burst_seeded:
        scene.particles.rng = np.random.default_rng(PARTICLE_SEED)
    scene.update(1)


def film(app: App, take: Take, out: str) -> None:
    """Frames run from just after the opening capture through the closing capture, so
    the last frame leads straight into the first."""
    scene = ring_scene(take.seed, f"champion after gen {app.evolver.gen}", app.evolver.champion(), app.rng)
    app.scene = scene
    for step in range(1, take.start_step + 1):
        step_scene(scene, step == take.start_step)
    scene.restart_laps()
    frames = take.end_step - take.start_step
    app.stars = LoopingStars(app.stars, frames * DT, STAR_FADE, app.field.get_size())
    record(app, app.screen.get_rect(), frames, lambda frame: step_scene(scene, frame == frames), out)


def main() -> None:
    seeds, out = int(sys.argv[1]), sys.argv[2]
    app = App()
    take = best_take(seeds, app.evolver.champion())
    seconds = (take.end_step - take.start_step) * DT
    print(f"ring seed {take.seed}: {seconds:.2f} s, seam {take.seam_px:.2f} px", flush=True)
    film(app, take, out)
    app.bg.shutdown()
    print(f"saved {out}")


if __name__ == "__main__":
    main()

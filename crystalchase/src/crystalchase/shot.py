"""Headless screenshots for docs and visual checks: `just cc-shot`. Uses weights.npz if present.
Saves shot-watch.png (a generation replay, swarm shown as while evolving) and shot-race.png (a race, mid-flight)."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from .game import App  # noqa: E402
from .scenes import COUNTDOWN_FRAMES, Replay  # noqa: E402


def main() -> None:
    seconds = float(sys.argv[1])
    app = App()
    gen = app.evolver.step().gen
    latest = app.evolver.latest
    app.scene = Replay(f"generation {gen}", latest.courses[0], latest.genomes, app.rng)
    for _ in range(int(seconds * 60)):
        app.scene.update(1)  # stepping the scene directly: app.tick() would switch to free flight
    app.draw()
    pygame.image.save(app.screen, "shot-watch.png")
    app.toggle_race()
    for _ in range(COUNTDOWN_FRAMES + int(seconds * 60)):
        app.scene.update(1)
    app.draw()
    pygame.image.save(app.screen, "shot-race.png")
    app.bg.shutdown()
    print("saved shot-watch.png, shot-race.png")


if __name__ == "__main__":
    main()

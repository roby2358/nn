"""Headless screenshots for visual checks: `just cc-shot <seconds>`.
Saves shot-watch.png (the champion, N seconds in) and shot-fly.png (a held burn, to see the flame)."""

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from .game import App  # noqa: E402


def main() -> None:
    seconds = float(sys.argv[1])
    app = App()
    app.scene.update(int(seconds * 60))
    app.draw()
    pygame.image.save(app.screen, "shot-watch.png")
    app.toggle_fly()
    app.human.thrust = 1.0
    app.human.fire = 1.0
    app.scene.update(45)
    app.draw()
    pygame.image.save(app.screen, "shot-fly.png")
    app.bg.shutdown()
    print("saved shot-watch.png, shot-fly.png")


if __name__ == "__main__":
    main()

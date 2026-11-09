"""Headless screenshots of the whole window: `just cc-shot <seconds> <dir>`.
Saves shot-watch.png (the leaders, N seconds in) and shot-drive.png (your tank, a short way in)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from .config import CMD_LEFT, CMD_RIGHT, DT  # noqa: E402
from .game import App  # noqa: E402

DRIVE_STEPS = 45


def main() -> None:
    seconds, out = float(sys.argv[1]), Path(sys.argv[2])
    app = App()
    app.scene.update(round(seconds / DT))
    app.draw()
    pygame.image.save(app.screen, out / "shot-watch.png")
    app.toggle_drive()
    app.human.command[[CMD_LEFT, CMD_RIGHT]] = 1.0
    app.scene.update(DRIVE_STEPS)
    app.draw()
    pygame.image.save(app.screen, out / "shot-drive.png")
    app.bg.shutdown()
    print(f"saved shot-watch.png and shot-drive.png in {out}")


if __name__ == "__main__":
    main()

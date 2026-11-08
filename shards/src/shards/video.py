"""Record the playing field alone to an MP4: `just cc-video <weights.npz> <seed> <seconds> <out.mp4>`."""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from .config import DT, RUN_STEPS  # noqa: E402
from .game import FIELD, App, Round  # noqa: E402
from .pilots import Brains  # noqa: E402

FPS = round(1.0 / DT)      # one physics step per video frame, so it plays at real speed


def record(app: App, frames: int, out: Path) -> None:
    """Advance the app's round one step per frame and encode just the field."""
    video = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FIELD[0]}x{FIELD[1]}", "-r", str(FPS),
                              "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                              "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for frame in range(1, frames + 1):
        app.scene.update(1)
        app.compose_field(frame * DT)
        video.stdin.write(pygame.image.tobytes(app.field, "RGB"))
    video.stdin.close()
    if video.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main() -> None:
    weights, seed, seconds, out = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    app = App()
    app.scene = Round(weights.stem, Brains.load(weights), seed, np.random.default_rng(seed), RUN_STEPS)
    record(app, round(seconds * FPS), out)
    app.bg.shutdown()
    w = app.scene.world
    print(f"wrote {out}: score {w.score[0]:.0f}, wave {w.wave[0] + 1}, alive {bool(w.alive[0])}")


if __name__ == "__main__":
    main()

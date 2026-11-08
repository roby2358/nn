"""Record a region of the App window to an MP4, one physics step per frame."""

import math
import subprocess
from typing import Callable

import imageio_ffmpeg
import pygame

from .config import DT
from .game import App

FPS = 60


def record(app: App, region: pygame.Rect, frames: int, advance: Callable[[int], None], out: str) -> None:
    """For each frame 1..frames: advance(frame), then paint the window and encode the region."""
    app.gen_clock = (0, math.inf, 0.0)        # an idle evolver: no gens/sec reading to update
    video = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{region.w}x{region.h}", "-r", str(FPS), "-i", "-",
                              "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                              "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    picture = app.screen.subsurface(region)
    for frame in range(1, frames + 1):
        advance(frame)
        app.compose(frame * DT)
        video.stdin.write(pygame.image.tobytes(picture, "RGB"))
    video.stdin.close()
    if video.wait() != 0:
        raise RuntimeError("ffmpeg failed")

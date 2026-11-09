"""Record one battle to an MP4: `just cc-video <file.npz> <seed> <seconds> <out.mp4>`.

A swarm file seats its first 16 nets, and a weights file seats its champion in every tank."""

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from .art import Battlefield  # noqa: E402
from .config import DT, FIELD_H, FIELD_W, TANKS  # noqa: E402
from .pilots import Brains, load_roster, seat  # noqa: E402
from .scenario import make_scenario  # noqa: E402
from .worlds import BLUE, GREEN, Worlds  # noqa: E402

FPS = round(1.0 / DT)      # one physics step per video frame, so it plays at real speed
ENDING_FRAMES = 2 * FPS    # how long the video lingers once a team is wiped out


def record(worlds: Worlds, brains: Brains, label: str, seed: int, out: Path) -> None:
    """Run world 0 to its step limit, or a little past the end of the battle, encoding each step."""
    pygame.init()
    field = Battlefield(seed)
    video = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{int(FIELD_W)}x{int(FIELD_H)}",
                              "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    ending = ENDING_FRAMES
    while not worlds.done and ending > 0:
        worlds.step(brains.act(worlds.observe()))
        field.draw(worlds, label)
        video.stdin.write(pygame.image.tobytes(field.surf, "RGB"))
        ending -= int(worlds.over[0])
    video.stdin.close()
    if video.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main() -> None:
    weights, seed, seconds, out = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    genomes = np.random.default_rng(seed).permutation(seat(load_roster(weights), TANKS))
    worlds = Worlds([make_scenario(seed)], round(seconds / DT))
    record(worlds, Brains(genomes), weights.stem, seed, out)
    print(f"wrote {out}: {worlds.steps * DT:.1f}s, green {worlds.alive[0, GREEN].sum()} left, "
          f"blue {worlds.alive[0, BLUE].sum()} left, {worlds.fired.sum()} shots")


if __name__ == "__main__":
    main()

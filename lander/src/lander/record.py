"""Render the net's landings to an mp4: the scene only, no training panel, no display.

Usage: python -m lander.record COUNT OUT.mp4
"""
import random
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pygame

from lander.ai.policy import PolicyNet
from lander.ai.train import TRAIN, WEIGHTS, Trainer
from lander.game import FPS, RESULT_PAUSE_FRAMES, NetPilot, draw_scene, make_stars, scene_fonts
from lander.physics import CONFIG, DT, Config, Outcome, World, new_world, step

MAX_ATTEMPTS_PER_LANDING = 25


def fly(pilot: NetPilot, config: Config, rng: random.Random) -> list[World]:
    """One whole round as a list of per-frame worlds, flown exactly as Watch flies it."""
    pilot.reset()
    worlds = [new_world(config, rng)]
    while worlds[-1].outcome is Outcome.FLYING and len(worlds) <= pilot.frame_limit:
        worlds.append(step(worlds[-1], pilot.controls(worlds[-1], None), DT))
    return worlds


def collect_landings(pilot: NetPilot, config: Config, rng: random.Random, count: int) -> tuple[list[list[World]], int]:
    """The first `count` rounds that landed, and how many rounds it took to get them."""
    landed: list[list[World]] = []
    attempts = 0
    while len(landed) < count:
        if attempts >= count * MAX_ATTEMPTS_PER_LANDING:
            raise RuntimeError(f"only {len(landed)} landings in {attempts} rounds; is the net trained?")
        attempts += 1
        worlds = fly(pilot, config, rng)
        if worlds[-1].outcome is Outcome.LANDED:
            landed.append(worlds)
    return landed, attempts


def with_result_pause(flights: list[list[World]], pause_frames: int) -> Iterator[World]:
    """Every frame in order, lingering on each flight's final world."""
    for worlds in flights:
        yield from worlds
        yield from [worlds[-1]] * pause_frames


def render(worlds: Iterable[World], config: Config, pilot: NetPilot, rng: random.Random) -> Iterator[bytes]:
    surf = pygame.Surface((config.width, config.height))
    fonts = scene_fonts()
    stars = make_stars(config, rng)
    for world in worlds:
        draw_scene(surf, fonts, stars, world, pilot)
        yield pygame.image.tobytes(surf, "RGB")


def write_mp4(frames: Iterable[bytes], width: int, height: int, fps: int, out: Path) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as ffmpeg:
        for frame in frames:
            ffmpeg.stdin.write(frame)
        ffmpeg.stdin.close()
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with {ffmpeg.returncode}")


def main() -> None:
    count, out = int(sys.argv[1]), Path(sys.argv[2])
    pygame.font.init()
    net = PolicyNet(np.random.default_rng())
    net.load_from(WEIGHTS)
    pilot = NetPilot(Trainer(net, TRAIN, None))
    rng = random.Random()
    flights, attempts = collect_landings(pilot, CONFIG, rng, count)
    worlds = with_result_pause(flights, RESULT_PAUSE_FRAMES)
    write_mp4(render(worlds, CONFIG, pilot, rng), CONFIG.width, CONFIG.height, FPS, out)
    print(f"{out}: {count} landings from {attempts} rounds flown")


if __name__ == "__main__":
    main()

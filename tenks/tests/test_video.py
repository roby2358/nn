"""The video recipe writes a playable file of the battle, and stats can read its nets."""

import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from tenks.config import POP_SIZE, TANKS  # noqa: E402
from tenks.pilots import Brains, random_genomes  # noqa: E402
from tenks.scenario import make_scenario  # noqa: E402
from tenks.stats import battle_against_random  # noqa: E402
from tenks.video import record  # noqa: E402
from tenks.worlds import Worlds  # noqa: E402


def test_record_writes_a_video(tmp_path):
    genomes = random_genomes(np.random.default_rng(0), TANKS)
    worlds = Worlds([make_scenario(0)], 6)
    out = tmp_path / "clip.mp4"
    record(worlds, Brains(genomes), "test", 0, out)
    assert out.stat().st_size > 0
    assert worlds.steps == 6


def test_stats_runs_the_battles_it_was_asked_for():
    roster = random_genomes(np.random.default_rng(1), POP_SIZE)
    w = battle_against_random(roster, 2)
    assert w.alive.shape == (2, TANKS) and w.done

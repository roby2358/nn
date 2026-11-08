"""cc-stats reports how living ships use their controls."""

import numpy as np

from shards.config import N_OUTPUTS
from shards.pilots import Brains, random_genomes
from shards.scenario import make_scenario
from shards.stats import Recorder
from shards.worlds import Worlds


def test_recorder_ignores_dead_ships():
    worlds = Worlds([make_scenario(1)], np.zeros(3, dtype=np.int64))
    worlds.alive[1] = False
    recorder = Recorder(Brains(random_genomes(np.random.default_rng(0), 3)), worlds)
    commands = recorder.act(worlds.observe())
    assert commands.shape == (3, N_OUTPUTS)
    assert np.array_equal(recorder.commands[0], commands[[0, 2]])

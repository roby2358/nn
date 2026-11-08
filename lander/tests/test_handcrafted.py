"""Exercise the hardest placement and initial drift combinations."""
from dataclasses import replace
import random

import pytest

from lander.ai.handcrafted import build_policy
from lander.ai.train import TRAIN
from lander.physics import CONFIG, DT, Outcome, hold, new_world, random_terrain


@pytest.mark.parametrize("vx", [-30, 0, 30])
@pytest.mark.parametrize("x", [CONFIG.lander_size,
                              CONFIG.width - CONFIG.pad_width - CONFIG.lander_size])
@pytest.mark.parametrize("y", [CONFIG.height * 2 / 3,
                              CONFIG.ground_y - 2 * CONFIG.lander_size])
def test_lands_at_placement_extremes(vx, x, y):
    rng = random.Random(0)
    world = new_world(CONFIG, rng)
    pad = replace(world.pad, x=x, y=y)
    world = replace(world, pad=pad, lander=replace(world.lander, vx=vx),
                    terrain=random_terrain(CONFIG, pad, rng))
    net = build_policy()
    for _ in range(TRAIN.max_decisions):
        world = hold(world, net.greedy(world), DT, TRAIN.decide_every)
        if world.outcome is not Outcome.FLYING:
            break
    assert world.outcome is Outcome.LANDED

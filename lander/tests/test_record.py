import random

import numpy as np
import pytest

from lander.ai.handcrafted import build_policy
from lander.ai.policy import PolicyNet
from lander.ai.train import TRAIN, Trainer
from lander.game import NetPilot
from lander.physics import CONFIG, Outcome
from lander.record import collect_landings, fly, with_result_pause, write_mp4


def pilot_for(net: PolicyNet) -> NetPilot:
    return NetPilot(Trainer(net, TRAIN, 0))


def test_fly_stops_at_the_end_of_the_round():
    pilot = pilot_for(build_policy())
    worlds = fly(pilot, CONFIG, random.Random(1))
    assert worlds[0].outcome is Outcome.FLYING
    assert worlds[-1].outcome is not Outcome.FLYING
    assert all(w.outcome is Outcome.FLYING for w in worlds[:-1])


def test_collect_landings_returns_only_landings():
    flights, attempts = collect_landings(pilot_for(build_policy()), CONFIG, random.Random(2), 3)
    assert len(flights) == 3
    assert attempts >= 3
    assert all(worlds[-1].outcome is Outcome.LANDED for worlds in flights)


def test_collect_landings_gives_up_on_a_net_that_cannot_land():
    net = PolicyNet(np.random.default_rng(0))
    for param in net.params:
        param.fill(0)  # p = 0.5 is not over one half: engines off, free fall into the ground
    with pytest.raises(RuntimeError, match="only 0 landings"):
        collect_landings(pilot_for(net), CONFIG, random.Random(3), 1)


def test_result_pause_repeats_each_final_world():
    flights = [["a1", "a2"], ["b1"]]
    assert list(with_result_pause(flights, 2)) == ["a1", "a2", "a2", "a2", "b1", "b1", "b1"]


def test_write_mp4_produces_a_file(tmp_path):
    out = tmp_path / "clip.mp4"
    write_mp4([bytes(4 * 4 * 3)] * 3, 4, 4, 60, out)
    assert out.stat().st_size > 0

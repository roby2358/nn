"""The hand-built turret: the ordinary net shape, and it aims before it fires."""

import numpy as np

from shards.calculated import turret
from shards.config import (CMD_FIRE, CMD_THRUST, CMD_TURN, N_INPUTS, ROCK_CLOSE, ROCK_COS, ROCK_RADIUS, ROCK_SIN,
                           ROCK_V_SIDEWAYS, SHOT_SPEED, SPEED_SCALE, rock_input)
from shards.evolve import fly
from shards.pilots import GENOME, Brains
from shards.scenario import make_scenario
from shards.worlds import Worlds


def seen(sin: float, cos: float, close: float, v_side: float, size: float) -> np.ndarray:
    """One observation holding only the nearest rock."""
    obs = np.zeros((1, N_INPUTS))
    for field, value in ((ROCK_SIN, sin), (ROCK_COS, cos), (ROCK_CLOSE, close), (ROCK_V_SIDEWAYS, v_side),
                         (ROCK_RADIUS, size)):
        obs[0, rock_input(0, field)] = value
    return obs


def test_it_is_an_ordinary_genome():
    assert turret().genomes.shape == (1, GENOME)


def test_it_turns_toward_the_rock():
    assert turret().act(seen(0.5, 0.87, 0.5, 0.0, 1.0))[0, CMD_TURN] > 0.5
    assert turret().act(seen(-0.5, 0.87, 0.5, 0.0, 1.0))[0, CMD_TURN] < -0.5


def test_it_leads_a_crossing_rock():
    sideways = 200.0 / SPEED_SCALE
    on_lead = -200.0 / SHOT_SPEED     # the bearing cancels the lead term, so the shot would intercept
    assert abs(turret().act(seen(on_lead, 0.97, 0.5, sideways, 1.0))[0, CMD_TURN]) < 0.05


def test_it_fires_only_on_target():
    assert turret().act(seen(0.0, 1.0, 0.5, 0.0, 1.0))[0, CMD_FIRE] > 0.5
    assert turret().act(seen(0.3, 0.95, 0.5, 0.0, 1.0))[0, CMD_FIRE] < 0.5


def test_it_holds_fire_with_nothing_to_shoot():
    assert turret().act(np.zeros((1, N_INPUTS)))[0, CMD_FIRE] < 0.5
    assert turret().act(seen(0.0, -1.0, 0.5, 0.0, 1.0))[0, CMD_FIRE] < 0.5


def test_it_does_not_thrust():
    assert turret().act(seen(0.0, 1.0, 0.5, 0.0, 1.0))[0, CMD_THRUST] < 0.01


def test_it_plays_a_real_round():
    scenarios = [make_scenario(s) for s in range(8)]
    w = fly(Worlds(scenarios, np.arange(8)), Brains(np.tile(turret().genomes, (8, 1))))
    assert w.hits.sum() / w.fired.sum() > 0.45
    assert w.survived.mean() > 40.0

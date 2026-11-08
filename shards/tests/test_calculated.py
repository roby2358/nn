"""The hand-built hunter: the ordinary net shape, it aims before it fires, closes in and avoids hazards."""

import numpy as np

from shards.calculated import hunter
from shards.config import (CMD_FIRE, CMD_THRUST, CMD_TURN, N_INPUTS, ROCK_CLOSE, ROCK_COS, ROCK_RADIUS, ROCK_SIN,
                           ROCK_V_FORWARD, ROCK_V_SIDEWAYS, RUN_SECONDS, RUN_STEPS, SHOT_SPEED, SPEED_SCALE, V_FORWARD,
                           rock_input)
from shards.evolve import fly
from shards.pilots import GENOME, Brains
from shards.scenario import make_scenario
from shards.worlds import Worlds


def seen(sin: float, cos: float, close: float, v_side: float, size: float, v_forward: float = 0.0) -> np.ndarray:
    """One observation holding only the nearest rock."""
    obs = np.zeros((1, N_INPUTS))
    for field, value in ((ROCK_SIN, sin), (ROCK_COS, cos), (ROCK_CLOSE, close), (ROCK_V_SIDEWAYS, v_side),
                         (ROCK_RADIUS, size), (ROCK_V_FORWARD, v_forward)):
        obs[0, rock_input(0, field)] = value
    return obs


def test_it_is_an_ordinary_genome():
    assert hunter().genomes.shape == (1, GENOME)


def test_it_turns_toward_the_rock():
    assert hunter().act(seen(0.5, 0.87, 0.5, 0.0, 1.0))[0, CMD_TURN] > 0.5
    assert hunter().act(seen(-0.5, 0.87, 0.5, 0.0, 1.0))[0, CMD_TURN] < -0.5


def test_it_leads_a_crossing_rock():
    sideways = 200.0 / SPEED_SCALE
    on_lead = -200.0 / SHOT_SPEED     # the bearing cancels the lead term, so the shot would intercept
    assert abs(hunter().act(seen(on_lead, 0.97, 0.5, sideways, 1.0))[0, CMD_TURN]) < 0.05


def test_it_fires_only_on_target():
    assert hunter().act(seen(0.0, 1.0, 0.5, 0.0, 1.0))[0, CMD_FIRE] > 0.5
    assert hunter().act(seen(0.3, 0.95, 0.5, 0.0, 1.0))[0, CMD_FIRE] < 0.5


def test_it_holds_fire_with_nothing_to_shoot():
    assert hunter().act(np.zeros((1, N_INPUTS)))[0, CMD_FIRE] < 0.5
    assert hunter().act(seen(0.0, -1.0, 0.5, 0.0, 1.0))[0, CMD_FIRE] < 0.5


def test_it_advances_when_the_way_ahead_is_open():
    assert hunter().act(seen(0.0, 1.0, 0.3, 0.0, 1.0))[0, CMD_THRUST] > 0.9


def test_it_holds_when_a_rock_blocks_the_way():
    assert hunter().act(seen(0.0, 1.0, 0.75, 0.0, 1.0))[0, CMD_THRUST] < 0.1


def test_a_rock_coming_at_the_nose_blocks_the_way_sooner():
    assert hunter().act(seen(0.0, 1.0, 0.5, 0.0, 1.0))[0, CMD_THRUST] > 0.9
    assert hunter().act(seen(0.0, 1.0, 0.5, 0.0, 1.0, v_forward=-100.0 / SPEED_SCALE))[0, CMD_THRUST] < 0.1


def test_it_stops_advancing_at_cruising_speed():
    obs = seen(0.0, 1.0, 0.3, 0.0, 1.0)
    obs[0, V_FORWARD] = 200.0 / SPEED_SCALE
    assert hunter().act(obs)[0, CMD_THRUST] < 0.1


def test_it_runs_from_a_rock_close_behind():
    assert hunter().act(seen(0.0, -1.0, 0.95, 0.0, 1.0))[0, CMD_THRUST] > 0.9


def test_a_rock_coming_up_from_behind_is_a_threat_sooner():
    drifting, chasing = seen(0.0, -1.0, 0.6, 0.0, 1.0), seen(0.0, -1.0, 0.6, 0.0, 1.0, v_forward=100.0 / SPEED_SCALE)
    drifting[0, V_FORWARD] = chasing[0, V_FORWARD] = 200.0 / SPEED_SCALE      # too fast for an open way to earn thrust
    assert hunter().act(drifting)[0, CMD_THRUST] < 0.1
    assert hunter().act(chasing)[0, CMD_THRUST] > 0.9


def test_it_does_not_burn_toward_a_close_rock_ahead():
    assert hunter().act(seen(0.0, 1.0, 0.95, 0.0, 1.0))[0, CMD_THRUST] < 0.1


def test_it_turns_away_from_a_close_rock_abeam():
    assert hunter().act(seen(1.0, 0.0, 0.92, 0.0, 1.0))[0, CMD_TURN] < -0.5
    assert hunter().act(seen(-1.0, 0.0, 0.92, 0.0, 1.0))[0, CMD_TURN] > 0.5


def test_it_plays_a_real_round():
    scenarios = [make_scenario(s) for s in range(8)]
    w = fly(Worlds(scenarios, np.arange(8), RUN_STEPS), Brains(np.tile(hunter().genomes, (8, 1))))
    assert w.hits.sum() / w.fired.sum() > 0.45
    assert w.survived.mean() > 0.85 * RUN_SECONDS

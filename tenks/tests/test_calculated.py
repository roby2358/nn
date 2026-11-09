"""The calculated tank's decisions, each read off one hand-built observation."""

import math

import numpy as np
import pytest

from tenks.calculated import MIN_RANGE, STOP_RANGE, charger, closeness
from tenks.config import CMD_LEFT, CMD_RIGHT, CMD_SHOOT, CONTACT_INPUTS, FIRE_ABOVE, N_INPUTS, NEAREST, SELF_INPUTS
from tenks.pilots import GENOME, Brains

Contact = tuple[float, float]    # degrees off the nose (positive is the +sin side), px away


def view(enemies: list[Contact], friends: list[Contact]) -> np.ndarray:
    obs = np.zeros(N_INPUTS)
    obs[0] = 1.0
    for first, contacts in ((SELF_INPUTS, enemies), (SELF_INPUTS + NEAREST * CONTACT_INPUTS, friends)):
        for slot, (degrees, px) in enumerate(contacts):
            bearing = math.radians(degrees)
            obs[first + slot * CONTACT_INPUTS:first + (slot + 1) * CONTACT_INPUTS] = (
                math.sin(bearing), math.cos(bearing), closeness(px))
    return obs


@pytest.fixture(scope="module")
def net():
    return Brains(charger()[None])


def command(net, enemies: list[Contact], friends: list[Contact]) -> np.ndarray:
    return net.act(view(enemies, friends)[None])[0]


def shoots(net, enemies: list[Contact], friends: list[Contact]) -> bool:
    return bool(command(net, enemies, friends)[CMD_SHOOT] > FIRE_ABOVE)


def test_the_design_fits_the_genome():
    assert charger().shape == (GENOME,)


@pytest.mark.parametrize("px", [40.0, 120.0, 180.0, 190.0])
def test_fires_on_an_enemy_dead_ahead_in_range(net, px):
    assert shoots(net, [(0.0, px)], [])


@pytest.mark.parametrize("enemy", [(0.0, 240.0), (20.0, 120.0), (180.0, 120.0), (0.0, MIN_RANGE - 5.0)])
def test_holds_fire_when_the_shot_would_be_wasted_or_suicidal(net, enemy):
    assert not shoots(net, [enemy], [])


def test_fires_on_a_farther_enemy_that_crosses_the_sights(net):
    assert shoots(net, [(60.0, 100.0), (1.0, 150.0)], [])


def test_a_nearest_enemy_too_close_holds_fire_even_with_another_in_the_sights(net):
    assert not shoots(net, [(60.0, MIN_RANGE - 5.0), (0.0, 150.0)], [])


@pytest.mark.parametrize("friend", [(0.0, 60.0), (3.0, 150.0), (2.0, 155.0)])
def test_a_teammate_on_the_line_or_by_the_blast_vetoes_the_shot(net, friend):
    assert not shoots(net, [(0.0, 150.0)], [friend])


@pytest.mark.parametrize("slot", range(NEAREST))
def test_every_visible_teammate_can_veto(net, slot):
    friends = [(90.0, 100.0)] * NEAREST
    friends[slot] = (0.0, 100.0)
    assert not shoots(net, [(0.0, 150.0)], friends)


def test_a_teammate_well_to_the_side_or_behind_does_not_veto(net):
    assert shoots(net, [(0.0, 150.0)], [(60.0, 60.0), (180.0, 60.0), (-90.0, 100.0)])


def test_steers_toward_the_nearest_enemy(net):
    out = command(net, [(25.0, 300.0)], [])
    assert out[CMD_LEFT] > out[CMD_RIGHT]
    out = command(net, [(-25.0, 300.0)], [])
    assert out[CMD_LEFT] < out[CMD_RIGHT]


def test_charges_when_lined_up(net):
    out = command(net, [(0.0, 300.0)], [])
    assert out[CMD_LEFT] > 0.8 and out[CMD_RIGHT] > 0.8


def test_stops_short_and_backs_off_inside_the_stop_distance(net):
    out = command(net, [(0.0, STOP_RANGE)], [])
    assert abs(out[CMD_LEFT]) < 0.1 and abs(out[CMD_RIGHT]) < 0.1
    out = command(net, [(0.0, STOP_RANGE - 7.5)], [])
    assert out[CMD_LEFT] < -0.3 and out[CMD_RIGHT] < -0.3


def test_an_enemy_dead_astern_still_gets_a_turn(net):
    out = command(net, [(180.0, 150.0)], [])
    assert abs(out[CMD_LEFT] - out[CMD_RIGHT]) > 0.5


def turn(out: np.ndarray) -> float:
    return float(out[CMD_LEFT] - out[CMD_RIGHT])


def test_on_the_approach_a_too_close_teammate_tilts_the_heading_away(net):
    alone = turn(command(net, [(0.0, 320.0)], []))
    assert turn(command(net, [(0.0, 320.0)], [(70.0, 18.0)])) < alone - 0.2


def test_on_the_approach_a_drifted_nearest_teammate_tilts_the_heading_toward_it(net):
    alone = turn(command(net, [(0.0, 320.0)], []))
    assert turn(command(net, [(0.0, 320.0)], [(70.0, 120.0)])) > alone + 0.2


def test_in_range_spacing_leaves_the_aim_alone(net):
    alone = turn(command(net, [(0.0, 150.0)], []))
    assert turn(command(net, [(0.0, 150.0)], [(70.0, 18.0)])) == pytest.approx(alone, abs=0.05)


def test_a_teammate_close_ahead_slows_the_tank(net):
    clear = command(net, [(0.0, 150.0)], [])
    blocked = command(net, [(0.0, 150.0)], [(10.0, 15.0)])
    assert blocked[CMD_LEFT] < clear[CMD_LEFT] - 0.3

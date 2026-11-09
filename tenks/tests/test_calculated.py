"""The calculated tank's decisions, each read off one hand-built observation."""

import math

import numpy as np
import pytest

from tenks.calculated import MIN_RANGE, STOP_RANGE, VEER_DEGREES, charger, closeness
from tenks.config import (CLOSE, CMD_LEFT, CMD_RIGHT, CMD_SHOOT, CONTACT_INPUTS, FIRE_ABOVE, GUN_COS, GUN_SIN, LOADED,
                          N_INPUTS, NEAREST, RUN_STEPS, SELF_INPUTS, SENSE_RANGE, TANKS)
from tenks.pilots import GENOME, Brains
from tenks.scenario import Scenario
from tenks.worlds import Worlds

Contact = tuple[float, float]    # degrees off the nose (positive is the +sin side), px away


def view(enemies: list[Contact], friends: list[Contact]) -> np.ndarray:
    obs = np.zeros(N_INPUTS)
    obs[0] = 1.0
    for first, contacts in ((SELF_INPUTS, enemies), (SELF_INPUTS + NEAREST * CONTACT_INPUTS, friends)):
        for slot, (degrees, px) in enumerate(contacts):
            bearing = math.radians(degrees)
            start = first + slot * CONTACT_INPUTS
            obs[start:start + CLOSE + 1] = math.sin(bearing), math.cos(bearing), closeness(px)
    return obs


@pytest.fixture(scope="module")
def net():
    return Brains(charger()[None])


def command(net, enemies: list[Contact], friends: list[Contact]) -> np.ndarray:
    return net.act(view(enemies, friends)[None])[0]


def shoots(net, enemies: list[Contact], friends: list[Contact]) -> bool:
    return bool(command(net, enemies, friends)[CMD_SHOOT] > FIRE_ABOVE)


def facing(net, bearing: float, px: float, gun_off: float, loaded: bool) -> np.ndarray:
    """The command for one enemy at (bearing, px) whose gun points gun_off degrees off this tank."""
    obs = view([(bearing, px)], [])
    obs[SELF_INPUTS + GUN_SIN], obs[SELF_INPUTS + GUN_COS] = math.sin(math.radians(gun_off)), math.cos(math.radians(gun_off))
    obs[SELF_INPUTS + LOADED] = float(loaded)
    return net.act(obs[None])[0]


def fires(out: np.ndarray) -> bool:
    return bool(out[CMD_SHOOT] > FIRE_ABOVE)


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


def test_charges_and_steers_at_an_enemy_too_far_away_to_register_closeness(net):
    ahead = command(net, [(0.0, SENSE_RANGE)], [])
    assert ahead[CMD_LEFT] > 0.8 and ahead[CMD_RIGHT] > 0.8
    off_side = command(net, [(25.0, SENSE_RANGE)], [])
    assert off_side[CMD_LEFT] > off_side[CMD_RIGHT]


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


@pytest.mark.parametrize("gun_off", [0.0, 10.0, 30.0])
def test_holds_fire_on_a_loaded_gun_that_could_swing_back_in_time(net, gun_off):
    assert not fires(facing(net, 0.0, 150.0, gun_off, True))


def test_fires_on_a_loaded_gun_turned_too_far_to_answer(net):
    assert fires(facing(net, 0.0, 150.0, 70.0, True))


def test_a_shorter_shell_flight_narrows_what_counts_as_able_to_answer(net):
    assert fires(facing(net, 0.0, 60.0, 30.0, True))


def test_fires_on_a_reloading_gun_even_when_it_points_straight_back(net):
    assert fires(facing(net, 0.0, 150.0, 0.0, False))


def test_veers_off_a_nearest_enemy_that_could_answer(net):
    assert abs(turn(facing(net, 0.0, 150.0, 0.0, True))) > 0.5


def test_the_veer_settles_at_its_angle_off_the_enemy(net):
    assert turn(facing(net, -VEER_DEGREES, 150.0, 0.0, True)) == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("px, gun_off, loaded", [(150.0, 0.0, False), (150.0, 70.0, True), (260.0, 0.0, True)])
def test_no_veer_when_it_cannot_answer_or_is_out_of_reach(net, px, gun_off, loaded):
    assert turn(facing(net, 0.0, px, gun_off, loaded)) == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("nudge", [0.002, -0.01, 0.05, 0.2])
def test_two_tanks_meeting_head_on_do_not_both_die(nudge):
    spots, headings = np.zeros((TANKS, 2)), np.zeros(TANKS)
    spots[0], spots[8] = (200.0, 360.0), (520.0, 360.0 + 40.0 * nudge)
    headings[0], headings[8] = nudge, math.pi
    worlds = Worlds([Scenario(0, spots, headings)], RUN_STEPS)
    worlds.alive[0] = False
    worlds.alive[0, [0, 8]] = True
    brains = Brains(np.tile(charger(), (TANKS, 1)))
    while not worlds.done and not worlds.over[0]:
        worlds.step(brains.act(worlds.observe()))
    assert worlds.alive[0].sum() == 1


def test_a_teammate_close_ahead_slows_the_tank(net):
    clear = command(net, [(0.0, 150.0)], [])
    blocked = command(net, [(0.0, 150.0)], [(10.0, 15.0)])
    assert blocked[CMD_LEFT] < clear[CMD_LEFT] - 0.3

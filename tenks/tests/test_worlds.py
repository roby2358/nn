"""The rules of the battlefield, one tank at a time."""

import math

import numpy as np
import pytest

from tenks.config import (CLOSE, COS, DT, MAX_SPEED, N_INPUTS, N_OUTPUTS, RELOAD, SELF_INPUTS, SENSE_RANGE,
                          SHELL_RANGE, SHELL_SPEED, SIN, SPLASH_R, SURVIVAL_POINTS, TANK_R, TANKS, TURN_RATE)
from tenks.scenario import Scenario, make_scenario
from tenks.torus import length, shortest
from tenks.worlds import Worlds

GREEN_A, GREEN_B, BLUE_A, BLUE_B = 0, 1, 8, 9
ENEMY_BLOCK = SELF_INPUTS
FRIEND_BLOCK = SELF_INPUTS + 4 * 3
FAR_GREEN, FAR_BLUE = 7, 15
FAR_SPOTS = {FAR_GREEN: (40.0, 40.0, 0.0), FAR_BLUE: (40.0, 680.0, 0.0)}


def arena(tanks: dict[int, tuple[float, float, float]]) -> Worlds:
    """One world with only the given tanks alive, as tank -> (x, y, heading). Two parked
    spectators, one per team, keep the battle undecided unless a test kills them."""
    placed = {**FAR_SPOTS, **tanks}
    spots, headings = np.zeros((TANKS, 2)), np.zeros(TANKS)
    alive = np.zeros(TANKS, dtype=bool)
    for tank, (x, y, heading) in placed.items():
        spots[tank], headings[tank], alive[tank] = (x, y), heading, True
    worlds = Worlds([Scenario(0, spots, headings)], 10 ** 6)
    worlds.alive[0] = alive
    return worlds


def orders(**by_tank: tuple) -> np.ndarray:
    """Commands from t<tank>=(left, right, shoot); every other tank does nothing."""
    out = np.zeros((TANKS, N_OUTPUTS))
    for name, command in by_tank.items():
        out[int(name[1:])] = command
    return out


def run(worlds: Worlds, commands: np.ndarray, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        worlds.step(commands)


def fire_once(worlds: Worlds, tank: int, seconds: float) -> None:
    """Pull one tank's trigger for a single step, then let the shell play out."""
    worlds.step(orders(**{f"t{tank}": (0.0, 0.0, 1.0)}))
    run(worlds, orders(), seconds)


def test_both_treads_forward_drive_along_the_heading():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    run(w, orders(t0=(1.0, 1.0, 0.0)), 1.0)
    assert w.pos[0, GREEN_A] == pytest.approx([100.0 + MAX_SPEED, 300.0])


def test_opposite_treads_spin_in_place():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    run(w, orders(t0=(1.0, -1.0, 0.0)), 0.5)
    assert w.pos[0, GREEN_A] == pytest.approx([100.0, 300.0])
    assert w.ang[0, GREEN_A] == pytest.approx(0.5 * TURN_RATE)


def test_a_stronger_right_tread_turns_left_on_screen():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    run(w, orders(t0=(0.0, 1.0, 0.0)), 0.5)
    assert w.ang[0, GREEN_A] < 0.0        # screen y points down, so heading falling is counterclockwise


def test_reverse_backs_up_and_speed_stops_with_the_treads():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    run(w, orders(t0=(-1.0, -1.0, 0.0)), 1.0)
    run(w, orders(), 1.0)
    assert w.pos[0, GREEN_A] == pytest.approx([100.0 - MAX_SPEED, 300.0])


def test_tanks_wrap_at_the_edge():
    w = arena({GREEN_A: (719.0, 300.0, 0.0)})
    run(w, orders(t0=(1.0, 1.0, 0.0)), 0.1)
    assert 0.0 < w.pos[0, GREEN_A, 0] < 10.0


def test_tanks_block_each_other():
    w = arena({GREEN_A: (300.0, 300.0, 0.0), BLUE_A: (300.0 + 2.0 * TANK_R - 3.0, 300.0, 0.0)})
    w.step(orders())
    assert length(shortest(w.pos[0, BLUE_A] - w.pos[0, GREEN_A])) == pytest.approx(2.0 * TANK_R)


def test_dead_tanks_do_not_block():
    w = arena({GREEN_A: (300.0, 300.0, 0.0), BLUE_A: (300.0 + 2.0 * TANK_R - 3.0, 300.0, 0.0)})
    w.alive[0, BLUE_A] = False
    w.step(orders())
    assert w.pos[0, GREEN_A] == pytest.approx([300.0, 300.0])


def test_reload_limits_the_rate_of_fire():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    run(w, orders(t0=(0.0, 0.0, 1.0)), 2.5 * RELOAD)
    assert w.fired[0, GREEN_A] == 3


def test_shoot_below_the_threshold_holds_fire():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    run(w, orders(t0=(0.0, 0.0, 0.4)), 1.0)
    assert w.fired[0, GREEN_A] == 0


def test_a_direct_hit_kills_outright_and_credits_the_owner():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), BLUE_A: (200.0, 300.0, 0.0)})
    fire_once(w, GREEN_A, 0.5)
    assert not w.alive[0, BLUE_A]
    assert w.kills[0, GREEN_A] == 1
    assert w.near_misses[0, GREEN_A] == 0


def test_splash_kills_every_tank_in_the_radius_and_spares_those_outside():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), BLUE_A: (200.0, 300.0, 0.0),
               BLUE_B: (200.0, 300.0 + SPLASH_R - 6.0, 0.0), 10: (200.0, 300.0 + SPLASH_R + 30.0, 0.0)})
    fire_once(w, GREEN_A, 0.5)
    assert not w.alive[0, BLUE_A] and not w.alive[0, BLUE_B] and w.alive[0, 10]
    assert w.kills[0, GREEN_A] == 2


def test_a_shell_goes_off_at_full_range():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), BLUE_A: (100.0 + TANK_R + SHELL_RANGE + 7.0, 300.0 + TANK_R + 2.0, 0.0)})
    fire_once(w, GREEN_A, 0.5)
    assert not w.alive[0, BLUE_A]     # never touched, but inside the blast where the shell ran out


def test_splash_kills_teammates_and_costs_the_shooter():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), GREEN_B: (200.0, 300.0, 0.0)})
    fire_once(w, GREEN_A, 0.5)
    assert not w.alive[0, GREEN_B]
    assert (w.kills[0, GREEN_A], w.friendly_kills[0, GREEN_A]) == (0, 1)


def test_point_blank_splash_kills_the_shooter_too():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), BLUE_A: (100.0 + 2.0 * TANK_R + 0.5, 300.0, 0.0)})
    fire_once(w, GREEN_A, 0.1)
    assert not w.alive[0, GREEN_A] and not w.alive[0, BLUE_A]
    assert (w.kills[0, GREEN_A], w.friendly_kills[0, GREEN_A]) == (1, 1)


def test_a_shell_passing_close_is_a_near_miss():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), BLUE_A: (200.0, 300.0 + SPLASH_R + 15.0, 0.0)})
    fire_once(w, GREEN_A, 0.5)
    assert w.alive[0, BLUE_A]
    assert w.near_misses[0, GREEN_A] == 1


def test_a_shell_near_only_friends_is_no_near_miss():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), GREEN_B: (200.0, 300.0 + SPLASH_R + 15.0, 0.0)})
    fire_once(w, GREEN_A, 0.5)
    assert w.near_misses[0, GREEN_A] == 0


def test_a_dead_tanks_shell_still_scores_for_it():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), BLUE_A: (200.0, 300.0, 0.0)})
    w.step(orders(t0=(0.0, 0.0, 1.0)))
    w.alive[0, GREEN_A] = False
    run(w, orders(), 0.5)
    assert not w.alive[0, BLUE_A]
    assert w.kills[0, GREEN_A] == 1


def test_two_shells_on_one_tank_both_get_the_kill():
    w = arena({GREEN_A: (100.0, 300.0, 0.0), GREEN_B: (300.0, 300.0, math.pi), BLUE_A: (200.0, 300.0, 0.0)})
    w.step(orders(t0=(0.0, 0.0, 1.0), t1=(0.0, 0.0, 1.0)))
    run(w, orders(), 0.5)
    assert w.kills[0, GREEN_A] == 1 and w.kills[0, GREEN_B] == 1


def test_shell_travels_at_shell_speed():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    w.step(orders(t0=(0.0, 0.0, 1.0)))
    start = w.shell_pos[0, GREEN_A].copy()
    w.step(orders())
    assert w.shell_pos[0, GREEN_A] - start == pytest.approx([SHELL_SPEED * DT, 0.0])


def test_observation_lists_nearest_enemies_first_in_the_tanks_own_frame():
    w = arena({GREEN_A: (300.0, 300.0, math.pi / 2), BLUE_A: (300.0, 500.0, 0.0), BLUE_B: (300.0, 400.0, 0.0)})
    obs = w.observe()
    assert obs.shape == (TANKS, N_INPUTS)
    first, second = obs[GREEN_A, ENEMY_BLOCK:ENEMY_BLOCK + 3], obs[GREEN_A, ENEMY_BLOCK + 3:ENEMY_BLOCK + 6]
    assert first[COS] == pytest.approx(1.0)         # dead ahead, since the tank faces +y
    assert first[SIN] == pytest.approx(0.0, abs=1e-9)
    assert first[CLOSE] == pytest.approx(1.0 - 100.0 / SENSE_RANGE)
    assert second[CLOSE] == pytest.approx(1.0 - 200.0 / SENSE_RANGE)


def test_observation_measures_across_the_wrap():
    w = arena({GREEN_A: (710.0, 300.0, 0.0), BLUE_A: (20.0, 300.0, 0.0)})
    enemy = w.observe()[GREEN_A, ENEMY_BLOCK:ENEMY_BLOCK + 3]
    assert enemy[COS] == pytest.approx(1.0)
    assert enemy[CLOSE] == pytest.approx(1.0 - 30.0 / SENSE_RANGE)


def test_observation_friends_leave_out_the_tank_itself_and_absent_slots_are_zero():
    w = arena({GREEN_A: (300.0, 300.0, 0.0), GREEN_B: (350.0, 300.0, 0.0)})
    friends = w.observe()[GREEN_A, FRIEND_BLOCK:]
    assert friends[COS] == pytest.approx(1.0)        # teammate B first, never tank A itself
    assert (friends[6:] == 0.0).all()                # B and the parked spectator fill two slots of four


def test_dead_tanks_see_nothing_and_are_not_seen():
    w = arena({GREEN_A: (300.0, 300.0, 0.0), BLUE_A: (350.0, 300.0, 0.0)})
    w.alive[0, BLUE_A] = False
    obs = w.observe()
    assert (obs[BLUE_A] == 0.0).all()
    parked = length(shortest(w.pos[0, FAR_BLUE] - w.pos[0, GREEN_A]))
    assert obs[GREEN_A, ENEMY_BLOCK + CLOSE] == pytest.approx(max(1.0 - parked / SENSE_RANGE, 0.0))
    assert (obs[GREEN_A, ENEMY_BLOCK + 3:FRIEND_BLOCK] == 0.0).all()   # the parked blue tank is the only one visible


def test_a_decided_battle_freezes_movement_and_survivors_keep_scoring():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    w.alive[0, FAR_BLUE] = False
    start = w.pos[0, GREEN_A].copy()
    run(w, orders(t0=(1.0, 1.0, 1.0)), 1.0)
    assert w.over[0]
    assert w.pos[0, GREEN_A] == pytest.approx(start)
    assert w.fired[0, GREEN_A] == 0
    assert w.survived[0, GREEN_A] == pytest.approx(1.0)
    assert w.fitness()[0, GREEN_A] == pytest.approx(SURVIVAL_POINTS)


def test_scenarios_are_repeatable_and_teams_start_out_of_range():
    a, b = make_scenario(5), make_scenario(5)
    assert np.array_equal(a.spots, b.spots) and np.array_equal(a.headings, b.headings)
    gaps = length(shortest(a.spots[:8][:, None] - a.spots[8:][None]))
    assert gaps.min() >= SHELL_RANGE - 1e-9


def test_commands_stay_clipped():
    w = arena({GREEN_A: (100.0, 300.0, 0.0)})
    run(w, orders(t0=(9.0, 9.0, 0.0)), 1.0)
    assert w.pos[0, GREEN_A] == pytest.approx([100.0 + MAX_SPEED, 300.0])

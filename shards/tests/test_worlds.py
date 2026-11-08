"""Rules of the rock field: motion, wrapping, splitting, scoring and fair spawning."""

import numpy as np

from shards.config import (BIG, CMD_FIRE, CMD_THRUST, CMD_TURN, COOLDOWN, DT, FIELD, FIELD_W, MAX_SPEED, N_INPUTS,
                           N_OUTPUTS, POINTS, ROCK_COS, ROCK_SIN, ROCK_SLOTS, RUN_STEPS, SHIP_R, SHOT_COST, SHOT_LIFE,
                           SHOT_SLOTS, SPAWN_CLEAR, SURVIVAL_POINTS, WAVE_BASE, rock_input)
from shards.scenario import make_scenario
from shards.worlds import Worlds, shortest


def full(output: int) -> np.ndarray:
    """A one-ship command with just this output at full."""
    command = np.zeros((1, N_OUTPUTS))
    command[0, output] = 1.0
    return command


IDLE = np.zeros((1, N_OUTPUTS))
BURN = full(CMD_THRUST)
FIRE = full(CMD_FIRE)
TURN = full(CMD_TURN)
NEAREST_BEARING = [rock_input(0, ROCK_SIN), rock_input(0, ROCK_COS)]
NEAREST_COS = rock_input(0, ROCK_COS)


def world_of_one() -> Worlds:
    return Worlds([make_scenario(1)], np.zeros(1, dtype=np.int64))


def empty_but_one_rock(w: Worlds, pos: tuple, size: int) -> None:
    """Leave a single motionless rock of `size` at `pos`."""
    w.rock_size[:] = 0
    w.rock_vel[:] = 0.0
    w.rock_pos[0, 0] = pos
    w.rock_size[0, 0] = size


def aim_at_right(w: Worlds) -> None:
    w.ang[0] = 0.0


def test_first_wave_keeps_clear_of_the_ship():
    for seed in range(50):
        w = Worlds([make_scenario(seed)], np.zeros(1, dtype=np.int64))
        assert (w.rock_size[0] == BIG).sum() == WAVE_BASE
        live = w.rock_size[0] > 0
        assert np.hypot(*shortest(w.rock_pos[0][live] - w.pos[0]).T).min() >= SPAWN_CLEAR


def test_same_scenario_same_rocks():
    a, b = world_of_one(), world_of_one()
    assert np.array_equal(a.rock_pos, b.rock_pos) and np.array_equal(a.rock_vel, b.rock_vel)


def test_ship_coasts_after_a_burn_and_slowly_loses_speed():
    w = world_of_one()
    empty_but_one_rock(w, (10.0, 10.0), 1)
    for _ in range(30):
        w.step(BURN)
    burned = np.hypot(*w.vel[0])
    for _ in range(30):
        w.step(IDLE)
    coasted = np.hypot(*w.vel[0])
    assert burned > 0.0 and 0.6 * burned < coasted < burned


def test_speed_is_capped():
    w = world_of_one()
    empty_but_one_rock(w, (10.0, 10.0), 1)
    w.vel[0] = (MAX_SPEED * 2, 0.0)
    w.step(IDLE)
    assert np.hypot(*w.vel[0]) <= MAX_SPEED


def test_ship_wraps_at_the_edge():
    w = world_of_one()
    empty_but_one_rock(w, (10.0, 360.0), 1)
    w.pos[0] = (FIELD_W - 1.0, 100.0)
    w.vel[0] = (200.0, 0.0)
    w.step(IDLE)
    assert 0.0 <= w.pos[0, 0] < 10.0


def test_partial_thrust_pushes_partly():
    whole, half = world_of_one(), world_of_one()
    for w in (whole, half):
        empty_but_one_rock(w, (10.0, 10.0), 1)
    whole.step(BURN)
    half.step(BURN * 0.5)
    assert np.hypot(*half.vel[0]) == np.hypot(*whole.vel[0]) / 2


def test_turn_command_scales_the_turn():
    whole, half = world_of_one(), world_of_one()
    a0 = whole.ang[0]
    whole.step(TURN)
    half.step(TURN * 0.5)
    assert np.isclose(half.ang[0] - a0, (whole.ang[0] - a0) / 2)


def test_shots_expire_and_respect_the_cooldown():
    w = world_of_one()
    empty_but_one_rock(w, (10.0, 10.0), 1)
    aim_at_right(w)
    w.step(FIRE)
    assert w.fired[0] == 1
    w.step(FIRE)
    assert w.fired[0] == 1
    for _ in range(int(COOLDOWN / DT) + 1):
        w.step(IDLE)
    assert (w.shot_ttl[0] > 0).sum() == 1
    for _ in range(int(SHOT_LIFE / DT) + 1):
        w.step(IDLE)
    assert (w.shot_ttl[0] > 0).sum() == 0


def test_never_more_shots_than_slots():
    w = world_of_one()
    empty_but_one_rock(w, (10.0, 10.0), 1)
    aim_at_right(w)
    for _ in range(int(SHOT_LIFE / DT)):
        w.step(FIRE)
        assert (w.shot_ttl[0] > 0).sum() <= SHOT_SLOTS


def shoot_rock(size: int) -> Worlds:
    w = world_of_one()
    empty_but_one_rock(w, (w.pos[0, 0] + 120.0, w.pos[0, 1]), size)
    aim_at_right(w)
    w.step(FIRE)
    while w.hits[0] == 0:
        w.step(IDLE)
    return w


def test_big_rock_splits_into_two_medium():
    w = shoot_rock(3)
    assert sorted(w.rock_size[0][w.rock_size[0] > 0]) == [2, 2]
    assert w.score[0] == POINTS[3] and (w.shot_ttl[0] > 0).sum() == 0


def test_medium_rock_splits_into_two_small():
    w = shoot_rock(2)
    assert sorted(w.rock_size[0][w.rock_size[0] > 0]) == [1, 1]


def test_clearing_the_last_rock_starts_a_bigger_wave():
    w = shoot_rock(1)
    assert w.wave[0] == 1 and (w.rock_size[0] == BIG).sum() == WAVE_BASE + 1


def test_a_single_wave_always_fits_its_slots():
    w = world_of_one()
    w.rock_size[:] = 0
    w.rock_size[0, :8] = BIG
    for _ in range(3 * 8):
        rock = int(np.argmax(w.rock_size[0] > 1))
        w.shot_ttl[0, 0] = 1.0
        w._strike(0, 0, rock)
    assert (w.rock_size[0] == 1).sum() == ROCK_SLOTS


def test_rock_kills_the_ship_and_ends_the_round():
    w = world_of_one()
    empty_but_one_rock(w, tuple(w.pos[0] + (SHIP_R, 0.0)), 1)
    w.step(IDLE)
    assert not w.alive[0] and w.done
    assert [e.kind for e in w.events] == ["crash"]


def test_a_dead_ships_shots_stop_scoring():
    w = world_of_one()
    empty_but_one_rock(w, tuple(w.pos[0] + (SHIP_R, 0.0)), 1)
    w.shot_ttl[0, 0] = 1.0
    w.shot_pos[0, 0] = (50.0, 50.0)
    w.step(IDLE)
    assert not w.alive[0] and (w.shot_ttl[0] == 0.0).all()


def test_collisions_reach_across_the_wrap():
    w = world_of_one()
    empty_but_one_rock(w, (FIELD_W - 2.0, 300.0), 1)
    w.pos[0] = (2.0, 300.0)
    w.step(IDLE)
    assert not w.alive[0]


def test_round_ends_on_time_and_survival_counts():
    w = world_of_one()
    empty_but_one_rock(w, tuple(w.pos[0] + FIELD / 4), 1)
    for _ in range(RUN_STEPS + 5):
        w.step(IDLE)
    assert w.steps == RUN_STEPS and w.done
    assert np.isclose(w.fitness()[0], SURVIVAL_POINTS * w.survived[0])


def test_observation_sees_a_rock_in_the_ships_frame():
    w = world_of_one()
    empty_but_one_rock(w, (w.pos[0, 0] + 100.0, w.pos[0, 1]), 3)
    aim_at_right(w)
    obs = w.observe()
    assert obs.shape == (1, N_INPUTS)
    assert np.allclose(obs[0, NEAREST_BEARING], (0.0, 1.0))       # sin, cos of the bearing: dead ahead
    w.ang[0] = np.pi / 2
    assert np.allclose(w.observe()[0, NEAREST_BEARING], (-1.0, 0.0))  # now off the left side


def test_observation_sees_across_the_wrap_and_zeros_absent_rocks():
    w = world_of_one()
    empty_but_one_rock(w, (FIELD_W - 10.0, 300.0), 3)
    w.pos[0] = (10.0, 300.0)
    aim_at_right(w)
    obs = w.observe()[0]
    assert np.isclose(obs[NEAREST_COS], -1.0)                  # the short way is behind the ship
    assert np.all(obs[rock_input(1, ROCK_SIN):] == 0.0)


def test_observation_lists_the_closest_rock_first():
    w = world_of_one()
    empty_but_one_rock(w, (w.pos[0, 0] + 300.0, w.pos[0, 1]), 1)
    w.rock_pos[0, 1] = (w.pos[0, 0] - 100.0, w.pos[0, 1])
    w.rock_size[0, 1] = 1
    aim_at_right(w)
    assert np.isclose(w.observe()[0, NEAREST_COS], -1.0)


def test_each_shot_costs_fitness():
    w = world_of_one()
    empty_but_one_rock(w, (10.0, 10.0), 1)
    aim_at_right(w)
    w.step(FIRE)
    assert w.fired[0] == 1
    assert np.isclose(w.fitness()[0], SURVIVAL_POINTS * w.survived[0] - SHOT_COST)


def test_waves_can_appear_away_from_the_edges():
    spots = np.concatenate([make_scenario(seed).spots.reshape(-1, 2) for seed in range(10)])
    margin = np.minimum(spots, FIELD - spots).min(1)
    assert (margin > FIELD_W / 4).mean() > 0.1

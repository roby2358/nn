import numpy as np
import pytest

from crystalchase.config import BONUS_FAST, BONUS_SLOW, DT, FIELD_W, MARGIN, MIN_HOP, PICKUP_R, SPAWN_PAD, THRUST
from crystalchase.course import Course, make_course
from crystalchase.flock import Flock, encode_action, shared_rule, solo_rule, speed_bonus

COAST, BURN = encode_action(0, 0), encode_action(0, 1)


def course_at(start, heading, crystals) -> Course:
    pad = [crystals[-1]] * 10
    return Course(0, np.array(start, float), heading, np.array(list(crystals) + pad, float))


def solo(course: Course, n: int) -> Flock:
    return Flock([course], np.zeros(n, dtype=np.int64), solo_rule)


def test_course_is_seeded_and_spread_out():
    a, b = make_course(7), make_course(7)
    assert np.array_equal(a.crystals, b.crystals) and a.heading == b.heading
    hops = np.hypot(*np.diff(np.vstack([a.start, a.crystals]), axis=0).T)
    assert hops.min() >= MIN_HOP
    assert a.crystals.min() >= SPAWN_PAD and a.crystals.max() <= FIELD_W - SPAWN_PAD


def test_speed_bonus_fades_from_fast_to_slow():
    b = speed_bonus(np.array([0.5, BONUS_FAST, (BONUS_FAST + BONUS_SLOW) / 2, BONUS_SLOW, 20.0]))
    assert b == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])


def test_thrust_accelerates_along_heading_and_coasting_keeps_velocity():
    f = solo(course_at((100, 100), 0.0, [(600, 600)]), 1)
    f.step(np.array([BURN]))
    assert f.vel[0] == pytest.approx([THRUST * DT, 0.0])
    for _ in range(10):
        f.step(np.array([COAST]))
    assert f.vel[0] == pytest.approx([THRUST * DT, 0.0])  # pure inertia, no drag


def test_positive_turn_rotates_toward_a_target_on_the_positive_side():
    f = solo(course_at((100, 100), 0.0, [(100, 500)]), 1)
    sin_before = f.observe()[0, 0]
    f.step(np.array([encode_action(1, 0)]))
    assert sin_before == pytest.approx(1.0) and f.observe()[0, 0] < sin_before


def test_observation_of_crystal_dead_ahead():
    f = solo(course_at((100, 100), 0.0, [(500, 100)]), 1)
    sinb, cosb, dist = f.observe()[0, :3]
    assert (sinb, cosb) == pytest.approx((0.0, 1.0)) and dist > 0


def test_crash_beyond_the_margin_ends_the_run_but_keeps_the_score():
    f = solo(course_at((FIELD_W + MARGIN - 1, 300), 0.0, [(100, 100)]), 1)
    f.score[0] = 3.0
    f.vel[0] = (200.0, 0.0)
    f.step(np.array([COAST]))
    assert not f.alive[0] and f.done and f.fitness()[0] >= 3.0
    assert [e.kind for e in f.events] == ["crash"]


def test_fast_pickup_scores_two_and_moves_on():
    f = solo(course_at((100, 100), 0.0, [(100 + PICKUP_R + 1, 100), (500, 500)]), 1)
    f.vel[0] = (120.0, 0.0)
    f.step(np.array([COAST]))
    assert f.crystals[0] == 1 and f.score[0] == pytest.approx(2.0)
    assert f.idx[0] == 1 and f.since[0] == 0.0 and f.closeness()[0] == 0.0


def test_closeness_credits_progress_toward_the_current_crystal():
    f = solo(course_at((100, 100), 0.0, [(500, 100)]), 1)
    f.vel[0] = (400.0, 0.0)
    for _ in range(30):
        f.step(np.array([COAST]))
    assert 0.0 < f.closeness()[0] < 0.5


def test_shared_rule_gives_the_crystal_to_the_closest_and_moves_everyone_on():
    c = course_at((100, 100), 0.0, [(105, 100), (500, 500)])
    f = Flock([c], np.zeros(3, dtype=np.int64), shared_rule)
    f.pos[0] = (110, 100)
    f.pos[1] = (104, 100)
    f.step(np.array([COAST, COAST, COAST]))
    assert f.crystals.tolist() == [0, 1, 0]
    assert f.idx.tolist() == [1, 1, 1]


def test_new_lap_resets_scoring_but_not_ships_or_crystals():
    f = solo(course_at((100, 100), 0.0, [(100 + PICKUP_R + 1, 100), (500, 500)]), 1)
    f.vel[0] = (120.0, 0.0)
    for _ in range(5):
        f.step(np.array([COAST]))
    pos, vel, idx = f.pos.copy(), f.vel.copy(), f.idx.copy()
    f.new_lap()
    assert f.steps == 0 and f.score[0] == 0.0 and f.crystals[0] == 0 and f.closeness()[0] == 0.0
    assert np.array_equal(f.pos, pos) and np.array_equal(f.vel, vel) and np.array_equal(f.idx, idx)


def test_crystal_sequence_wraps_for_endless_flight():
    c = make_course(9)
    f = solo(c, 1)
    f.idx[0] = len(c.crystals) + 2
    assert np.array_equal(f.targets()[0], c.crystals[2])

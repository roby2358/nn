import random
from dataclasses import replace

from lander.physics import (
    CONFIG, NO_CONTROLS, Controls, Lander, Outcome, Rect, Terrain, World, new_world, step,
)

DT = 1 / 60
MAIN = Controls(main=True, left=False, right=False)
LEFT = Controls(main=False, left=True, right=False)


def world_with(lander: Lander, pad: Rect) -> World:
    return World(CONFIG, lander, pad, FLAT, Outcome.FLYING, NO_CONTROLS)


def above_pad(pad: Rect, vx: float, vy: float, x_offset: float) -> Lander:
    x = pad.x + (pad.w - CONFIG.lander_size) / 2 + x_offset
    return Lander(x=x, y=pad.y - CONFIG.lander_size - 0.1, vx=vx, vy=vy)


PAD = Rect(400, 500, CONFIG.pad_width, CONFIG.pad_height)
FLAT = Terrain(((0, CONFIG.ground_y), (CONFIG.width, CONFIG.ground_y)))


def test_pad_is_three_landers_wide_and_in_bottom_third():
    for seed in range(50):
        pad = new_world(CONFIG, random.Random(seed)).pad
        assert pad.w == 3 * CONFIG.lander_size
        assert 0 <= pad.x and pad.right <= CONFIG.width
        assert CONFIG.height * 2 / 3 <= pad.y and pad.bottom < CONFIG.ground_y


def test_gravity_pulls_down():
    w = world_with(Lander(100, 100, 0, 0), PAD)
    assert step(w, NO_CONTROLS, DT).lander.vy > 0


def test_main_thrust_beats_gravity():
    w = step(world_with(Lander(100, 100, 0, 0), PAD), MAIN, DT)
    assert w.lander.vy < 0


def test_left_key_pushes_left():
    w = step(world_with(Lander(100, 100, 0, 0), PAD), LEFT, DT)
    assert w.lander.vx < 0


def test_soft_centered_touchdown_lands():
    w = step(world_with(above_pad(PAD, 0, 20, 0), PAD), NO_CONTROLS, DT)
    assert w.outcome is Outcome.LANDED
    assert w.lander.y + CONFIG.lander_size == PAD.y


def test_fast_touchdown_crashes():
    w = step(world_with(above_pad(PAD, 0, 120, 0), PAD), NO_CONTROLS, DT)
    assert w.outcome is Outcome.CRASHED


def test_hanging_off_pad_edge_crashes():
    w = step(world_with(above_pad(PAD, 0, 20, CONFIG.lander_size * 1.2), PAD), NO_CONTROLS, DT)
    assert w.outcome is Outcome.CRASHED


def test_hitting_ground_crashes():
    lander = Lander(50, CONFIG.ground_y - CONFIG.lander_size - 0.1, 0, 30)
    w = step(world_with(lander, PAD), NO_CONTROLS, DT)
    assert w.outcome is Outcome.CRASHED


def test_leaving_screen_is_lost():
    lander = Lander(-CONFIG.lander_size - 0.5, 100, -60, 0)
    assert step(world_with(lander, PAD), NO_CONTROLS, DT).outcome is Outcome.LOST


def test_finished_world_is_frozen():
    w = replace(world_with(Lander(100, 100, 0, 0), PAD), outcome=Outcome.LANDED)
    assert step(w, MAIN, DT) is w


def test_terrain_meets_pad_and_never_rises_above_it():
    for seed in range(50):
        w = new_world(CONFIG, random.Random(seed))
        t, pad = w.terrain, w.pad
        assert t.points[0][0] == 0 and t.points[-1][0] == CONFIG.width
        assert t.surface_y(pad.x) == pad.bottom == t.surface_y(pad.right)
        assert all(pad.bottom <= y <= CONFIG.ground_y for _, y in t.points)


def test_hitting_a_hill_crashes():
    hill = Terrain(((0, CONFIG.ground_y), (100, 300), (200, CONFIG.ground_y), (CONFIG.width, CONFIG.ground_y)))
    lander = Lander(90, 300 - CONFIG.lander_size - 0.1, 0, 30)
    w = step(World(CONFIG, lander, PAD, hill, Outcome.FLYING, NO_CONTROLS), NO_CONTROLS, DT)
    assert w.outcome is Outcome.CRASHED

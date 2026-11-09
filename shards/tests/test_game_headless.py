"""The App runs without a display: scenes advance, buttons act, rounds roll over."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

from shards import art  # noqa: E402
from shards.config import FREE_PLAY_STEPS  # noqa: E402
from shards.game import END_HOLD_FRAMES, App  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = App()
    yield a
    a.bg.shutdown()


def test_composes_a_frame_with_a_round_underway(app):
    app.tick()
    app.compose(1.0)
    assert app.scene.world.steps == 1


def test_fly_toggles_between_human_and_champion(app):
    app.toggle_fly()
    assert app.scene.pilot is app.human
    app.toggle_fly()
    assert app.scene.pilot is not app.human


def test_rounds_on_screen_have_no_time_limit(app):
    assert app.scene.world.step_limit == FREE_PLAY_STEPS      # Watch, where the game opens
    app.toggle_fly()
    assert app.scene.world.step_limit == FREE_PLAY_STEPS


def test_ship_runs_white_to_orange_with_patience():
    assert art.ship_color(-1.0) == art.IMPATIENT
    assert art.ship_color(1.0) == art.PATIENT


def test_passion_halo_follows_the_heading_and_reddens():
    wide, tall = art.passion_halo(0.5, 0.0).get_size(), art.passion_halo(0.5, 1.5708).get_size()
    assert wide[0] > wide[1] and tall[1] > tall[0]
    center = (art.HALO_LONG, art.HALO_SHORT)
    calm, passionate = art.passion_halo(0.0, 0.0).get_at(center), art.passion_halo(1.0, 0.0).get_at(center)
    assert passionate.r > calm.r and passionate.b < calm.b and passionate.a > calm.a


def test_held_keys_reach_the_ship(app):
    app.toggle_fly()
    app.human.thrust = 1.0
    app.scene.update(1)
    assert app.scene.world.thrust[0] == 1.0


def test_a_finished_round_rolls_over_after_the_hold(app):
    first = app.scene
    first.world.alive[:] = False
    for _ in range(END_HOLD_FRAMES + 1):
        app.tick()
    assert app.scene is not first


def test_quit_keys_stop_the_loop(app):
    assert not app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))


def test_clear_drops_history(app):
    app.evolver.step()
    app.clear()
    assert app.evolver.gen == 0


def test_chart_keeps_negative_fitness_inside_its_frame():
    pygame.font.init()
    surf = pygame.Surface((200, 200))
    surf.fill((255, 0, 255))
    rect = pygame.Rect(20, 20, 160, 100)
    art.draw_chart(surf, rect, [([-40.0, -80.0, -20.0], art.BEST)], pygame.font.Font(None, 16))
    below = surf.subsurface(pygame.Rect(0, rect.bottom, 200, 200 - rect.bottom))
    assert pygame.mask.from_threshold(below, (255, 0, 255), (1, 1, 1)).count() == below.get_width() * below.get_height()

"""The App runs without a display: battles advance, buttons act, rounds roll over."""

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pytest  # noqa: E402

from tenks.config import CMD_LEFT, CMD_RIGHT, DT, MAX_SPEED  # noqa: E402
from tenks.evolve import SWARM, WEIGHTS  # noqa: E402
from tenks.game import DRIVE, END_HOLD_STEPS, WATCH, YOUR_TANK, App, key_treads  # noqa: E402
from tenks.pilots import Commandeered  # noqa: E402
from tenks.worlds import GREEN  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = App()
    yield a
    a.bg.shutdown()


def play_out_the_hold(app):
    for _ in range(END_HOLD_STEPS + 1):
        app.tick()


def test_composes_a_frame_with_a_battle_underway(app):
    app.tick()
    app.compose()
    assert app.scene.world.steps == 1


def test_arrows_map_to_treads():
    assert key_treads(True, False, False, False) == (1.0, 1.0)
    assert key_treads(False, True, False, False) == (-1.0, -1.0)
    assert key_treads(False, False, False, True) == (1.0, -1.0)
    assert key_treads(True, False, True, False) == (0.0, 1.0)


def test_drive_toggles_between_human_and_nets(app):
    app.toggle_drive()
    assert app.mode is DRIVE and isinstance(app.scene.pilots, Commandeered)
    app.toggle_drive()
    assert app.mode is WATCH and not isinstance(app.scene.pilots, Commandeered)


def test_the_human_command_reaches_their_tank_alone(app):
    app.toggle_drive()
    app.human.command[[CMD_LEFT, CMD_RIGHT]] = 1.0
    start, nose = app.scene.world.pos[0, YOUR_TANK].copy(), app.scene.world.nose()[0, YOUR_TANK]
    app.scene.update(1)
    assert app.scene.world.pos[0, YOUR_TANK] == pytest.approx(start + nose * MAX_SPEED * DT)


def test_a_finished_battle_rolls_over_after_the_hold(app):
    first = app.scene
    first.world.alive[:, GREEN] = False
    play_out_the_hold(app)
    assert app.scene is not first


def test_a_drive_ends_when_your_tank_dies(app):
    app.toggle_drive()
    first = app.scene
    first.world.alive[0, YOUR_TANK] = False
    play_out_the_hold(app)
    assert app.scene is not first


def test_every_blast_of_a_fast_frame_is_shown(app):
    w = app.scene.world
    w.shell_ttl[0, 0] = DT       # goes off on the first of the frame's four steps
    app.scene.update(4)
    assert len(app.scene.field.rings.rings) >= 1


def test_quit_keys_stop_the_loop(app):
    assert not app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert app.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))


def test_clear_drops_history(app):
    app.evolver.step()
    app.clear()
    assert app.evolver.gen == 0


def test_load_with_nothing_saved_keeps_the_population(app):
    before = app.evolver.genomes
    app.load()
    assert app.evolver.genomes is before


def test_save_then_load_restores_the_population(app):
    app.evolver.step()
    app.save()
    assert SWARM.exists() and WEIGHTS.exists()
    saved = app.evolver.genomes
    app.clear()
    app.load()
    assert np.array_equal(app.evolver.genomes, saved) and app.evolver.gen == 1


def test_evolution_makes_progress_behind_a_running_window(app):
    app.evolve()
    deadline = time.perf_counter() + 30.0
    while app.evolver.gen < 2 and time.perf_counter() < deadline:
        app.tick()
        app.draw()
        app.clock.tick(60)
    assert app.evolver.gen >= 2

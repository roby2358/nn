import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from crystalchase import art  # noqa: E402
from crystalchase.config import RUN_STEPS  # noqa: E402
from crystalchase.course import make_course  # noqa: E402
from crystalchase.game import App  # noqa: E402
from crystalchase.pilots import AutoPilot, random_genomes  # noqa: E402
from crystalchase.scenes import COUNTDOWN_FRAMES, Fonts, FreeFlight, Race, Replay  # noqa: E402


def test_watch_evolve_and_race_draw_frames(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep weights.npz and racers/ out of the repo
    app = App()
    assert isinstance(app.scene, FreeFlight)  # not evolving: the champion flies alone
    app.draw()
    app.evolver.step()
    app.scene.update(RUN_STEPS)
    app.tick()
    app.draw()
    app.snap()
    app.toggle_race()
    assert isinstance(app.scene, Race) and len(app.scene.racers) == 4  # you, champion, bot, snapped racer
    app.draw()
    for _ in range(COUNTDOWN_FRAMES):
        app.scene.update(1)
    for _ in range(RUN_STEPS // 30):
        app.scene.update(30)
    assert app.scene.flock.done
    app.draw()
    app.toggle_race()
    app.bg.shutdown()
    pygame.quit()


def test_only_the_replay_draws_the_swarm(monkeypatch):
    pygame.display.init()
    pygame.font.init()
    calls = []
    monkeypatch.setattr(art, "draw_ghost", lambda *a: calls.append("ghost"))
    rng = np.random.default_rng(0)
    font = pygame.font.SysFont(None, 20)
    surf, glow = pygame.Surface((720, 720)), pygame.Surface((720, 720), pygame.SRCALPHA)
    scenes = [(Replay("t", make_course(1), random_genomes(rng, 8), rng), 9),   # 8 ghosts + the bot outline
              (FreeFlight("t", make_course(1), AutoPilot(), rng), 0)]
    for scene, expected in scenes:
        calls.clear()
        scene.update(10)
        scene.draw(surf, glow, 0.0, Fonts(font, font, font))
        assert len(calls) == expected
    pygame.quit()


def test_free_flight_laps_reset_the_count_but_not_the_ship():
    rng = np.random.default_rng(1)
    ff = FreeFlight("t", make_course(4), AutoPilot(), rng)
    ff.update(RUN_STEPS - 1)
    f = ff.flock
    pos, vel, idx = f.pos.copy(), f.vel.copy(), f.idx.copy()
    ff.update(1)  # the step that reaches 30 s, then the rollover
    assert ff.lap == 2 and ff.last[0] > 0
    assert f.crystals[0] == 0 and f.score[0] == 0.0 and f.seconds == 0.0
    assert np.hypot(*(f.pos - pos)[0]) < 15 and np.hypot(*(f.vel - vel)[0]) < 15  # one step, no jump
    assert f.idx[0] >= idx[0]  # still chasing the same (or the next) crystal
    assert not f.done and not ff.finished


def test_watch_view_follows_the_evolver(tmp_path, monkeypatch):
    """Generation replays only while evolving; the moment it stops, free flight."""
    monkeypatch.chdir(tmp_path)
    app = App()
    app.evolver.step()
    monkeypatch.setattr(type(app.bg), "running", property(lambda self: True))
    app.tick()
    assert isinstance(app.scene, Replay)
    monkeypatch.setattr(type(app.bg), "running", property(lambda self: False))
    app.tick()
    assert isinstance(app.scene, FreeFlight)
    app.bg.shutdown()
    pygame.quit()

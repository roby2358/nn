import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"


def test_app_runs_frames_in_both_pilots():
    import pygame
    from lander.game import PANEL_W, App
    from lander.physics import CONFIG

    pygame.display.init()
    pygame.font.init()
    app = App(pygame.display.set_mode((CONFIG.width + PANEL_W, CONFIG.height)), CONFIG)
    for name in ("Fly", "Watch"):
        app.actions[name]()
        for _ in range(10):
            app.tick()
        app.draw()
    app.actions["Train"]()
    app.draw()
    app.bg.shutdown()
    pygame.quit()


def landed_app(pilot_name: str):
    from dataclasses import replace

    import pygame
    from lander.game import PANEL_W, App
    from lander.physics import CONFIG, Outcome

    pygame.display.init()
    pygame.font.init()
    app = App(pygame.display.set_mode((CONFIG.width + PANEL_W, CONFIG.height)), CONFIG)
    app.actions[pilot_name]()
    app.world = replace(app.world, outcome=Outcome.LANDED)
    return app


def scene_pixels(app) -> bytes:
    import pygame

    app.draw()
    return pygame.image.tobytes(app.screen.subsurface((0, 0, app.config.width, app.config.height)), "RGB")


def bare_scene_pixels(app) -> bytes:
    import pygame
    from lander.game import draw_scene

    draw_scene(app.screen, app.fonts, app.stars, app.world, app.pilot)
    return pygame.image.tobytes(app.screen.subsurface((0, 0, app.config.width, app.config.height)), "RGB")


def flameless(app):
    from dataclasses import replace

    from lander.physics import NO_CONTROLS

    app.world = replace(app.world, controls=NO_CONTROLS)  # flames flicker randomly between draws
    return app


def test_watch_shows_no_result_banner_and_pauses_half_a_second_before_the_next_round():
    from lander.game import FPS, RESULT_PAUSE_FRAMES
    from lander.physics import Outcome

    assert RESULT_PAUSE_FRAMES == FPS / 2
    app = flameless(landed_app("Watch"))
    assert scene_pixels(app) == bare_scene_pixels(app)
    for _ in range(RESULT_PAUSE_FRAMES - 1):
        app.tick()
    assert app.world.outcome is Outcome.LANDED
    app.tick()
    assert app.world.outcome is Outcome.FLYING
    app.bg.shutdown()


def test_fly_keeps_its_banner_and_waits_for_restart():
    from lander.physics import Outcome

    app = flameless(landed_app("Fly"))
    assert scene_pixels(app) != bare_scene_pixels(app)
    app.tick()
    assert app.world.outcome is Outcome.LANDED
    app.bg.shutdown()

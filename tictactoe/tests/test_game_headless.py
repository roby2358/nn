import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"


def test_app_draws_one_frame():
    import pygame
    from tictactoe.game import App

    app = App()
    app.on_button("Train")
    app.draw()
    app.bg.halt()
    app.on_button("Play")
    app.draw()
    app.bg.shutdown()
    pygame.quit()

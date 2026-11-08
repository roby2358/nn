"""Pygame shell: the scene on the left, the training panel on the right.
All rules live in physics; all learning lives in ai."""
import math
import os
import random
import signal
import threading

import numpy as np
import pygame

from lander import panel
from lander.ai.policy import PolicyNet
from lander.ai.train import TRAIN, WEIGHTS, BackgroundTrainer, Trainer
from lander.physics import CONFIG, DT, NO_CONTROLS, Config, Controls, Lander, Outcome, World, new_world, step

FPS = round(1 / DT)
RESULT_PAUSE_FRAMES = FPS // 2  # half a second on the result before the net's next round
PANEL_W = 440
GAP = 16
BTN_H = 36
TRAIN_STEPS_PER_CLICK = 1000
EVAL_EVERY = 10

BG = (8, 8, 20)
STAR = (200, 200, 220)
GROUND = (96, 96, 104)
RIDGE = (150, 150, 160)
PAD = (60, 220, 120)
HULL = (220, 220, 230)
LEG = (160, 160, 170)
FLAME = (255, 160, 40)
TEXT = (230, 230, 230)
HINT = (140, 140, 160)
OK = (90, 230, 120)
BAD = (240, 80, 80)

BANNERS = {
    Outcome.LANDED: ("Landed!", OK),
    Outcome.CRASHED: ("Crashed", BAD),
    Outcome.LOST: ("Lost in space", BAD),
}


def read_controls(keys) -> Controls:
    return Controls(
        main=keys[pygame.K_w] or keys[pygame.K_s],
        left=keys[pygame.K_a],
        right=keys[pygame.K_d],
    )


# ---- pilots: who flies the lander. The App only talks to this interface. -------
class HumanPilot:
    name = "you"
    auto_restart = False
    frame_limit = math.inf

    def reset(self) -> None:
        pass

    def draw_result(self, surf, fonts, world: World) -> None:
        font, big = fonts
        draw_banner(surf, big, font, world, "Press R to fly again")

    def controls(self, world: World, keys) -> Controls:
        return read_controls(keys)


class NetPilot:
    """Flies greedily, choosing every `decide_every` frames and holding in between,
    exactly as the net does during training."""
    name = "net"
    auto_restart = True

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer
        self.decide_every = trainer.train.decide_every
        self.frame_limit = trainer.train.max_decisions * trainer.train.decide_every
        self.frame = 0
        self.held = NO_CONTROLS

    def reset(self) -> None:
        self.frame = 0
        self.held = NO_CONTROLS

    def draw_result(self, surf, fonts, world: World) -> None:
        pass  # the lander's own ending is the show; no banner

    def controls(self, world: World, keys) -> Controls:
        if self.frame % self.decide_every == 0:
            with self.trainer.lock:
                self.held = self.trainer.net.greedy(world)
        self.frame += 1
        return self.held


# ---- scene ----------------------------------------------------------------------
def draw_flames(surf, config: Config, lander: Lander, c: Controls) -> None:
    s = config.lander_size
    x, y = lander.x, lander.y
    flicker = random.uniform(0.7, 1.3)
    if c.main:
        cx, base = x + s / 2, y + s * 0.75
        pygame.draw.polygon(surf, FLAME, [(cx - 6, base), (cx + 6, base), (cx, base + 18 * flicker)])
    # A pushes left, so its jet fires out the right side; D the reverse.
    if c.left:
        my = y + s * 0.4
        pygame.draw.polygon(surf, FLAME, [(x + s, my - 4), (x + s, my + 4), (x + s + 12 * flicker, my)])
    if c.right:
        my = y + s * 0.4
        pygame.draw.polygon(surf, FLAME, [(x, my - 4), (x, my + 4), (x - 12 * flicker, my)])


def draw_lander(surf, config: Config, lander: Lander) -> None:
    s = config.lander_size
    x, y = lander.x, lander.y
    pygame.draw.rect(surf, HULL, pygame.Rect(x + s * 0.15, y, s * 0.7, s * 0.75), border_radius=5)
    pygame.draw.line(surf, LEG, (x + s * 0.25, y + s * 0.7), (x, y + s), 2)
    pygame.draw.line(surf, LEG, (x + s * 0.75, y + s * 0.7), (x + s, y + s), 2)


def speed_text(font, label: str, value: float, limit: float):
    color = OK if abs(value) <= limit else BAD
    return font.render(f"{label} {value:6.1f}", True, color)


def draw_hud(surf, font, world: World, pilot_name: str) -> None:
    config, lander = world.config, world.lander
    lines = [
        speed_text(font, "vx", lander.vx, config.safe_vx),
        speed_text(font, "vy", lander.vy, config.safe_vy),
        font.render(f"pilot: {pilot_name}", True, TEXT),
    ]
    for i, line in enumerate(lines):
        surf.blit(line, (12, 10 + i * 22))
    hint = font.render("W/S thrust  A/D sideways  R restart  Esc quit", True, HINT)
    surf.blit(hint, (config.width - hint.get_width() - 12, 10))


def draw_banner(surf, big, font, world: World, hint: str) -> None:
    if world.outcome not in BANNERS:
        return
    text, color = BANNERS[world.outcome]
    cx, cy = world.config.width // 2, world.config.height // 3
    title = big.render(text, True, color)
    surf.blit(title, title.get_rect(center=(cx, cy)))
    sub = font.render(hint, True, TEXT)
    surf.blit(sub, sub.get_rect(center=(cx, cy + 50)))


def draw_terrain(surf, world: World) -> None:
    """Gray fill from the surface down to the bottom of the screen, with a lighter ridge line."""
    pts = list(world.terrain.points)
    h = world.config.height
    pygame.draw.polygon(surf, GROUND, pts + [(pts[-1][0], h), (pts[0][0], h)])
    pygame.draw.lines(surf, RIDGE, False, pts, 2)


def make_stars(config: Config, rng: random.Random) -> list[tuple[int, int]]:
    return [(rng.randrange(config.width), rng.randrange(config.ground_y)) for _ in range(120)]


def scene_fonts() -> tuple[pygame.font.Font, pygame.font.Font]:
    return pygame.font.SysFont("monospace", 18), pygame.font.SysFont("monospace", 48, bold=True)


def draw_scene(surf, fonts, stars, world: World, pilot) -> None:
    config = world.config
    font, _ = fonts
    surf.fill(BG, (0, 0, config.width, config.height))
    for star in stars:
        surf.set_at(star, STAR)
    draw_terrain(surf, world)
    p = world.pad
    pygame.draw.rect(surf, PAD, pygame.Rect(p.x, p.y, p.w, p.h))
    draw_flames(surf, config, world.lander, world.controls)
    draw_lander(surf, config, world.lander)
    draw_hud(surf, font, world, pilot.name)


# ---- app ------------------------------------------------------------------------
class App:
    def __init__(self, screen: pygame.Surface, config: Config) -> None:
        self.screen = screen
        self.config = config
        self.fonts = scene_fonts()
        self.small = pygame.font.SysFont(None, 22)
        self.mono = pygame.font.SysFont("monospace", 16)
        self.clock = pygame.time.Clock()
        self.rng = random.Random()
        self.stars = make_stars(config, self.rng)
        self.message = "Fly it yourself, or Train the net and Watch it."

        self.trainer = Trainer(PolicyNet(np.random.default_rng()), TRAIN, None)
        self.load_weights_if_present()
        self.bg = BackgroundTrainer(self.trainer, EVAL_EVERY)
        self.bg.start()
        self.pilots = {"Fly": HumanPilot(), "Watch": NetPilot(self.trainer)}
        self.pilot = self.pilots["Fly"]

        x, w = config.width + GAP, PANEL_W - 2 * GAP
        self.buttons = panel.button_row(["Fly", "Watch"], x, GAP, w, BTN_H, 8)
        self.buttons |= panel.button_row(["Train", "Stop", "Clear", "Save", "Load"], x, GAP + BTN_H + 8, w, BTN_H, 8)
        chart_top = GAP + 2 * BTN_H + 8 + 34
        self.chart_rect = pygame.Rect(x, chart_top, w, config.height - chart_top - 7 * panel.ROW_H - 3 * GAP)
        self.actions = {
            "Fly": lambda: self.set_pilot("Fly"),
            "Watch": lambda: self.set_pilot("Watch"),
            "Train": self.train,
            "Stop": self.stop,
            "Clear": self.clear,
            "Save": self.save,
            "Load": self.load,
        }
        self.new_round()

    # ---- rounds -----------------------------------------------------------------
    def new_round(self) -> None:
        self.world = new_world(self.config, self.rng)
        self.pilot.reset()
        self.round_frames = 0
        self.result_frames = 0

    def set_pilot(self, name: str) -> None:
        self.pilot = self.pilots[name]
        self.message = {"Fly": "You have the controls.", "Watch": "The net is flying (greedy)."}[name]
        self.new_round()

    def tick(self) -> None:
        """Advance one frame: let the pilot fly, and roll over to the next round when due."""
        if self.world.outcome is Outcome.FLYING and self.round_frames >= self.pilot.frame_limit:
            self.new_round()  # the net hovered too long; same cap as in training
        if self.world.outcome is Outcome.FLYING:
            controls = self.pilot.controls(self.world, pygame.key.get_pressed())
            self.world = step(self.world, controls, DT)
            self.round_frames += 1
            return
        self.result_frames += 1
        if self.pilot.auto_restart and self.result_frames >= RESULT_PAUSE_FRAMES:
            self.new_round()

    # ---- training actions -------------------------------------------------------
    def train(self) -> None:
        self.bg.start_run(TRAIN_STEPS_PER_CLICK)
        self.message = f"Training {TRAIN_STEPS_PER_CLICK} steps of {TRAIN.episodes_per_step} rounds."

    def stop(self) -> None:
        self.bg.halt()
        self.message = "Training stopped."

    def clear(self) -> None:
        self.bg.halt()
        self.trainer.reset(PolicyNet(np.random.default_rng()))
        self.bg.steps_done = 0
        self.message = "Fresh random net, history cleared."

    def save(self) -> None:
        self.trainer.save(WEIGHTS)
        self.message = f"Saved {WEIGHTS}"

    def load(self) -> None:
        if not WEIGHTS.exists():
            self.message = f"No {WEIGHTS} to load."
            return
        self.load_weights_if_present()

    def load_weights_if_present(self) -> None:
        if not WEIGHTS.exists():
            return
        try:
            self.trainer.load(WEIGHTS)
        except ValueError as e:
            print(e)
            self.message = f"Not loaded: {WEIGHTS} doesn't fit this net."
            return
        self.message = f"Loaded {WEIGHTS}"

    # ---- input ------------------------------------------------------------------
    def handle(self, event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.new_round()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for name, btn in self.buttons.items():
                if btn.hit(event.pos):
                    self.actions[name]()

    # ---- drawing ----------------------------------------------------------------
    def draw_panel(self) -> None:
        s, c = self.screen, self.config
        s.fill(panel.PANEL_BG, (c.width, 0, PANEL_W, c.height))
        running = self.bg.running
        for name, pilot in self.pilots.items():
            self.buttons[name].active = self.pilot is pilot
        self.buttons["Train"].active = running
        self.buttons["Train"].enabled = not running
        self.buttons["Stop"].enabled = running
        for btn in self.buttons.values():
            btn.draw(s, self.small)
        history = list(self.trainer.history)
        panel.draw_chart(s, self.small, self.chart_rect, history)
        steps = f"{self.bg.steps_done:,}" + (f" ({self.bg.remaining:,} left)" if running else "")
        stats_rect = pygame.Rect(self.chart_rect.x, self.chart_rect.bottom + 10, self.chart_rect.w, 0)
        y = panel.draw_stats(s, (self.small, self.mono), stats_rect, history, [
            ("training rounds flown", f"{self.trainer.total_episodes:,}"),
            ("training steps" + (" (running)" if running else ""), steps),
        ])
        s.blit(self.small.render(self.message, True, TEXT), (self.chart_rect.x, y + 8))

    def draw(self) -> None:
        draw_scene(self.screen, self.fonts, self.stars, self.world, self.pilot)
        self.pilot.draw_result(self.screen, self.fonts, self.world)
        self.draw_panel()

    def run(self) -> None:
        """Game loop; returns when the player closes the window or presses Esc."""
        while True:
            for event in pygame.event.get():
                if wants_quit(event):
                    return
                self.handle(event)
            self.tick()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)


def wants_quit(event) -> bool:
    return event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)


def shutdown(app: App) -> None:
    app.bg.shutdown()
    # If SDL teardown hangs anyway, force the exit after a moment.
    threading.Timer(2.0, os._exit, args=(0,)).start()
    pygame.quit()
    os._exit(0)


def main() -> None:
    # Only video and fonts: the game has no sound, and audio init can hang
    # pygame.quit() on WSLg/PulseAudio and freeze the app on close.
    pygame.display.init()
    pygame.font.init()
    # SDL turns Ctrl-C into a QUIT event; restore the default so Ctrl-C always kills.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    screen = pygame.display.set_mode((CONFIG.width + PANEL_W, CONFIG.height))
    pygame.display.set_caption("Lander")
    app = App(screen, CONFIG)
    app.run()
    shutdown(app)


if __name__ == "__main__":
    main()

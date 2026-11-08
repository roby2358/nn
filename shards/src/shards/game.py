"""Pygame front end: the rock field on the left, evolution controls and a live chart on the right."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygame

from . import art
from .config import DT, FIELD_H, FIELD_W, GENS_PER_PRESS, RUN_SECONDS
from .evolve import SWARM, WEIGHTS, BackgroundEvolver, Evolver
from .pilots import HumanPilot, Pilot
from .scenario import make_scenario
from .worlds import Worlds

FIELD = (int(FIELD_W), int(FIELD_H))
PANEL_W = 520
STATUS_H = 44
GAP = 18
BTN_H = 40
WIN_W = FIELD[0] + PANEL_W
WIN_H = FIELD[1] + STATUS_H
STATS_H = 150
SPEEDS = (1, 4, 16)
END_HOLD_FRAMES = 60    # the finished round stays on screen this long

BTN = (50, 54, 72)
BTN_HOVER = (68, 74, 98)
BTN_ACTIVE = (60, 130, 110)
BTN_DISABLED = (34, 36, 48)


@dataclass(frozen=True)
class Fonts:
    small: pygame.font.Font
    mono: pygame.font.Font


class Button:
    def __init__(self, label: str, rect: pygame.Rect) -> None:
        self.label = label
        self.rect = rect
        self.enabled = True
        self.active = False

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        color = BTN_DISABLED if not self.enabled else BTN_ACTIVE if self.active else BTN_HOVER if hover else BTN
        pygame.draw.rect(screen, color, self.rect, border_radius=6)
        t = font.render(self.label, True, art.TEXT if self.enabled else art.MUTED)
        screen.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


class Round:
    """One pilot flying one scenario, with the debris it makes."""

    def __init__(self, label: str, pilot: Pilot, seed: int, rng: np.random.Generator) -> None:
        self.label = label
        self.pilot = pilot
        self.world = Worlds([make_scenario(seed)], np.zeros(1, dtype=np.int64))
        self.shards = art.Shards(rng)
        self.hold = END_HOLD_FRAMES

    @property
    def finished(self) -> bool:
        return self.hold <= 0

    def update(self, steps: int) -> None:
        for _ in range(steps):
            self._advance()
        self.shards.update(steps * DT)

    def _advance(self) -> None:
        w = self.world
        if w.done:
            self.hold -= 1
            return
        w.step(self.pilot.act(w.observe()))
        for event in w.events:
            self.shards.burst(event)

    def draw(self, surf: pygame.Surface, glow: pygame.Surface, t: float, fonts: Fonts) -> None:
        w = self.world
        art.draw_rocks(surf, w, t)
        art.draw_shots(surf, w)
        art.draw_player(surf, w, t)
        self.shards.draw(glow)
        hud = f"{self.label}   score {w.score[0]:.0f}   wave {w.wave[0] + 1}   {RUN_SECONDS - w.steps * DT:4.1f}s"
        surf.blit(fonts.mono.render(hud, True, art.TEXT), (10, 8))


@dataclass(frozen=True)
class Mode:
    button: str                             # what the Fly/Watch button reads in this mode
    message: str                            # status line on entering the mode
    opens: Callable[[App, int], Round]      # starts a round on the given seed
    steps: Callable[[App], int]             # physics steps per frame


FLY = Mode("Watch", "Left/right turn, up thrusts, space fires, R restarts.",
           lambda app, seed: Round("you", app.human, seed, app.rng),
           lambda app: 1)
WATCH = Mode("Fly", "Watching the champion.",
             lambda app, seed: Round(f"champion, gen {app.evolver.gen}", app.evolver.champion(), seed, app.rng),
             lambda app: SPEEDS[app.speed])


def make_fonts() -> Fonts:
    mono = pygame.font.match_font("dejavusansmono,liberationmono,consolas,menlo,couriernew,freemono")
    small = pygame.font.SysFont(None, 22)
    return Fonts(small, pygame.font.Font(mono, 15) if mono else small)


class App:
    def __init__(self) -> None:
        # Only what we use: pygame.init() also opens audio, whose teardown can hang on WSLg.
        pygame.display.init()
        pygame.font.init()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Shards")
        self.fonts = make_fonts()
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()
        self.field = pygame.Surface(FIELD)
        self.glow = pygame.Surface(FIELD, pygame.SRCALPHA)
        self.stars = art.Starfield(self.rng, FIELD)
        self.human = HumanPilot()
        self.mode = WATCH
        self.speed = 0
        self.t0 = time.perf_counter()
        self.gen_clock = (0, time.perf_counter(), 0.0)   # (gen, when, gens/sec)

        self.evolver = Evolver(int(self.rng.integers(2 ** 31)))
        self.message = "Evolve breeds pilots. Fly takes the ship yourself. F changes watch speed."
        if SWARM.exists() or WEIGHTS.exists():
            self.load_files()
        self.bg = BackgroundEvolver(self.evolver)
        self.bg.start()
        self.scene = self.next_round()

        names = ["Fly", "Evolve", "Stop", "Clear", "Save", "Load"]
        w = (PANEL_W - GAP * 2 - 6 * (len(names) - 1)) // len(names)
        self.buttons = {n: Button(n, pygame.Rect(FIELD[0] + GAP + i * (w + 6), GAP, w, BTN_H))
                        for i, n in enumerate(names)}
        top = GAP + BTN_H + 34
        self.chart_rect = pygame.Rect(FIELD[0] + GAP, top, PANEL_W - GAP * 2, FIELD[1] - top - STATS_H)
        self.actions = {"Fly": self.toggle_fly, "Evolve": self.evolve, "Stop": self.bg.halt,
                        "Clear": self.clear, "Save": self.save, "Load": self.load}

    def next_round(self) -> Round:
        return self.mode.opens(self, int(self.rng.integers(2 ** 31)))

    def toggle_fly(self) -> None:
        self.mode = WATCH if self.mode is FLY else FLY
        self.scene = self.next_round()
        self.message = self.mode.message

    def evolve(self) -> None:
        self.bg.start_run(GENS_PER_PRESS)
        self.message = f"Evolving up to {GENS_PER_PRESS} generations."

    def clear(self) -> None:
        self.bg.halt()
        self.evolver.reset()
        self.scene = self.next_round()
        self.message = "Fresh random population."

    def save(self) -> None:
        try:
            self.evolver.save(SWARM, WEIGHTS)
            self.message = f"Saved {SWARM} (swarm) and {WEIGHTS} (leader)."
        except OSError as e:
            self.message = f"Failed: {e}"

    def load_files(self) -> None:
        try:
            self.message = self.evolver.load(SWARM, WEIGHTS).describe()
        except (OSError, ValueError, KeyError) as e:
            self.message = f"Failed: {e}"

    def load(self) -> None:
        self.bg.halt()
        self.load_files()
        self.scene = self.next_round()

    def handle(self, ev: pygame.event.Event) -> bool:
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
            return False
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_f:
            self.speed = (self.speed + 1) % len(SPEEDS)
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
            self.scene = self.next_round()
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for name, b in self.buttons.items():
                if b.hit(ev.pos):
                    self.actions[name]()
        return True

    def steps_per_frame(self) -> int:
        return self.mode.steps(self)

    def tick(self) -> None:
        keys = pygame.key.get_pressed()
        self.human.turn = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
        self.human.thrust = float(keys[pygame.K_UP])
        self.human.fire = float(keys[pygame.K_SPACE])
        self.scene.update(self.steps_per_frame())
        if self.scene.finished:
            self.scene = self.next_round()

    def draw(self) -> None:
        self.compose(time.perf_counter() - self.t0)
        pygame.display.flip()

    def compose(self, t: float) -> None:
        """Paint the whole window for animation time t, without presenting it."""
        self.compose_field(t)
        self.screen.fill(art.PANEL_BG)
        self.screen.blit(self.field, (0, 0))
        self.draw_panel()
        msg = self.fonts.small.render(self.message, True, art.TEXT)
        self.screen.blit(msg, (GAP, FIELD[1] + (STATUS_H - msg.get_height()) // 2))

    def compose_field(self, t: float) -> None:
        """Paint the playing field alone for animation time t."""
        self.stars.draw(self.field)
        self.glow.fill((0, 0, 0, 0))
        self.scene.draw(self.field, self.glow, t, self.fonts)
        self.field.blit(self.glow, (0, 0))
        if self.steps_per_frame() > 1:
            s = self.fonts.mono.render(f"x{self.steps_per_frame()}", True, art.MUTED)
            self.field.blit(s, (FIELD[0] - s.get_width() - 10, 8))

    def draw_panel(self) -> None:
        self.buttons["Fly"].label = self.mode.button
        self.buttons["Evolve"].active = self.bg.running
        self.buttons["Stop"].enabled = self.bg.running
        for b in self.buttons.values():
            b.draw(self.screen, self.fonts.small)
        hist = list(self.evolver.history)
        x = self.chart_rect.left
        for label, color in (("best", art.BEST), ("mean", art.MEAN)):
            pygame.draw.line(self.screen, color, (x, self.chart_rect.top - 12), (x + 18, self.chart_rect.top - 12), 3)
            t = self.fonts.small.render(label, True, art.MUTED)
            self.screen.blit(t, (x + 24, self.chart_rect.top - 20))
            x += 40 + t.get_width()
        art.draw_chart(self.screen, self.chart_rect, [([s.mean for s in hist], art.MEAN),
                                                      ([s.best for s in hist], art.BEST)], self.fonts.small)
        self.draw_stats(hist)

    def draw_stats(self, hist: list) -> None:
        gen, when, rate = self.gen_clock
        now = time.perf_counter()
        if now - when > 1.0:
            rate = (len(hist) - gen) / (now - when)
            self.gen_clock = (len(hist), now, rate)
        rows = [(f"generation   {len(hist):6d}    {rate:5.1f} gen/s", art.TEXT)]
        if hist:
            s = hist[-1]
            rows += [(f"best         {s.best:6.1f}    {s.best_hits:5.1f} hits  {s.best_accuracy:4.0%} acc", art.BEST),
                     (f"mean         {s.mean:6.1f}", art.MEAN),
                     (f"survived     {s.best_seconds:6.1f}s", art.TEXT),
                     (f"died         {s.death_rate:6.0%}", art.DANGER)]
        y = self.chart_rect.bottom + 14
        for text, color in rows:
            self.screen.blit(self.fonts.mono.render(text, True, color), (self.chart_rect.left, y))
            y += 22

    def run(self) -> None:
        while all(self.handle(ev) for ev in pygame.event.get()):
            self.tick()
            self.draw()
            self.clock.tick(60)
        self.bg.shutdown()
        pygame.display.quit()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()

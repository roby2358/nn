"""Pygame front end: the battlefield on the left, evolution controls and a live chart on the right."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygame

from . import art
from .config import CMD_LEFT, CMD_RIGHT, CMD_SHOOT, DT, FIELD_H, FIELD_W, FREE_PLAY_STEPS, GENS_PER_PRESS, RUN_STEPS
from .evolve import SWARM, WEIGHTS, BackgroundEvolver, Evolver
from .pilots import Brains, Commandeered, HumanDriver
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
END_HOLD_STEPS = 60     # the finished battle stays on screen this long
YOUR_TANK = 0           # the first green seat
NO_RING = np.zeros(0, dtype=np.int64)
YOUR_RING = np.array([YOUR_TANK])

BTN = (58, 58, 50)
BTN_HOVER = (78, 78, 66)
BTN_ACTIVE = (70, 130, 80)
BTN_DISABLED = (40, 40, 35)


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
    """One battle on screen. It ends with the step limit, a wiped-out team, or the death of a ringed tank."""

    def __init__(self, label: str, pilots: Brains | Commandeered, seed: int, step_limit: float,
                 ringed: np.ndarray) -> None:
        self.label = label
        self.pilots = pilots
        self.ringed = ringed
        self.world = Worlds([make_scenario(seed)], step_limit)
        self.field = art.Battlefield(seed)
        self.hold = END_HOLD_STEPS

    @property
    def ended(self) -> bool:
        w = self.world
        return w.done or bool(w.over[0]) or not w.alive[0, self.ringed].all()

    @property
    def finished(self) -> bool:
        return self.hold <= 0

    def update(self, steps: int) -> None:
        for _ in range(steps):
            self._advance()
        self.field.rings.update(steps * DT)

    def _advance(self) -> None:
        if self.ended:
            self.hold -= 1
            return
        w = self.world
        w.step(self.pilots.act(w.observe()))
        self.field.rings.add(w.blasts)

    def draw(self) -> None:
        self.field.draw(self.world, self.label)
        self.field.ring(self.world, self.ringed)


@dataclass(frozen=True)
class Mode:
    button: str                             # what the Drive/Watch button reads in this mode
    message: str                            # status line on entering the mode
    opens: Callable[[App, int], Round]      # starts a battle on the given seed
    steps: Callable[[App], int]             # physics steps per frame


DRIVE = Mode("Watch", "You're the ringed green tank: arrows drive, space fires, R restarts.",
             lambda app, seed: Round("you", Commandeered(app.crew(), app.human, YOUR_TANK), seed, FREE_PLAY_STEPS,
                                     YOUR_RING),
             lambda app: 1)
WATCH = Mode("Drive", "Watching the leaders fight.",
             lambda app, seed: Round(f"gen {app.evolver.gen}", app.crew(), seed, RUN_STEPS, NO_RING),
             lambda app: SPEEDS[app.speed])


def key_treads(up: bool, down: bool, left: bool, right: bool) -> tuple[float, float]:
    """(left tread, right tread) for the held arrows. Pressing a side alone spins in place."""
    forward, turn = float(up) - float(down), float(right) - float(left)
    return float(np.clip(forward + turn, -1.0, 1.0)), float(np.clip(forward - turn, -1.0, 1.0))


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
        pygame.display.set_caption("Tenks")
        self.fonts = make_fonts()
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()
        self.human = HumanDriver()
        self.mode = WATCH
        self.speed = 0
        self.gen_clock = (0, time.perf_counter(), 0.0)   # (gen, when, gens/sec)

        self.evolver = Evolver(int(self.rng.integers(2 ** 31)))
        self.message = "Evolve breeds tanks. Drive takes one yourself. F changes watch speed."
        if SWARM.exists() or WEIGHTS.exists():
            self.load_files()
        self.bg = BackgroundEvolver(self.evolver)
        self.bg.start()
        self.scene = self.next_round()

        names = ["Drive", "Evolve", "Stop", "Clear", "Save", "Load"]
        w = (PANEL_W - GAP * 2 - 6 * (len(names) - 1)) // len(names)
        self.buttons = {n: Button(n, pygame.Rect(FIELD[0] + GAP + i * (w + 6), GAP, w, BTN_H))
                        for i, n in enumerate(names)}
        top = GAP + BTN_H + 34
        self.chart_rect = pygame.Rect(FIELD[0] + GAP, top, PANEL_W - GAP * 2, FIELD[1] - top - STATS_H)
        self.actions = {"Drive": self.toggle_drive, "Evolve": self.evolve, "Stop": self.bg.halt,
                        "Clear": self.clear, "Save": self.save, "Load": self.load}

    def crew(self) -> Brains:
        """The current leaders, one per tank, in shuffled seats."""
        return Brains(self.rng.permutation(self.evolver.leaders()))

    def next_round(self) -> Round:
        return self.mode.opens(self, int(self.rng.integers(2 ** 31)))

    def toggle_drive(self) -> None:
        self.mode = WATCH if self.mode is DRIVE else DRIVE
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
        if not (SWARM.exists() or WEIGHTS.exists()):
            self.message = f"Nothing to load: no {SWARM} or {WEIGHTS} here yet."
            return
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

    def read_keys(self) -> None:
        keys = pygame.key.get_pressed()
        left, right = key_treads(keys[pygame.K_UP], keys[pygame.K_DOWN], keys[pygame.K_LEFT], keys[pygame.K_RIGHT])
        self.human.command[[CMD_LEFT, CMD_RIGHT, CMD_SHOOT]] = left, right, float(keys[pygame.K_SPACE])

    def tick(self) -> None:
        self.read_keys()
        self.scene.update(self.steps_per_frame())
        if self.scene.finished:
            self.scene = self.next_round()

    def draw(self) -> None:
        self.compose()
        pygame.display.flip()

    def compose(self) -> None:
        """Paint the whole window, without presenting it."""
        self.compose_field()
        self.screen.fill(art.PANEL_BG)
        self.screen.blit(self.scene.field.surf, (0, 0))
        self.draw_panel()
        msg = self.fonts.small.render(self.message, True, art.TEXT)
        self.screen.blit(msg, (GAP, FIELD[1] + (STATUS_H - msg.get_height()) // 2))

    def compose_field(self) -> None:
        self.scene.draw()
        if self.steps_per_frame() > 1:
            s = self.fonts.mono.render(f"x{self.steps_per_frame()}", True, art.MUTED)
            self.scene.field.surf.blit(s, (FIELD[0] - s.get_width() - 10, 8))

    def draw_panel(self) -> None:
        self.buttons["Drive"].label = self.mode.button
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
            rows += [(f"best         {s.best:6.1f}    {s.best_kills:5.2f} kills", art.BEST),
                     (f"mean         {s.mean:6.1f}    {s.kills:5.2f} kills", art.MEAN),
                     (f"friendly     {s.friendly_kills:6.2f}    {s.near_misses:5.2f} near", art.DANGER),
                     (f"shots        {s.shots:6.1f}    {s.survived:5.1f}s alive", art.TEXT),
                     (f"decided      {s.decided:6.0%}", art.TEXT)]
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

"""Pygame front end: the starfield on the left, evolution controls and a live chart on the right."""

from __future__ import annotations

import signal
import time
from pathlib import Path

import numpy as np
import pygame

from . import art
from .config import FIELD_H, FIELD_W, GENS_PER_PRESS
from .course import make_course
from .evolve import SWARM, WEIGHTS, BackgroundEvolver, Evolver
from .pilots import AutoPilot, Brains, HumanPilot, load_pilot
from .scenes import FreeFlight, Fonts, Race, Racer, Replay, blit_lines

# ---- layout -------------------------------------------------------------------
FIELD = (int(FIELD_W), int(FIELD_H))
PANEL_W = 520
STATUS_H = 44
GAP = 18
BTN_H = 40
WIN_W = FIELD[0] + PANEL_W
WIN_H = FIELD[1] + STATUS_H
STATS_H = 150
SPEEDS = (1, 4, 16)
RACERS_DIR = Path("racers")

BTN = (50, 54, 72)
BTN_HOVER = (68, 74, 98)
BTN_ACTIVE = (60, 130, 110)
BTN_DISABLED = (34, 36, 48)


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


def make_fonts() -> Fonts:
    mono = pygame.font.match_font("dejavusansmono,liberationmono,consolas,menlo,couriernew,freemono")
    small = pygame.font.SysFont(None, 22)
    return Fonts(pygame.font.SysFont(None, 96), small, pygame.font.Font(mono, 15) if mono else small)


class App:
    def __init__(self) -> None:
        # Only what we use: pygame.init() also opens audio, whose teardown can hang on WSLg.
        pygame.display.init()
        pygame.font.init()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Crystal Chase")
        self.fonts = make_fonts()
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()
        self.field = pygame.Surface(FIELD)
        self.glow = pygame.Surface(FIELD, pygame.SRCALPHA)
        self.stars = art.Starfield(self.rng, FIELD)
        self.human = HumanPilot()
        self.speed = 0
        self.t0 = time.perf_counter()
        self.gen_clock = (0, time.perf_counter(), 0.0)   # (gen, when, gens/sec)

        self.evolver = Evolver(int(self.rng.integers(2 ** 31)))
        self.message = "Evolve breeds pilots. Race pits you against the champion. F changes replay speed."
        if SWARM.exists() or WEIGHTS.exists():
            self._try_load()
        self.bg = BackgroundEvolver(self.evolver)
        self.bg.start()
        self.scene: Replay | FreeFlight | Race = self.next_replay()

        names = ["Race", "Evolve", "Stop", "Clear", "Save", "Load", "Snap"]
        w = (PANEL_W - GAP * 2 - 6 * (len(names) - 1)) // len(names)
        self.buttons = {n: Button(n, pygame.Rect(FIELD[0] + GAP + i * (w + 6), GAP, w, BTN_H))
                        for i, n in enumerate(names)}
        top = GAP + BTN_H + 34
        self.chart_rect = pygame.Rect(FIELD[0] + GAP, top, PANEL_W - GAP * 2, FIELD[1] - top - STATS_H)
        self.actions = {"Race": self.toggle_race, "Evolve": self.evolve, "Stop": self.bg.halt,
                        "Clear": self.clear, "Save": self.save, "Load": self.load, "Snap": self.snap}

    # ---- scenes ---------------------------------------------------------------------
    def next_replay(self) -> Replay | FreeFlight:
        """While evolving: the newest evaluated generation, swarm and all. Otherwise: free
        flight, the champion alone on a fresh endless course."""
        latest = self.evolver.latest
        if self.wants_swarm():
            return Replay(f"generation {latest.stats.gen}", latest.courses[0], latest.genomes, self.rng)
        return FreeFlight(f"champion after gen {self.evolver.gen}", make_course(int(self.rng.integers(2 ** 31))),
                          self.evolver.champion(), self.rng)

    def wants_swarm(self) -> bool:
        return self.bg.running and self.evolver.latest is not None

    def racers(self) -> list[Racer]:
        field = [Racer("You", art.HUMAN, self.human), Racer("Champion", art.LEADER, self.evolver.champion()),
                 Racer("Autopilot", art.BOT, AutoPilot())]
        for i, path in enumerate(sorted(RACERS_DIR.glob("*.npz"))):
            field.append(Racer(path.stem, art.RIVALS[i % len(art.RIVALS)], load_pilot(path)))
        return field

    # ---- actions ----------------------------------------------------------------------
    def _try(self, action, success) -> None:
        """Run action; on success show success() (called after, so it can report new state)."""
        try:
            action()
            self.message = success()
        except (OSError, ValueError, KeyError) as e:
            self.message = f"Failed: {e}"

    def toggle_race(self) -> None:
        if isinstance(self.scene, Race):
            self.scene = self.next_replay()
            self.message = "Watching evolution."
            return
        self.scene = Race(self.racers(), make_course(int(self.rng.integers(2 ** 31))), self.rng)
        self.message = "Arrow keys: left/right to turn, up to thrust. First to touch the crystal takes it."

    def evolve(self) -> None:
        self.bg.start_run(GENS_PER_PRESS)
        self.message = f"Evolving up to {GENS_PER_PRESS} generations."

    def clear(self) -> None:
        self.bg.halt()
        self.evolver.reset()
        self.scene = self.next_replay()
        self.message = "Fresh random population."

    def save(self) -> None:
        self._try(lambda: self.evolver.save(SWARM, WEIGHTS), lambda: f"Saved {SWARM} (swarm) and {WEIGHTS} (leader).")

    def _try_load(self) -> None:
        try:
            self.message = self.evolver.load(SWARM, WEIGHTS).describe()
        except (OSError, ValueError, KeyError) as e:
            self.message = f"Failed: {e}"

    def load(self) -> None:
        self.bg.halt()
        self._try_load()
        if not isinstance(self.scene, Race):
            self.scene = self.next_replay()  # fly the loaded leader now, not after a crash

    def snap(self) -> None:
        RACERS_DIR.mkdir(exist_ok=True)
        path = RACERS_DIR / f"gen-{self.evolver.gen:04d}.npz"
        self._try(lambda: self.evolver.champion().save(path), lambda: f"Champion saved as racer {path.stem}.")

    # ---- loop ---------------------------------------------------------------------------
    def handle(self, ev: pygame.event.Event) -> bool:
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
            return False
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_f:
            self.speed = (self.speed + 1) % len(SPEEDS)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for name, b in self.buttons.items():
                if b.hit(ev.pos):
                    self.actions[name]()
        return True

    def tick(self) -> None:
        keys = pygame.key.get_pressed()
        self.human.turn = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
        self.human.thrust = int(keys[pygame.K_UP])
        racing = isinstance(self.scene, Race)
        self.scene.update(1 if racing else SPEEDS[self.speed])
        # The Watch view follows the evolver: switch as soon as evolving starts or stops.
        if not racing and (self.scene.finished or isinstance(self.scene, Replay) != self.wants_swarm()):
            self.scene = self.next_replay()

    def draw(self) -> None:
        self.compose(time.perf_counter() - self.t0)
        pygame.display.flip()

    def compose(self, t: float) -> None:
        """Paint the whole window for animation time t, without presenting it."""
        self.stars.draw(self.field, t)
        self.glow.fill((0, 0, 0, 0))
        self.scene.draw(self.field, self.glow, t, self.fonts)
        self.field.blit(self.glow, (0, 0))
        if not isinstance(self.scene, Race) and SPEEDS[self.speed] > 1:
            s = self.fonts.mono.render(f"x{SPEEDS[self.speed]}", True, art.MUTED)
            self.field.blit(s, (FIELD[0] - s.get_width() - 10, 8))
        self.screen.fill(art.PANEL_BG)
        self.screen.blit(self.field, (0, 0))
        self.draw_panel()
        msg = self.fonts.small.render(self.message, True, art.TEXT)
        self.screen.blit(msg, (GAP, FIELD[1] + (STATUS_H - msg.get_height()) // 2))

    def draw_panel(self) -> None:
        self.buttons["Race"].label = "Watch" if isinstance(self.scene, Race) else "Race"
        self.buttons["Evolve"].active = self.bg.running
        self.buttons["Stop"].enabled = self.bg.running
        for b in self.buttons.values():
            b.draw(self.screen, self.fonts.small)
        hist = list(self.evolver.history)
        legend = [("best", art.LEADER), ("mean", art.MEAN), ("autopilot", art.BOT)]
        x = self.chart_rect.left
        for label, color in legend:
            pygame.draw.line(self.screen, color, (x, self.chart_rect.top - 12), (x + 18, self.chart_rect.top - 12), 3)
            t = self.fonts.small.render(label, True, art.MUTED)
            self.screen.blit(t, (x + 24, self.chart_rect.top - 20))
            x += 40 + t.get_width()
        art.draw_chart(self.screen, self.chart_rect, [([s.mean for s in hist], art.MEAN),
                                                      ([s.bot for s in hist], art.BOT),
                                                      ([s.best for s in hist], art.LEADER)], self.fonts.small)
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
            rows += [(f"best         {s.best:6.2f}    {s.best_crystals:4.1f} crystals", art.LEADER),
                     (f"mean         {s.mean:6.2f}", art.MEAN),
                     (f"autopilot    {s.bot:6.2f}", art.BOT),
                     (f"crashed      {s.crash_rate:6.0%}", art.CRASH)]
        blit_lines(self.screen, self.fonts.mono, rows, (self.chart_rect.left, self.chart_rect.bottom + 14))

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

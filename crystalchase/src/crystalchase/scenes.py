"""What the field shows:

- Replay: re-flies one evaluated generation, every net as a ghost, the leader bright, the bot in gold
- FreeFlight: one net flying an endless course, scored in 30 s laps
- Race: named racers going for one shared crystal
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pygame

from . import art
from .config import DT, RUN_STEPS
from .course import Course
from .evolve import population_flight
from .flock import Event, Flock, shared_rule, solo_rule
from .pilots import Crew, Pilot, Seat

HOLD_FRAMES = 90          # pause on the final positions before the next scene
COUNTDOWN_FRAMES = 180
TRAIL = 45


@dataclass(frozen=True)
class Fonts:
    big: pygame.font.Font
    small: pygame.font.Font
    mono: pygame.font.Font


def blit_lines(surf: pygame.Surface, font: pygame.font.Font, lines: list[tuple[str, tuple]],
               at: tuple[int, int]) -> None:
    x, y = at
    for text, color in lines:
        surf.blit(font.render(text, True, color), (x, y))
        y += font.get_linesize()


class Scene:
    """Steps a Flock with a Crew and turns its events into particles. Once the flock is
    done it holds the final frame for HOLD_FRAMES, then reports finished."""

    def __init__(self, flock: Flock, crew: Crew, rng: np.random.Generator) -> None:
        self.flock = flock
        self.crew = crew
        self.rng = rng
        self.particles = art.Particles(rng)
        self.hold = HOLD_FRAMES

    @property
    def finished(self) -> bool:
        return self.flock.done and self.hold <= 0

    def advance(self, steps: int) -> None:
        for _ in range(steps):
            self.flock.step(self.crew.act(self.flock.observe()))
            for e in self.flock.events:
                self.on_event(e)
        self.particles.update(DT * steps)

    def holding(self) -> bool:
        """True (and counting down) once the flock is done."""
        if not self.flock.done:
            return False
        self.hold -= 1
        self.particles.update(DT)
        return True

    def on_event(self, e: Event) -> None:
        raise NotImplementedError

    def flicker(self) -> float:
        return float(self.rng.uniform(0.0, 0.9))

    def draw_hero(self, field: pygame.Surface, glow: pygame.Surface, t: float, k: int,
                  trail: deque) -> None:
        """Ship k drawn bright, with its trail and its crystal."""
        f = self.flock
        art.draw_trail(glow, list(trail), art.LEADER)
        self.particles.draw(glow)
        art.draw_crystal(field, glow, f.targets()[k], t)
        if f.alive[k]:
            art.draw_ship(field, f.pos[k], f.ang[k], art.LEADER, bool(f.thrusting[k]), self.flicker())
        art.draw_edge_marker(field, f.pos[k], art.LEADER)


class Replay(Scene):
    """One evaluated generation: every net as a ghost chasing its own crystal, the gold bot
    as an outline, and whoever currently leads drawn bright."""

    def __init__(self, title: str, course: Course, genomes: np.ndarray, rng: np.random.Generator) -> None:
        super().__init__(*population_flight(genomes, [course]), rng)
        self.title = title
        self.p = len(genomes)       # ship p is the bot
        self.trail: deque = deque(maxlen=TRAIL)
        self.leader = 0

    def update(self, steps: int) -> None:
        if self.holding():
            return
        self.advance(steps)
        self._follow_leader()
        self.trail.append(tuple(self.flock.pos[self.leader]))

    def _follow_leader(self) -> None:
        leader = int(np.argmax(self.flock.fitness()[:self.p]))
        if leader != self.leader:
            self.trail.clear()
            self.leader = leader

    def on_event(self, e: Event) -> None:
        if e.kind == "crash":
            self.particles.burst(e.pos, art.CRASH, 10, 90)
        elif e.ship in (self.leader, self.p):
            self.particles.burst(e.pos, art.CRYSTAL_CORE, 26, 160)

    def draw(self, field: pygame.Surface, glow: pygame.Surface, t: float, fonts: Fonts) -> None:
        f, bot = self.flock, self.p
        tgt = f.targets()
        for c in np.unique(tgt[:self.p][f.alive[:self.p]], axis=0):
            art.draw_faint_crystal(glow, c, t, art.CRYSTAL, 0.6)
        for k in np.flatnonzero(f.alive[:self.p]):
            art.draw_ghost(glow, f.pos[k], f.ang[k], (*art.GHOST, 90))
        if f.alive[bot]:
            art.draw_faint_crystal(glow, tgt[bot], t, art.BOT, 0.85)
            art.draw_ghost(glow, f.pos[bot], f.ang[bot], (*art.BOT, 255))
        art.draw_edge_marker(field, f.pos[bot], art.BOT)
        self.draw_hero(field, glow, t, self.leader, self.trail)
        fit, lead = f.fitness(), self.leader
        blit_lines(field, fonts.mono, [
            (self.title, art.TEXT),
            (f"t {f.seconds:5.1f}s   alive {int(f.alive[:self.p].sum()):2d}/{self.p}", art.MUTED),
            (f"leader {fit[lead]:5.2f}  ({f.crystals[lead]} crystals)", art.LEADER),
            (f"bot    {fit[bot]:5.2f}  ({f.crystals[bot]} crystals)", art.BOT),
        ], (10, 8))


class FreeFlight(Scene):
    """One pilot on an endless course. Every RUN_STEPS the lap's crystals and score reset;
    the ship and its crystal carry on. Only a crash ends it."""

    def __init__(self, title: str, course: Course, pilot: Pilot, rng: np.random.Generator) -> None:
        super().__init__(Flock([course], np.zeros(1, dtype=np.int64), solo_rule),
                         Crew([Seat(pilot, slice(0, 1))]), rng)
        self.title = title
        self.trail: deque = deque(maxlen=TRAIL)
        self.lap = 1
        self.last: tuple[int, float] | None = None   # (crystals, score) of the previous lap

    def update(self, steps: int) -> None:
        if self.holding():
            return
        for _ in range(steps):
            self.advance(1)
            if self.flock.steps >= RUN_STEPS:  # roll over at once, so the time limit never ends the flight
                self.last = (int(self.flock.crystals[0]), float(self.flock.fitness()[0]))
                self.flock.new_lap()
                self.lap += 1
        self.trail.append(tuple(self.flock.pos[0]))

    def restart_laps(self) -> None:
        """Back to lap 1 with a clean scoreboard; the ship and its crystal carry on."""
        self.flock.new_lap()
        self.lap = 1
        self.last = None

    def on_event(self, e: Event) -> None:
        if e.kind == "crash":
            self.particles.burst(e.pos, art.CRASH, 30, 120)
        else:
            self.particles.burst(e.pos, art.CRYSTAL_CORE, 26, 160)

    def draw(self, field: pygame.Surface, glow: pygame.Surface, t: float, fonts: Fonts) -> None:
        f = self.flock
        self.draw_hero(field, glow, t, 0, self.trail)
        rows = [(self.title, art.TEXT), (f"lap {self.lap}   t {f.seconds:5.1f}s", art.MUTED),
                (f"score {f.fitness()[0]:5.2f}  ({f.crystals[0]} crystals)", art.LEADER)]
        if self.last is not None:
            rows.append((f"last lap {self.last[1]:5.2f}  ({self.last[0]} crystals)", art.MUTED))
        blit_lines(field, fonts.mono, rows, (10, 8))


@dataclass(frozen=True)
class Racer:
    name: str
    color: tuple
    pilot: Pilot


class Race(Scene):
    def __init__(self, racers: list[Racer], course: Course, rng: np.random.Generator) -> None:
        n = len(racers)
        crew = Crew([Seat(r.pilot, slice(i, i + 1)) for i, r in enumerate(racers)])
        super().__init__(Flock([course], np.zeros(n, dtype=np.int64), shared_rule), crew, rng)
        self.racers = racers
        self.trails = [deque(maxlen=TRAIL) for _ in racers]
        self.countdown = COUNTDOWN_FRAMES

    def update(self, steps: int) -> None:
        if self.countdown > 0:
            self.countdown -= 1
            return
        self.advance(steps)
        for trail, pos, alive in zip(self.trails, self.flock.pos, self.flock.alive):
            if alive:
                trail.append(tuple(pos))

    def on_event(self, e: Event) -> None:
        color = self.racers[e.ship].color
        if e.kind == "crash":
            self.particles.burst(e.pos, art.CRASH, 40, 150)
            self.particles.burst(e.pos, color, 20, 80)
        else:
            self.particles.burst(e.pos, art.CRYSTAL_CORE, 30, 170)
            self.particles.burst(e.pos, color, 16, 110)

    def standings(self) -> list[int]:
        return sorted(range(len(self.racers)), key=lambda k: -self.flock.score[k])

    def draw(self, field: pygame.Surface, glow: pygame.Surface, t: float, fonts: Fonts) -> None:
        f = self.flock
        for trail, r in zip(self.trails, self.racers):
            art.draw_trail(glow, list(trail), r.color)
        self.particles.draw(glow)
        art.draw_crystal(field, glow, f.targets()[0], t)
        for k, r in enumerate(self.racers):
            if f.alive[k]:
                art.draw_ship(field, f.pos[k], f.ang[k], r.color, bool(f.thrusting[k]), self.flicker())
            art.draw_edge_marker(field, f.pos[k], r.color)
        rows = [(f"RACE  {f.seconds:5.1f}s", art.TEXT)]
        rows += [(f"{self.racers[k].name:<10} {f.crystals[k]:2d}  {f.score[k]:5.2f}"
                  + ("" if f.alive[k] else "  crashed"), self.racers[k].color) for k in self.standings()]
        blit_lines(field, fonts.mono, rows, (10, 8))
        self._banner(field, fonts)

    def _banner(self, field: pygame.Surface, fonts: Fonts) -> None:
        center = (field.get_width() // 2, field.get_height() // 2)
        if self.countdown > 0:
            n = fonts.big.render(str(self.countdown // 60 + 1), True, art.TEXT)
            field.blit(n, n.get_rect(center=center))
            return
        if not self.flock.done:
            return
        win = self.racers[self.standings()[0]]
        msg = fonts.big.render(f"{win.name} wins!", True, win.color)
        field.blit(msg, msg.get_rect(center=center))

"""Everything that puts pixels on the video."""

from __future__ import annotations

import math

import numpy as np
import pygame

from .config import DT, FIELD_H, FIELD_W, SPLASH_R, TANK_R, TANKS
from .worlds import GREEN, Blast, Worlds

GROUND = (46, 44, 36)
GROUND_FLECK = (58, 55, 45)
TEXT = (232, 232, 222)
WRECK = (74, 70, 62)
WRECK_DARK = (52, 49, 43)
SHELL = (255, 240, 170)
BLAST_RIM = (255, 180, 90)
BLAST_FILL = (255, 130, 50)
PANEL_BG = (30, 30, 26)
MUTED = (150, 148, 134)
CHART_BG = (20, 20, 17)
CHART_GRID = (46, 46, 40)
BEST = (150, 220, 150)
MEAN = (82, 130, 214)
DANGER = (255, 110, 80)

# (hull, treads, turret) by team
GREEN_TANK = ((86, 168, 92), (40, 80, 46), (150, 220, 150))
BLUE_TANK = ((82, 130, 214), (36, 60, 110), (150, 190, 245))
WRECK_TANK = (WRECK, WRECK_DARK, WRECK)

BLAST_LIFE = 0.35          # seconds a blast stays on screen
SHELL_SIDE = 2             # a shell is a square dot this many pixels across
HUD_MARGIN = 10


def _at(x: float, y: float, c: float, s: float, along: float, across: float) -> tuple[float, float]:
    return x + c * along - s * across, y + s * along + c * across


def _box(x: float, y: float, c: float, s: float, along: tuple[float, float], across: tuple[float, float]) -> list:
    return [_at(x, y, c, s, along[i], across[j]) for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))]


def draw_tank(surf: pygame.Surface, x: float, y: float, ang: float, colors: tuple) -> None:
    hull, treads, turret = colors
    c, s = math.cos(ang), math.sin(ang)
    length, width = TANK_R * 1.15, TANK_R * 0.85
    for side in (-1.0, 1.0):
        across = (side * width * 0.6, side * width * 1.15)
        pygame.draw.polygon(surf, treads, _box(x, y, c, s, (-length, length), across))
    pygame.draw.polygon(surf, hull, _box(x, y, c, s, (-length * 0.9, length * 0.9), (-width * 0.65, width * 0.65)))
    pygame.draw.line(surf, turret, (x, y), _at(x, y, c, s, TANK_R * 1.7, 0.0), 3)
    pygame.draw.circle(surf, turret, (x, y), TANK_R * 0.45)


class BlastRings:
    """Each blast shown at the size of its real splash, fading out."""

    def __init__(self) -> None:
        self.rings: list[list] = []   # [x, y, life left]

    def add(self, blasts: list[Blast]) -> None:
        self.rings.extend([*b.pos, BLAST_LIFE] for b in blasts)

    def update(self, dt: float) -> None:
        for ring in self.rings:
            ring[2] -= dt
        self.rings = [r for r in self.rings if r[2] > 0.0]

    def draw(self, surf: pygame.Surface) -> None:
        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        for x, y, life in self.rings:
            fade = life / BLAST_LIFE
            pygame.draw.circle(overlay, (*BLAST_FILL, int(110 * fade)), (x, y), SPLASH_R)
            pygame.draw.circle(overlay, (*BLAST_RIM, int(230 * fade)), (x, y), SPLASH_R, 2)
        surf.blit(overlay, (0, 0))


class Battlefield:
    """Draws world 0 of a Worlds onto one surface."""

    def __init__(self, seed: int) -> None:
        self.surf = pygame.Surface((int(FIELD_W), int(FIELD_H)))
        self.ground = self._make_ground(seed)
        self.rings = BlastRings()
        self.font = pygame.font.Font(None, 26)

    def _make_ground(self, seed: int) -> pygame.Surface:
        rng = np.random.default_rng(seed)
        ground = pygame.Surface(self.surf.get_size())
        ground.fill(GROUND)
        for _ in range(900):
            pos = rng.uniform(0.0, FIELD_W), rng.uniform(0.0, FIELD_H)
            pygame.draw.circle(ground, GROUND_FLECK, pos, float(rng.uniform(1.0, 3.0)))
        return ground

    def draw(self, worlds: Worlds, label: str) -> None:
        """Paint the battle as it stands. The caller feeds `rings` as it steps the world,
        because a frame can cover several steps and each step replaces worlds.blasts."""
        self.surf.blit(self.ground, (0, 0))
        self._draw_tanks(worlds, np.flatnonzero(~worlds.alive[0]), lambda tank: WRECK_TANK)
        self._draw_tanks(worlds, np.flatnonzero(worlds.alive[0]), lambda tank: GREEN_TANK if GREEN[tank] else BLUE_TANK)
        self._draw_shells(worlds)
        self.rings.draw(self.surf)
        self._draw_hud(worlds, label)

    def _draw_tanks(self, worlds: Worlds, tanks: np.ndarray, colors_of) -> None:
        for tank in tanks:
            x, y = map(float, worlds.pos[0, tank])
            draw_tank(self.surf, x, y, float(worlds.ang[0, tank]), colors_of(tank))

    def _draw_shells(self, worlds: Worlds) -> None:
        for shell in range(TANKS):
            if worlds.shell_ttl[0, shell] <= 0.0:
                continue
            x, y = worlds.shell_pos[0, shell]
            self.surf.fill(SHELL, (round(x) - SHELL_SIDE // 2, round(y) - SHELL_SIDE // 2, SHELL_SIDE, SHELL_SIDE))

    def _draw_hud(self, worlds: Worlds, label: str) -> None:
        green, blue = int(worlds.alive[0, GREEN].sum()), int(worlds.alive[0, ~GREEN].sum())
        line = f"green {green}   blue {blue}   {worlds.steps * DT:4.1f}s   {label}"
        self.surf.blit(self.font.render(line, True, TEXT), (HUD_MARGIN, HUD_MARGIN))

    def ring(self, worlds: Worlds, tanks: np.ndarray) -> None:
        """Circle these tanks, while they live, so that you can find yours in the crowd."""
        for tank in tanks[worlds.alive[0, tanks]]:
            x, y = map(float, worlds.pos[0, tank])
            pygame.draw.circle(self.surf, TEXT, (x, y), TANK_R * 2, 2)


def smooth(values: list[float], keep: float) -> list[float]:
    """Exponential moving average of the series."""
    out, avg = [], values[0]
    for v in values:
        avg = keep * avg + (1.0 - keep) * v
        out.append(avg)
    return out


def faded(color: tuple) -> tuple:
    return tuple(c // 3 for c in color)


def draw_chart(surf: pygame.Surface, rect: pygame.Rect, series: list[tuple[list[float], tuple]],
               font: pygame.font.Font) -> None:
    """Each series drawn raw and faint, with its moving average bold on top. One shared set of axes."""
    pygame.draw.rect(surf, CHART_BG, rect, border_radius=6)
    n = max(len(s[0]) for s in series)
    if n < 2:
        msg = font.render("Press Evolve to start a run of generations.", True, MUTED)
        surf.blit(msg, msg.get_rect(center=rect.center))
        return
    top = max(0.0, *(max(s[0]) for s in series)) * 1.1 or 1.0
    inner = rect.inflate(-16, -16)
    grid = [(inner.bottom - frac * inner.height, top * frac) for frac in (0.25, 0.5, 0.75)]
    for y, _ in grid:
        pygame.draw.line(surf, CHART_GRID, (inner.left, y), (inner.right, y))

    def points(values: list[float]) -> list[tuple[float, float]]:
        return [(inner.left + i * inner.width / (n - 1), inner.bottom - max(v, 0.0) / top * inner.height)
                for i, v in enumerate(values)]

    for values, color in series:
        pygame.draw.lines(surf, faded(color), False, points(values), 1)
    for values, color in series:
        pygame.draw.lines(surf, color, False, points(smooth(values, 0.95)), 2)
    for y, value in grid:
        surf.blit(font.render(f"{value:.0f}", True, MUTED), (inner.left + 2, y - 16))

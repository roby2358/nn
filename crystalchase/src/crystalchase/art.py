"""Drawing only: starfield, ships, crystals, particles, edge markers, the chart.
Nothing here knows about evolution or races; it draws what it is handed."""

from __future__ import annotations

import math

import numpy as np
import pygame

from .config import CRYSTAL_R, FIELD_H, FIELD_W, SHIP_R

SHIP_DRAW_R = SHIP_R / 2   # drawn at half the physical size; pickups still use SHIP_R

# ---- palette --------------------------------------------------------------------
SPACE = (6, 8, 18)
PANEL_BG = (22, 24, 34)
TEXT = (230, 232, 240)
MUTED = (140, 146, 165)
CRYSTAL = (120, 240, 255)
CRYSTAL_CORE = (230, 255, 255)
FLAME = (255, 170, 60)
GHOST = (150, 190, 255)
LEADER = (90, 230, 255)
BOT = (255, 210, 80)
HUMAN = (120, 255, 140)
MEAN = (70, 120, 160)
CRASH = (255, 90, 70)
RIVALS = [(255, 110, 220), (255, 140, 70), (180, 130, 255), (120, 255, 220), (255, 255, 150)]
NEBULA = [(80, 40, 140), (20, 70, 140), (120, 30, 80), (30, 100, 110)]


class Starfield:
    """Three drifting layers of twinkling stars over a soft, pre-rendered nebula."""

    def __init__(self, rng: np.random.Generator, size: tuple[int, int]) -> None:
        self.size = size
        n = 260
        self.xy = rng.random((n, 2)) * size
        self.layer = rng.integers(0, 3, n)
        self.phase = rng.random(n) * math.tau
        self.speed = np.array([2.0, 5.0, 11.0])[self.layer]
        self.nebula = self._nebula(rng)

    def _nebula(self, rng: np.random.Generator) -> pygame.Surface:
        """Soft clouds: rings at 1/8 size with alpha rising inward, smooth-scaled up so no
        ring edges survive."""
        k = 8
        small = (self.size[0] // k, self.size[1] // k)
        glow = pygame.Surface(small, pygame.SRCALPHA)
        for _ in range(7):
            color = NEBULA[int(rng.integers(len(NEBULA)))]
            cx, cy = rng.random(2) * small
            r0 = rng.uniform(120, 300) / k
            for i in range(14):
                pygame.draw.circle(glow, (*color, 3 + 2 * i), (cx, cy), r0 * (1 - i / 14))
        surf = pygame.Surface(self.size)
        surf.fill(SPACE)
        surf.blit(pygame.transform.smoothscale(glow, self.size), (0, 0))
        return surf

    def draw(self, screen: pygame.Surface, t: float) -> None:
        screen.blit(self.nebula, (0, 0))
        x = (self.xy[:, 0] - self.speed * t) % self.size[0]
        tw = 0.55 + 0.45 * np.sin(self.phase + t * (1.0 + self.layer))
        for xi, yi, layer, b in zip(x, self.xy[:, 1], self.layer, tw):
            v = int(70 + 60 * layer * b + 50 * b)
            pygame.draw.circle(screen, (v, v, min(255, v + 30)), (xi, yi), 0.6 + 0.5 * layer)


def _ship_points(pos: np.ndarray, ang: float, scale: float) -> list[tuple[float, float]]:
    c, s = math.cos(ang), math.sin(ang)
    shape = [(1.6, 0.0), (-1.0, -0.9), (-0.55, 0.0), (-1.0, 0.9)]
    return [(pos[0] + scale * (x * c - y * s), pos[1] + scale * (x * s + y * c)) for x, y in shape]


def draw_ship(surf: pygame.Surface, pos: np.ndarray, ang: float, color: tuple, thrusting: bool,
              flicker: float) -> None:
    if thrusting:
        c, s = math.cos(ang), math.sin(ang)
        back = (pos[0] - c * SHIP_DRAW_R * 0.6, pos[1] - s * SHIP_DRAW_R * 0.6)
        tip = (pos[0] - c * SHIP_DRAW_R * (1.8 + flicker), pos[1] - s * SHIP_DRAW_R * (1.8 + flicker))
        side = (-s * SHIP_DRAW_R * 0.45, c * SHIP_DRAW_R * 0.45)
        pygame.draw.polygon(surf, FLAME, [(back[0] + side[0], back[1] + side[1]), tip,
                                          (back[0] - side[0], back[1] - side[1])])
    pts = _ship_points(pos, ang, SHIP_DRAW_R)
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, (255, 255, 255), pts, 1)


def draw_ghost(surf: pygame.Surface, pos: np.ndarray, ang: float, rgba: tuple) -> None:
    pygame.draw.polygon(surf, rgba, _ship_points(pos, ang, SHIP_DRAW_R * 0.85), 1)


def _gem(center: np.ndarray, radius: float, spin: float) -> list[tuple[float, float]]:
    pts = []
    for i in range(6):
        a = spin + i * math.tau / 6
        r = radius * (1.0 if i % 2 == 0 else 0.62)
        pts.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a)))
    return pts


def draw_crystal(surf: pygame.Surface, glow: pygame.Surface, center: np.ndarray, t: float) -> None:
    pulse = 0.5 + 0.5 * math.sin(t * 4.0)
    for i in range(6):  # rising alpha inward: rings overwrite, giving a radial falloff
        pygame.draw.circle(glow, (*CRYSTAL, int((6 + 9 * i) * (0.7 + 0.3 * pulse))), center,
                           CRYSTAL_R * (3.0 - 0.3 * i))
    pts = _gem(center, CRYSTAL_R, t * 1.3)
    pygame.draw.polygon(surf, CRYSTAL, pts)
    pygame.draw.polygon(surf, CRYSTAL_CORE, pts, 2)
    for p in pts[::2]:
        pygame.draw.line(surf, CRYSTAL_CORE, center, p, 1)


def draw_faint_crystal(glow: pygame.Surface, center: np.ndarray, t: float, color: tuple, scale: float) -> None:
    pygame.draw.polygon(glow, (*color, 120), _gem(center, CRYSTAL_R * scale, t * 1.3), 1)


class Particles:
    """Sparks for pickups and debris for crashes: position, velocity, life, color."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.items: list[list] = []

    def burst(self, pos: tuple[float, float], color: tuple, count: int, speed: float) -> None:
        for _ in range(count):
            a = self.rng.uniform(0, math.tau)
            v = self.rng.uniform(0.3, 1.0) * speed
            life = self.rng.uniform(0.4, 1.0)
            self.items.append([pos[0], pos[1], v * math.cos(a), v * math.sin(a), life, life, color])

    def update(self, dt: float) -> None:
        for p in self.items:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[2] *= 0.97
            p[3] *= 0.97
            p[4] -= dt
        self.items = [p for p in self.items if p[4] > 0]

    def draw(self, glow: pygame.Surface) -> None:
        for x, y, _, _, life, full, color in self.items:
            pygame.draw.circle(glow, (*color, int(255 * life / full)), (x, y), 1.5 + 1.5 * life / full)


def draw_trail(glow: pygame.Surface, points: list[tuple[float, float]], color: tuple) -> None:
    n = len(points)
    for i in range(1, n):
        pygame.draw.line(glow, (*color, int(160 * i / n)), points[i - 1], points[i], 2)


def draw_edge_marker(surf: pygame.Surface, pos: np.ndarray, color: tuple) -> None:
    """A ship beyond the visible edge (but short of the crash wall): an arrow on the
    border pointing at it, shrinking as it gets farther out."""
    x, y = float(pos[0]), float(pos[1])
    if 0 <= x <= FIELD_W and 0 <= y <= FIELD_H:
        return
    cx, cy = min(max(x, 8.0), FIELD_W - 8), min(max(y, 8.0), FIELD_H - 8)
    ang = math.atan2(y - cy, x - cx)
    size = max(4.0, 12.0 - math.hypot(x - cx, y - cy) / 15.0)
    pts = [(cx + size * math.cos(ang + d), cy + size * math.sin(ang + d)) for d in (0.0, 2.5, -2.5)]
    pygame.draw.polygon(surf, color, pts)


def smooth(values: list[float], keep: float) -> list[float]:
    """Exponential moving average; each generation flies new courses, so raw scores are noisy."""
    out, acc = [], values[0] if values else 0.0
    for v in values:
        acc = keep * acc + (1 - keep) * v
        out.append(acc)
    return out


def faded(color: tuple) -> tuple:
    return tuple(int(c * 0.35 + 14 * 0.65) for c in color)


def draw_chart(surf: pygame.Surface, rect: pygame.Rect, series: list[tuple[list[float], tuple]],
               font: pygame.font.Font) -> None:
    """Each series drawn raw and faint, with its moving average bold on top.
    All share one x (generation) and y (score) axis."""
    pygame.draw.rect(surf, (14, 16, 24), rect, border_radius=6)
    n = max(len(s[0]) for s in series)
    if n < 2:
        msg = font.render("Press Evolve to start a run of generations.", True, MUTED)
        surf.blit(msg, msg.get_rect(center=rect.center))
        return
    top = max(max(s[0]) for s in series) * 1.1 or 1.0
    inner = rect.inflate(-16, -16)
    grid = [(inner.bottom - frac * inner.height, top * frac) for frac in (0.25, 0.5, 0.75)]
    for y, _ in grid:
        pygame.draw.line(surf, (34, 38, 52), (inner.left, y), (inner.right, y))

    def points(values: list[float]) -> list[tuple[float, float]]:
        return [(inner.left + i * inner.width / (n - 1), inner.bottom - v / top * inner.height)
                for i, v in enumerate(values)]

    for values, color in series:
        pygame.draw.lines(surf, faded(color), False, points(values), 1)
    for values, color in series:
        pygame.draw.lines(surf, color, False, points(smooth(values, 0.95)), 2)
    for y, value in grid:
        surf.blit(font.render(f"{value:.0f}", True, MUTED), (inner.left + 2, y - 16))

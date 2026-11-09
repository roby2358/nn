"""Everything that puts pixels on the screen, apart from the panel's layout."""

from __future__ import annotations

import math

import numpy as np
import pygame

from .config import FIELD_H, FIELD_W, MOOD_PASSION, MOOD_PATIENCE, RADIUS, ROCK_SLOTS, SHIP_R, TAU
from .worlds import Event, Worlds

SPACE = (7, 8, 20)
PANEL_BG = (22, 24, 34)
TEXT = (230, 232, 240)
MUTED = (140, 146, 165)
IMPATIENT = (255, 255, 255)
PATIENT = (255, 140, 40)
CALM_HALO = (90, 110, 150)
PASSION_HALO = (255, 40, 30)
SHIP_TRIM = (255, 120, 90)
FLAME = (255, 170, 60)
FLAME_CORE = (255, 240, 190)
SHOT = (255, 250, 180)
BEST = (120, 240, 255)
MEAN = (70, 120, 160)
DANGER = (255, 90, 70)
HIT = (200, 170, 255)

# (bright facet, dark facet, edge), indexed by rock size
GEMS = [None,
        ((150, 245, 235), (45, 130, 140), (215, 255, 250)),
        ((130, 175, 255), (45, 70, 165), (205, 220, 255)),
        ((190, 135, 255), (72, 40, 150), (232, 210, 255))]
FACETS = 8
FLAME_FLICKER_HZ = 19.0
FLAME_HALF_WIDTH = 0.45 * SHIP_R
FLAME_CORE_HALF_WIDTH = 0.225 * SHIP_R
TRIM_R = round(0.36 * SHIP_R)
HALO_LONG = round(3.4 * SHIP_R)     # half the halo's length, along the ship's heading
HALO_SHORT = round(2.2 * SHIP_R)
HALO_EDGE_PX = 2
HALO_FILL_ALPHA = (20, 120)         # calm and at full passion
HALO_EDGE_ALPHA = (70, 235)
SHOT_TAIL = 0.64 * SHIP_R
SHOT_TIP = 0.27 * SHIP_R
DEBRIS_BASE = 0.18 * SHIP_R
DEBRIS_GROWTH = 0.45 * SHIP_R     # extra debris radius per second of life left
ROCK_EDGE_PX = 1
SHOT_WIDTH_PX = 2
BURST_COLOR = {"break": HIT, "crash": DANGER}
BURST_PER_SIZE = {"break": 6, "crash": 12}


def _shard_outlines() -> np.ndarray:
    """(ROCK_SLOTS, FACETS, 2): each slot's jagged outline, as angle and radius fraction."""
    rng = np.random.default_rng(7)
    base = np.arange(FACETS) * TAU / FACETS
    angles = base + rng.uniform(-0.22, 0.22, (ROCK_SLOTS, FACETS))
    return np.stack([angles, rng.uniform(0.7, 1.05, (ROCK_SLOTS, FACETS))], axis=2)


OUTLINES = _shard_outlines()
SPINS = np.random.default_rng(11).uniform(-0.9, 0.9, ROCK_SLOTS)


class Starfield:
    def __init__(self, rng: np.random.Generator, size: tuple[int, int]) -> None:
        self.sky = pygame.Surface(size)
        self.sky.fill(SPACE)
        xs, ys, shades = rng.integers(0, size[0], 260), rng.integers(0, size[1], 260), rng.integers(50, 200, 260)
        for x, y, shade in zip(xs, ys, shades):
            self.sky.set_at((int(x), int(y)), (int(shade), int(shade), min(255, int(shade) + 30)))

    def draw(self, surf: pygame.Surface) -> None:
        surf.blit(self.sky, (0, 0))


class Shards:
    """Short-lived triangles flung out when a rock breaks or the ship dies."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.bits: list[list] = []   # [x, y, vx, vy, life, spin, color]

    def burst(self, event: Event) -> None:
        for _ in range(BURST_PER_SIZE[event.kind] * event.size):
            angle, speed = self.rng.uniform(0.0, TAU), self.rng.uniform(40.0, 260.0)
            self.bits.append([*event.pos, math.cos(angle) * speed, math.sin(angle) * speed,
                              self.rng.uniform(0.3, 0.8), self.rng.uniform(0.0, TAU), BURST_COLOR[event.kind]])

    def update(self, dt: float) -> None:
        for b in self.bits:
            b[0] += b[2] * dt
            b[1] += b[3] * dt
            b[4] -= dt
            b[5] += 8.0 * dt
        self.bits = [b for b in self.bits if b[4] > 0.0]

    def draw(self, glow: pygame.Surface) -> None:
        for x, y, _, _, life, spin, color in self.bits:
            r = DEBRIS_BASE + DEBRIS_GROWTH * life
            pts = [(x + r * math.cos(spin + k * 2.4), y + r * math.sin(spin + k * 2.4)) for k in range(3)]
            pygame.draw.polygon(glow, (*color, min(255, int(life * 420))), pts)


def _draw_shard(surf: pygame.Surface, x: float, y: float, slot: int, size: int, t: float) -> None:
    angles, fractions = OUTLINES[slot].T
    spin = t * SPINS[slot]
    r = RADIUS[size] * fractions
    pts = np.stack([x + r * np.cos(angles + spin), y + r * np.sin(angles + spin)], axis=1).tolist()
    core = (x + RADIUS[size] * 0.18 * math.cos(spin + slot), y + RADIUS[size] * 0.18 * math.sin(spin + slot))
    bright, dark, edge = GEMS[size]
    for k in range(FACETS):
        pygame.draw.polygon(surf, bright if k % 2 == 0 else dark, [core, pts[k], pts[(k + 1) % FACETS]])
    pygame.draw.polygon(surf, edge, pts, ROCK_EDGE_PX)


def _echoes(x: float, y: float, reach: float) -> list[tuple[float, float]]:
    """Every place a wrapped thing must be drawn so that it slides off one edge and onto the other."""
    xs = [x] + [x + d for d in (-FIELD_W, FIELD_W) if -reach < x + d < FIELD_W + reach]
    ys = [y] + [y + d for d in (-FIELD_H, FIELD_H) if -reach < y + d < FIELD_H + reach]
    return [(ex, ey) for ex in xs for ey in ys]


def draw_rocks(surf: pygame.Surface, world: Worlds, t: float) -> None:
    for slot in np.flatnonzero(world.rock_size[0] > 0):
        size = int(world.rock_size[0, slot])
        x, y = world.rock_pos[0, slot]
        for ex, ey in _echoes(float(x), float(y), RADIUS[size] * 1.1):
            _draw_shard(surf, ex, ey, int(slot), size, t)


def draw_shots(surf: pygame.Surface, world: Worlds) -> None:
    for slot in np.flatnonzero(world.shot_ttl[0] > 0.0):
        x, y = world.shot_pos[0, slot]
        vx, vy = world.shot_vel[0, slot] / np.hypot(*world.shot_vel[0, slot])
        pygame.draw.line(surf, SHOT, (x - vx * SHOT_TAIL, y - vy * SHOT_TAIL), (x + vx * SHOT_TIP, y + vy * SHOT_TIP),
                         SHOT_WIDTH_PX)


def _at(x: float, y: float, c: float, s: float, along: float, across: float) -> tuple[float, float]:
    return x + c * along - s * across, y + s * along + c * across


def _draw_flame(surf: pygame.Surface, x: float, y: float, c: float, s: float, thrust: float, t: float) -> None:
    """Length follows the thrust command; a fast flicker keeps it alive."""
    flicker = 0.78 + 0.22 * math.sin(t * FLAME_FLICKER_HZ * TAU)
    reach = SHIP_R * (0.8 + 2.2 * thrust * flicker)
    rear = -SHIP_R * 0.7
    pygame.draw.polygon(surf, FLAME, [_at(x, y, c, s, rear, -FLAME_HALF_WIDTH), _at(x, y, c, s, rear, FLAME_HALF_WIDTH),
                                      _at(x, y, c, s, rear - reach, 0)])
    pygame.draw.polygon(surf, FLAME_CORE, [_at(x, y, c, s, rear, -FLAME_CORE_HALF_WIDTH),
                                           _at(x, y, c, s, rear, FLAME_CORE_HALF_WIDTH),
                                           _at(x, y, c, s, rear - reach * 0.55, 0)])


def blend(low: tuple, high: tuple, frac: float) -> tuple:
    """Each value `frac` of the way from low to high, holding at the ends."""
    frac = min(max(frac, 0.0), 1.0)
    return tuple(round(a + (b - a) * frac) for a, b in zip(low, high))


def _alpha(ends: tuple[int, int], passion: float) -> int:
    return blend(ends[:1], ends[1:], passion)[0]


def ship_color(patience: float) -> tuple:
    """White at no patience (-1), orange at full patience (+1)."""
    return blend(IMPATIENT, PATIENT, (patience + 1.0) / 2.0)


def passion_halo(passion: float, ang: float) -> pygame.Surface:
    """An ellipse along the ship's heading, redder and more solid the more passionate it is."""
    color = blend(CALM_HALO, PASSION_HALO, passion)
    halo = pygame.Surface((2 * HALO_LONG, 2 * HALO_SHORT), pygame.SRCALPHA)
    pygame.draw.ellipse(halo, (*color, _alpha(HALO_FILL_ALPHA, passion)), halo.get_rect())
    pygame.draw.ellipse(halo, (*color, _alpha(HALO_EDGE_ALPHA, passion)), halo.get_rect(), HALO_EDGE_PX)
    return pygame.transform.rotate(halo, -math.degrees(ang))


def draw_ship(surf: pygame.Surface, x: float, y: float, ang: float, thrust: float, t: float, color: tuple) -> None:
    c, s = math.cos(ang), math.sin(ang)
    if thrust > 0.0:
        _draw_flame(surf, x, y, c, s, thrust, t)
    hull = [_at(x, y, c, s, SHIP_R * 1.3, 0), _at(x, y, c, s, -SHIP_R, SHIP_R * 0.85),
            _at(x, y, c, s, -SHIP_R * 0.45, 0), _at(x, y, c, s, -SHIP_R, -SHIP_R * 0.85)]
    pygame.draw.polygon(surf, color, hull)
    pygame.draw.circle(surf, SHIP_TRIM, _at(x, y, c, s, SHIP_R * 0.1, 0), TRIM_R)


def draw_player(surf: pygame.Surface, world: Worlds, t: float) -> None:
    if not world.alive[0]:
        return
    x, y = map(float, world.pos[0])
    ang = float(world.ang[0])
    passion, patience = float(world.mood[0, MOOD_PASSION]), float(world.mood[0, MOOD_PATIENCE])
    halo, color = passion_halo(passion, ang), ship_color(patience)
    for ex, ey in _echoes(x, y, max(HALO_LONG, SHIP_R * 4)):
        surf.blit(halo, halo.get_rect(center=(round(ex), round(ey))))
        draw_ship(surf, ex, ey, ang, float(world.thrust[0]), t, color)


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
    pygame.draw.rect(surf, (14, 16, 24), rect, border_radius=6)
    n = max(len(s[0]) for s in series)
    if n < 2:
        msg = font.render("Press Evolve to start a run of generations.", True, MUTED)
        surf.blit(msg, msg.get_rect(center=rect.center))
        return
    top = max(0.0, *(max(s[0]) for s in series)) * 1.1 or 1.0
    inner = rect.inflate(-16, -16)
    grid = [(inner.bottom - frac * inner.height, top * frac) for frac in (0.25, 0.5, 0.75)]
    for y, _ in grid:
        pygame.draw.line(surf, (34, 38, 52), (inner.left, y), (inner.right, y))

    def points(values: list[float]) -> list[tuple[float, float]]:
        return [(inner.left + i * inner.width / (n - 1), inner.bottom - max(v, 0.0) / top * inner.height)
                for i, v in enumerate(values)]

    for values, color in series:
        pygame.draw.lines(surf, faded(color), False, points(values), 1)
    for values, color in series:
        pygame.draw.lines(surf, color, False, points(smooth(values, 0.95)), 2)
    for y, value in grid:
        surf.blit(font.render(f"{value:.0f}", True, MUTED), (inner.left + 2, y - 16))

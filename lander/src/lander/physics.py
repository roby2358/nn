"""Pure lander simulation: immutable world state advanced by step().

Coordinates are screen pixels, y grows downward. Rects are top-left anchored.
"""
from dataclasses import dataclass, replace
from enum import Enum
import math
import random

DT = 1.0 / 60  # simulation timestep, shared by the game and training


@dataclass(frozen=True)
class Config:
    width: int
    height: int
    ground_height: int  # the terrain never dips below ground_y
    terrain_step: int  # px between terrain points
    terrain_jitter: float  # max px the terrain climbs or drops per point
    gravity: float  # px/s^2, downward
    main_thrust: float  # px/s^2, upward
    side_thrust: float  # px/s^2, lateral
    lander_size: int
    pad_height: int
    safe_vx: float  # px/s
    safe_vy: float  # px/s

    @property
    def ground_y(self) -> int:
        return self.height - self.ground_height

    @property
    def pad_width(self) -> int:
        return 3 * self.lander_size


CONFIG = Config(
    width=900,
    height=700,
    ground_height=30,
    terrain_step=12,
    terrain_jitter=14.0,
    gravity=40.0,
    main_thrust=100.0,
    side_thrust=45.0,
    lander_size=28,
    pad_height=8,
    safe_vx=20.0,
    safe_vy=40.0,
)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def overlaps(self, other: "Rect") -> bool:
        return (self.x < other.right and other.x < self.right
                and self.y < other.bottom and other.y < self.bottom)

    def within_x(self, other: "Rect") -> bool:
        return other.x <= self.x and self.right <= other.right


@dataclass(frozen=True)
class Controls:
    main: bool
    left: bool
    right: bool


NO_CONTROLS = Controls(main=False, left=False, right=False)


@dataclass(frozen=True)
class Lander:
    x: float
    y: float
    vx: float
    vy: float

    def rect(self, config: Config) -> Rect:
        return Rect(self.x, self.y, config.lander_size, config.lander_size)


class Outcome(Enum):
    FLYING = "flying"
    LANDED = "landed"
    CRASHED = "crashed"
    LOST = "lost"


@dataclass(frozen=True)
class World:
    config: Config
    lander: Lander
    pad: Rect
    terrain: "Terrain"
    outcome: Outcome
    controls: Controls  # controls applied last step (for flames)


@dataclass(frozen=True)
class Terrain:
    """The surface as a polyline of (x, y) points, left to right across the screen."""
    points: tuple[tuple[float, float], ...]

    def surface_y(self, x: float) -> float:
        """Surface height at x, interpolated between points (clamped at the screen edges)."""
        pts = self.points
        if x <= pts[0][0]:
            return pts[0][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]

    def highest_under(self, x0: float, x1: float) -> float:
        """Smallest surface y (the highest ground) anywhere between x0 and x1."""
        inside = [y for x, y in self.points if x0 < x < x1]
        return min([self.surface_y(x0), self.surface_y(x1)] + inside)


def walk_out(config: Config, start_x: float, end_x: float, y: float, top: float,
             rng: random.Random) -> list[tuple[float, float]]:
    """Jagged surface from start_x out to end_x, starting at height y and never above `top`.
    The bias is slightly downhill, so hills fall away from the pad."""
    dist = abs(end_x - start_x)
    direction = 1 if end_x > start_x else -1
    n = math.ceil(dist / config.terrain_step)
    jitter = config.terrain_jitter
    out = []
    for i in range(1, n + 1):
        y = min(max(y + rng.uniform(-jitter, jitter * 1.2), top), config.ground_y)
        out.append((start_x + direction * min(i * config.terrain_step, dist), y))
    return out


def random_terrain(config: Config, pad: Rect, rng: random.Random) -> Terrain:
    """Surface flush with the pad's base at both edges, wandering outward on each side.
    It never rises above the pad, so every approach from above stays open."""
    left = walk_out(config, pad.x, 0, pad.bottom, pad.bottom, rng)
    right = walk_out(config, pad.right, config.width, pad.bottom, pad.bottom, rng)
    return Terrain(tuple(left[::-1] + [(pad.x, pad.bottom), (pad.right, pad.bottom)] + right))


def random_pad(config: Config, rng: random.Random) -> Rect:
    margin = config.lander_size
    x = rng.uniform(margin, config.width - config.pad_width - margin)
    # Always within the bottom third, leaving room to slow down on the way in.
    y = rng.uniform(config.height * 2 / 3, config.ground_y - 2 * config.lander_size)
    return Rect(x, y, config.pad_width, config.pad_height)


def new_world(config: Config, rng: random.Random) -> World:
    lander = Lander(
        x=(config.width - config.lander_size) / 2,
        y=config.lander_size,
        vx=rng.uniform(-30, 30),
        vy=0.0,
    )
    pad = random_pad(config, rng)
    return World(config, lander, pad, random_terrain(config, pad, rng), Outcome.FLYING, NO_CONTROLS)


def acceleration(config: Config, c: Controls) -> tuple[float, float]:
    ax = config.side_thrust * (int(c.right) - int(c.left))
    ay = config.gravity - config.main_thrust * int(c.main)
    return ax, ay


def integrate(config: Config, lander: Lander, c: Controls, dt: float) -> Lander:
    ax, ay = acceleration(config, c)
    vx, vy = lander.vx + ax * dt, lander.vy + ay * dt
    return Lander(x=lander.x + vx * dt, y=lander.y + vy * dt, vx=vx, vy=vy)


def is_soft(config: Config, lander: Lander) -> bool:
    return abs(lander.vx) <= config.safe_vx and 0 <= lander.vy <= config.safe_vy


def judge_pad_contact(config: Config, before: Lander, after: Lander, pad: Rect) -> tuple[Lander, Outcome]:
    came_from_above = before.rect(config).bottom <= pad.y
    if not came_from_above:
        return after, Outcome.CRASHED
    settled = replace(after, y=pad.y - config.lander_size)
    if not settled.rect(config).within_x(pad):
        return settled, Outcome.CRASHED
    if not is_soft(config, after):
        return settled, Outcome.CRASHED
    return replace(settled, vx=0.0, vy=0.0), Outcome.LANDED


def judge(world: World, after: Lander) -> tuple[Lander, Outcome]:
    config = world.config
    box = after.rect(config)
    if box.right < 0 or box.x > config.width:
        return after, Outcome.LOST
    if box.overlaps(world.pad):
        return judge_pad_contact(config, world.lander, after, world.pad)
    if box.bottom >= world.terrain.highest_under(box.x, box.right):
        return after, Outcome.CRASHED
    return after, Outcome.FLYING


def step(world: World, controls: Controls, dt: float) -> World:
    if world.outcome is not Outcome.FLYING:
        return world
    moved = integrate(world.config, world.lander, controls, dt)
    lander, outcome = judge(world, moved)
    return replace(world, lander=lander, outcome=outcome, controls=controls)


def hold(world: World, controls: Controls, dt: float, frames: int) -> World:
    """Apply the same controls for several frames (stops early once the round ends)."""
    for _ in range(frames):
        if world.outcome is not Outcome.FLYING:
            break
        world = step(world, controls, dt)
    return world

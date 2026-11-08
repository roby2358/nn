"""Vectorized physics for many rounds at once, one ship per world. Training flies the whole
population in one Worlds; the GUI flies a Worlds of size one.

Commands are (K, 3): turn in -1..1, thrust in 0..1, fire in 0..1. Rocks live in fixed slots
with size 0 meaning empty, so all worlds keep the same array shape however many rocks remain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (BIG, CMD_FIRE, CMD_THRUST, CMD_TURN, COOLDOWN, DRAG, DRIFT_TABLE, DT, FIELD, FIRE_ABOVE, GUN_READY,
                     MAX_SPEED, NEAREST, POINTS, RADIUS, RADIUS_SCALE, ROCK_CLOSE, ROCK_COS, ROCK_INPUTS, ROCK_RADIUS,
                     ROCK_SIN, ROCK_SLOTS, ROCK_V_FORWARD, ROCK_V_SIDEWAYS, RUN_STEPS, SELF_INPUTS, SENSE_RANGE, SHIP_R,
                     SHOT_COST, SHOT_LIFE, SHOT_SLOTS, SHOT_SPEED, SPAWN_CLEAR, SPEED_HI, SPEED_LO, SPEED_SCALE,
                     SURVIVAL_POINTS, THRUST, TURN_RATE, V_FORWARD, V_SIDEWAYS, WAVE_BASE, WAVE_MAX, WAVE_TABLE)
from .scenario import Scenario

DRAG_PER_STEP = float(np.exp(-DRAG * DT))


def wrap(pos: np.ndarray) -> np.ndarray:
    return pos - FIELD * np.floor(pos / FIELD)


def shortest(delta: np.ndarray) -> np.ndarray:
    """The shortest way across the wrapped field for a raw position difference."""
    return delta - FIELD * np.rint(delta / FIELD)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


@dataclass(frozen=True)
class Event:
    kind: str        # "break" or "crash"
    world: int
    pos: tuple[float, float]
    size: int        # the broken rock's size; a crash reports BIG so its burst reads large


class Worlds:
    def __init__(self, scenarios: list[Scenario], scenario_of: np.ndarray) -> None:
        k = len(scenario_of)
        self.scenario_of = scenario_of
        self.spots = np.stack([s.spots for s in scenarios])     # (S, WAVE_TABLE, WAVE_MAX, 2)
        self.drifts = np.stack([s.drifts for s in scenarios])   # (S, DRIFT_TABLE, 2)
        self.pos = np.tile(FIELD / 2, (k, 1))
        self.vel = np.zeros((k, 2))
        self.ang = np.array([scenarios[s].heading for s in scenario_of], dtype=np.float64)
        self.alive = np.ones(k, dtype=bool)
        self.thrust = np.zeros(k)
        self.cooldown = np.zeros(k)
        self.rock_pos = np.zeros((k, ROCK_SLOTS, 2))
        self.rock_vel = np.zeros((k, ROCK_SLOTS, 2))
        self.rock_size = np.zeros((k, ROCK_SLOTS), dtype=np.int64)
        self.shot_pos = np.zeros((k, SHOT_SLOTS, 2))
        self.shot_vel = np.zeros((k, SHOT_SLOTS, 2))
        self.shot_ttl = np.zeros((k, SHOT_SLOTS))
        self.wave = np.zeros(k, dtype=np.int64)
        self.drawn = np.zeros(k, dtype=np.int64)
        self.score = np.zeros(k)
        self.hits = np.zeros(k, dtype=np.int64)
        self.fired = np.zeros(k, dtype=np.int64)
        self.survived = np.zeros(k)
        self.steps = 0
        self.events: list[Event] = []
        for world in range(k):
            self._spawn_wave(world)

    @property
    def size(self) -> int:
        return len(self.scenario_of)

    @property
    def done(self) -> bool:
        return self.steps >= RUN_STEPS or not self.alive.any()

    def fitness(self) -> np.ndarray:
        return self.score + SURVIVAL_POINTS * self.survived - SHOT_COST * self.fired

    def nose(self) -> np.ndarray:
        return np.stack([np.cos(self.ang), np.sin(self.ang)], axis=1)

    def observe(self) -> np.ndarray:
        """(K, N_INPUTS): the ship's velocity and gun state, then the NEAREST rocks by surface
        distance across the wrapped field. All in the ship's own frame; an absent rock is zeros."""
        delta = shortest(self.rock_pos - self.pos[:, None])
        dist = np.hypot(delta[..., 0], delta[..., 1])
        gap = np.where(self.rock_size > 0, dist - RADIUS[self.rock_size], np.inf)
        rows = np.arange(self.size)[:, None]
        near = np.argsort(gap, axis=1)[:, :NEAREST]
        size = self.rock_size[rows, near]
        present = size > 0
        offset = delta[rows, near]
        rel = self.rock_vel[rows, near] - self.vel[:, None]
        span = np.maximum(dist[rows, near], 1e-9)
        closeness = np.clip(1.0 - np.where(present, gap[rows, near], 0.0) / SENSE_RANGE, 0.0, 1.0)

        fwd = self.nose()
        side = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
        rocks = np.zeros((self.size, NEAREST, ROCK_INPUTS))
        rocks[..., ROCK_SIN] = dot(offset, side[:, None]) / span
        rocks[..., ROCK_COS] = dot(offset, fwd[:, None]) / span
        rocks[..., ROCK_CLOSE] = closeness
        rocks[..., ROCK_V_FORWARD] = dot(rel, fwd[:, None]) / SPEED_SCALE
        rocks[..., ROCK_V_SIDEWAYS] = dot(rel, side[:, None]) / SPEED_SCALE
        rocks[..., ROCK_RADIUS] = RADIUS[size] / RADIUS_SCALE
        rocks *= present[..., None]
        own = np.zeros((self.size, SELF_INPUTS))
        own[:, V_FORWARD] = dot(self.vel, fwd) / SPEED_SCALE
        own[:, V_SIDEWAYS] = dot(self.vel, side) / SPEED_SCALE
        own[:, GUN_READY] = self.cooldown <= 0.0
        return np.concatenate([own, rocks.reshape(self.size, -1)], axis=1)

    def step(self, commands: np.ndarray) -> None:
        self.events = []
        if self.done:
            return
        live = self.alive
        self.thrust = np.clip(commands[:, CMD_THRUST], 0.0, 1.0) * live
        self.ang = self.ang + np.clip(commands[:, CMD_TURN], -1.0, 1.0) * live * TURN_RATE * DT
        nose = self.nose()
        self._move_ship(nose)
        self.rock_pos = wrap(self.rock_pos + self.rock_vel * DT)
        self.shot_pos = wrap(self.shot_pos + self.shot_vel * DT)
        self.shot_ttl = np.maximum(self.shot_ttl - DT, 0.0)
        self.cooldown = np.maximum(self.cooldown - DT, 0.0)
        self._fire(live & (commands[:, CMD_FIRE] > FIRE_ABOVE), nose)
        self._resolve_shots()
        self._resolve_ships()
        self._start_next_waves()
        self.survived += self.alive * DT
        self.steps += 1

    def _move_ship(self, nose: np.ndarray) -> None:
        vel = (self.vel + nose * (self.thrust * THRUST * DT)[:, None]) * DRAG_PER_STEP
        speed = np.maximum(np.hypot(*vel.T), 1e-9)
        self.vel = vel * np.minimum(1.0, MAX_SPEED / speed)[:, None]
        self.pos = wrap(self.pos + self.vel * DT)

    def _fire(self, trigger: np.ndarray, nose: np.ndarray) -> None:
        can = trigger & (self.cooldown <= 0.0) & (self.shot_ttl <= 0.0).any(1)
        worlds = np.flatnonzero(can)
        if len(worlds) == 0:
            return
        slots = np.argmax(self.shot_ttl[worlds] <= 0.0, axis=1)
        self.shot_pos[worlds, slots] = wrap(self.pos[worlds] + nose[worlds] * SHIP_R)
        self.shot_vel[worlds, slots] = self.vel[worlds] + nose[worlds] * SHOT_SPEED
        self.shot_ttl[worlds, slots] = SHOT_LIFE
        self.cooldown[worlds] = COOLDOWN
        self.fired[worlds] += 1

    def _resolve_shots(self) -> None:
        worlds, shots = np.nonzero(self.shot_ttl > 0.0)
        if len(worlds) == 0:
            return
        delta = shortest(self.rock_pos[worlds] - self.shot_pos[worlds, shots][:, None])
        struck = dot(delta, delta) < RADIUS[self.rock_size[worlds]] ** 2
        for line, rock in zip(*np.nonzero(struck)):
            self._strike(int(worlds[line]), int(shots[line]), int(rock))

    def _strike(self, world: int, shot: int, rock: int) -> None:
        size = int(self.rock_size[world, rock])
        if size == 0 or self.shot_ttl[world, shot] <= 0.0:      # already spent earlier this step
            return
        self.shot_ttl[world, shot] = 0.0
        self.score[world] += POINTS[size]
        self.hits[world] += 1
        self.events.append(Event("break", world, tuple(self.rock_pos[world, rock]), size))
        self.rock_size[world, rock] = size - 1
        if size == 1:
            return
        self.rock_vel[world, rock] = self._draw_drift(world, size - 1)
        twin = int(np.argmax(self.rock_size[world] == 0))
        self.rock_size[world, twin] = size - 1
        self.rock_pos[world, twin] = self.rock_pos[world, rock]
        self.rock_vel[world, twin] = self._draw_drift(world, size - 1)

    def _resolve_ships(self) -> None:
        delta = shortest(self.rock_pos - self.pos[:, None])
        reach = RADIUS[self.rock_size] + SHIP_R
        touching = (dot(delta, delta) < reach ** 2) & (self.rock_size > 0)
        crashed = self.alive & touching.any(1)
        self.alive = self.alive & ~crashed
        self.vel[crashed] = 0.0
        self.shot_ttl[crashed] = 0.0       # a dead ship's shots can't go on to score
        self.thrust = self.thrust * self.alive
        for world in np.flatnonzero(crashed):
            self.events.append(Event("crash", int(world), tuple(self.pos[world]), BIG))

    def _start_next_waves(self) -> None:
        for world in np.flatnonzero(self.alive & (self.rock_size == 0).all(1)):
            self.wave[world] += 1
            self._spawn_wave(int(world))

    def _spawn_wave(self, world: int) -> None:
        count = min(WAVE_BASE + int(self.wave[world]), WAVE_MAX)
        spots = self.spots[self.scenario_of[world], self.wave[world] % WAVE_TABLE, :count]
        too_close = np.hypot(*shortest(spots - self.pos[world]).T) < SPAWN_CLEAR
        self.rock_pos[world, :count] = wrap(spots + too_close[:, None] * FIELD / 2)
        self.rock_size[world, :count] = BIG
        for rock in range(count):
            self.rock_vel[world, rock] = self._draw_drift(world, BIG)

    def _draw_drift(self, world: int, size: int) -> np.ndarray:
        angle, fraction = self.drifts[self.scenario_of[world], self.drawn[world] % DRIFT_TABLE]
        self.drawn[world] += 1
        speed = SPEED_LO[size] + fraction * (SPEED_HI[size] - SPEED_LO[size])
        return speed * np.array([np.cos(angle), np.sin(angle)])

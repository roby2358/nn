"""Vectorized physics for many ships at once: training flies the whole population in
one Flock, the GUI replays it, and a race is just a Flock with the shared crystal rule.

Every ship chases crystal `idx[k]` of its own course. A *rule* decides, when ships touch
their crystal, who scores and who moves on to the next one:

- solo:   every toucher scores and advances on its own (training, ghost replays)
- shared: the closest toucher scores, and everybody advances together (races)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import (BONUS_FAST, BONUS_SLOW, CLOSENESS_MAX, DIST_SCALE, DT, FIELD_H, FIELD_W, MARGIN,
                     PICKUP_R, RUN_STEPS, SPEED_SCALE, THRUST, TURN_RATE)
from .course import Course

# rule(hits, dist) -> (scorers, advancers), both boolean (K,)
Rule = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def solo_rule(hits: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return hits, hits


def shared_rule(hits: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scorers = np.zeros_like(hits)
    if not hits.any():
        return scorers, scorers
    scorers[np.argmin(np.where(hits, dist, np.inf))] = True
    return scorers, np.ones_like(hits)


def speed_bonus(seconds: np.ndarray) -> np.ndarray:
    """+1 for a pickup within BONUS_FAST seconds, fading linearly to 0 at BONUS_SLOW."""
    return np.clip((BONUS_SLOW - seconds) / (BONUS_SLOW - BONUS_FAST), 0.0, 1.0)


def decode(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """action -> (turn in {-1, 0, 1}, thrust in {0, 1})."""
    return actions // 2 - 1, actions % 2


def encode_action(turn, thrust):
    """(turn, thrust) -> action; works on ints and on numpy arrays alike."""
    return (turn + 1) * 2 + thrust


@dataclass(frozen=True)
class Event:
    kind: str        # "pickup" or "crash"
    ship: int
    pos: tuple[float, float]


class Flock:
    def __init__(self, courses: list[Course], course_of: np.ndarray, rule: Rule) -> None:
        self.courses = courses
        self.course_of = course_of
        self.rule = rule
        self.table = np.stack([c.crystals for c in courses])          # (C, M, 2)
        k = len(course_of)
        self.pos = np.stack([courses[c].start for c in course_of]).astype(np.float64)
        self.vel = np.zeros((k, 2))
        self.ang = np.array([courses[c].heading for c in course_of], dtype=np.float64)
        self.alive = np.ones(k, dtype=bool)
        self.thrusting = np.zeros(k, dtype=bool)
        self.idx = np.zeros(k, dtype=np.int64)
        self.score = np.zeros(k)
        self.crystals = np.zeros(k, dtype=np.int64)
        self.since = np.zeros(k)
        self.start_dist = self._dist()
        self.min_dist = self.start_dist.copy()
        self.steps = 0
        self.events: list[Event] = []

    @property
    def size(self) -> int:
        return len(self.course_of)

    @property
    def done(self) -> bool:
        return self.steps >= RUN_STEPS or not self.alive.any()

    @property
    def seconds(self) -> float:
        return self.steps * DT

    def targets(self) -> np.ndarray:
        return self.table[self.course_of, self.idx % self.table.shape[1]]  # endless flight wraps around

    def new_lap(self) -> None:
        """Start a fresh scoring window (clock, crystals, score, closeness baseline) without
        touching the ships or which crystal each is chasing."""
        self.steps = 0
        self.score[:] = 0.0
        self.crystals[:] = 0
        self.start_dist = self._dist()
        self.min_dist = self.start_dist.copy()

    def _dist(self) -> np.ndarray:
        return np.hypot(*(self.targets() - self.pos).T)

    def closeness(self) -> np.ndarray:
        progress = 1.0 - self.min_dist / np.maximum(self.start_dist, 1e-9)
        return CLOSENESS_MAX * np.clip(progress, 0.0, 1.0)

    def fitness(self) -> np.ndarray:
        return self.score + self.closeness()

    def observe(self) -> np.ndarray:
        """(K, 6) egocentric inputs: sin & cos of the bearing to the crystal, distance,
        velocity forward & sideways, and closing speed (how fast the distance shrinks)."""
        d = self.targets() - self.pos
        f = np.stack([np.cos(self.ang), np.sin(self.ang)], axis=1)
        r = np.stack([-f[:, 1], f[:, 0]], axis=1)
        dist = np.maximum(np.hypot(*d.T), 1e-9)
        dx, dy = (d * f).sum(1), (d * r).sum(1)
        vf, vs = (self.vel * f).sum(1), (self.vel * r).sum(1)
        closing = (self.vel * d).sum(1) / dist
        return np.stack([dy / dist, dx / dist, dist / DIST_SCALE,
                         vf / SPEED_SCALE, vs / SPEED_SCALE, closing / SPEED_SCALE], axis=1)

    def step(self, actions: np.ndarray) -> None:
        self.events = []
        if self.done:
            return
        turn, thrust = decode(actions)
        live = self.alive
        self.thrusting = live & (thrust == 1)
        self.ang = np.where(live, self.ang + turn * TURN_RATE * DT, self.ang)
        push = self.thrusting * THRUST * DT
        self.vel = self.vel + np.stack([np.cos(self.ang), np.sin(self.ang)], axis=1) * push[:, None]
        self.pos = np.where(live[:, None], self.pos + self.vel * DT, self.pos)
        self._crash()
        self.since += DT
        dist = self._dist()
        self.min_dist = np.where(self.alive, np.minimum(self.min_dist, dist), self.min_dist)
        self._pickup(self.alive & (dist < PICKUP_R), dist)
        self.steps += 1

    def _crash(self) -> None:
        x, y = self.pos.T
        out = (x < -MARGIN) | (x > FIELD_W + MARGIN) | (y < -MARGIN) | (y > FIELD_H + MARGIN)
        crashed = self.alive & out
        self.alive = self.alive & ~out
        self.vel[crashed] = 0.0
        self.thrusting = self.thrusting & self.alive
        self.events += [Event("crash", int(k), tuple(self.pos[k])) for k in np.flatnonzero(crashed)]

    def _pickup(self, hits: np.ndarray, dist: np.ndarray) -> None:
        scorers, advancers = self.rule(hits, dist)
        if not advancers.any():
            return
        self.score[scorers] += 1.0 + speed_bonus(self.since[scorers])
        self.crystals[scorers] += 1
        self.events += [Event("pickup", int(k), tuple(self.targets()[k])) for k in np.flatnonzero(scorers)]
        self.idx[advancers] += 1
        self.since[advancers] = 0.0
        fresh = self._dist()
        self.start_dist[advancers] = fresh[advancers]
        self.min_dist[advancers] = fresh[advancers]

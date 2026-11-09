"""Vectorized physics for many battles at once. Every array is (W worlds, T tanks, ...).

Each tank owns one shell slot, so a shell's owner is simply its slot index. A shell keeps
flying and scoring after its owner dies. Commands and observations cross this boundary
flattened to (W*T, ...), one row per tank, because every tank has its own net.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (CLOSE, CMD_LEFT, CMD_RIGHT, CMD_SHOOT, CONTACT_INPUTS, COS, DT, FIELD, FIRE_ABOVE,
                     FRIENDLY_KILL_COST, GUN_COS, GUN_SIN, KILL_POINTS, LOADED, MAX_SPEED, MUZZLE_GAP, N_INPUTS, NEAR_MISS_POINTS, NEAR_MISS_R, NEAREST, N_OUTPUTS,
                     RELOAD, SENSE_RANGE, SHELL_LIFE, SHELL_SPEED, SIN, SPLASH_R, SURVIVAL_POINTS, TANK_R,
                     TANKS, TEAM_SIZE, TURN_RATE)
from .geometry import dot, length
from .scenario import Scenario

GREEN = np.arange(TANKS) < TEAM_SIZE
BLUE = ~GREEN
ENEMY = GREEN[:, None] != GREEN[None, :]      # (T, T): tank j is an enemy of tank i
FRIEND = ~ENEMY & ~np.eye(TANKS, dtype=bool)  # (T, T): tank j is a teammate of i, and not i


@dataclass(frozen=True)
class Blast:
    world: int
    pos: tuple[float, float]


class Worlds:
    def __init__(self, scenarios: list[Scenario], step_limit: float) -> None:
        w = len(scenarios)
        self.count = w
        self.step_limit = step_limit
        self.pos = np.stack([s.spots for s in scenarios])
        self.ang = np.stack([s.headings for s in scenarios])
        self.alive = np.ones((w, TANKS), dtype=bool)
        self.cooldown = np.zeros((w, TANKS))
        self.shell_pos = np.zeros((w, TANKS, 2))
        self.shell_vel = np.zeros((w, TANKS, 2))
        self.shell_ttl = np.zeros((w, TANKS))
        self.shell_near = np.zeros((w, TANKS), dtype=bool)   # came within NEAR_MISS_R of an enemy
        self.kills = np.zeros((w, TANKS), dtype=np.int64)
        self.friendly_kills = np.zeros((w, TANKS), dtype=np.int64)
        self.near_misses = np.zeros((w, TANKS), dtype=np.int64)
        self.fired = np.zeros((w, TANKS), dtype=np.int64)
        self.survived = np.zeros((w, TANKS))
        self.steps = 0
        self.blasts: list[Blast] = []

    @property
    def done(self) -> bool:
        return self.steps >= self.step_limit

    @property
    def over(self) -> np.ndarray:
        """(W,): the battles with a team wiped out. Their tanks hold still, and survivors keep scoring time."""
        return ~((self.alive & GREEN).any(1) & (self.alive & BLUE).any(1))

    def fitness(self) -> np.ndarray:
        return (SURVIVAL_POINTS * self.survived + KILL_POINTS * self.kills
                - FRIENDLY_KILL_COST * self.friendly_kills + NEAR_MISS_POINTS * self.near_misses)

    def nose(self) -> np.ndarray:
        return np.stack([np.cos(self.ang), np.sin(self.ang)], axis=2)

    def _tank_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """(W, T, T, 2) from each tank i to each tank j, and its (W, T, T) length."""
        delta = self.pos[:, None] - self.pos[:, :, None]
        return delta, length(delta)

    def observe(self) -> np.ndarray:
        """(W*T, N_INPUTS): gun ready, the NEAREST live enemies, then the NEAREST live friends.
        Each is closest first; an absent contact, or a dead tank's view, is zeros."""
        delta, dist = self._tank_offsets()
        seen = self.alive[:, :, None] & self.alive[:, None]
        gun = (self.cooldown <= 0.0) & self.alive
        views = self._views(delta, dist, gun)
        enemies = self._contacts(views, dist, seen & ENEMY)
        friends = self._contacts(views, dist, seen & FRIEND)
        rows = np.concatenate([gun[:, :, None], enemies, friends], axis=2)
        return rows.reshape(self.count * TANKS, N_INPUTS)

    def _views(self, delta: np.ndarray, dist: np.ndarray, gun: np.ndarray) -> np.ndarray:
        """(W, T, T, CONTACT_INPUTS): what each tank i sees of each tank j. Bearing is in i's own frame.
        How far j's gun points off i is the bearing of i in j's frame, so it's the same array transposed."""
        forward = self.nose()[:, :, None]
        side = np.stack([-forward[..., 1], forward[..., 0]], axis=3)
        span = np.maximum(dist, 1e-9)
        views = np.zeros((self.count, TANKS, TANKS, CONTACT_INPUTS))
        views[..., SIN] = dot(delta, side) / span
        views[..., COS] = dot(delta, forward) / span
        views[..., CLOSE] = np.clip(1.0 - span / SENSE_RANGE, 0.0, 1.0)
        views[..., GUN_SIN] = views[..., SIN].swapaxes(1, 2)
        views[..., GUN_COS] = views[..., COS].swapaxes(1, 2)
        views[..., LOADED] = gun[:, None]
        return views

    def _contacts(self, views: np.ndarray, dist: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """(W, T, NEAREST * CONTACT_INPUTS) for the visible tanks closest to each tank."""
        rows, cols, _ = np.ogrid[:self.count, :TANKS, :1]
        near = np.argsort(np.where(visible, dist, np.inf), axis=2)[:, :, :NEAREST]
        contacts = views[rows, cols, near] * visible[rows, cols, near][..., None]
        return contacts.reshape(self.count, TANKS, NEAREST * CONTACT_INPUTS)

    def step(self, commands: np.ndarray) -> None:
        self.blasts = []
        if self.done:
            return
        orders = commands.reshape(self.count, TANKS, N_OUTPUTS)
        fighting = self.alive & ~self.over[:, None]
        self._drive(orders, fighting)
        self._block()
        self._blow_up_strays()
        expired = self._advance_shells()
        self.cooldown = np.maximum(self.cooldown - DT, 0.0)
        self._fire(fighting & self.alive & (orders[..., CMD_SHOOT] > FIRE_ABOVE))
        self._resolve_shells(expired)
        self.survived += self.alive * DT
        self.steps += 1

    def _drive(self, orders: np.ndarray, fighting: np.ndarray) -> None:
        """Differential drive. Screen y points down, so a stronger right tread lowers the heading (turns left)."""
        left = np.clip(orders[..., CMD_LEFT], -1.0, 1.0) * fighting
        right = np.clip(orders[..., CMD_RIGHT], -1.0, 1.0) * fighting
        self.ang = self.ang + (left - right) / 2.0 * TURN_RATE * DT
        speed = (left + right) / 2.0 * MAX_SPEED
        self.pos = self.pos + self.nose() * speed[..., None] * DT

    def _block(self) -> None:
        """Overlapping live tanks each back off by half the overlap."""
        delta, dist = self._tank_offsets()
        pairs = self.alive[:, :, None] & self.alive[:, None] & ~np.eye(TANKS, dtype=bool)
        overlap = np.maximum(2.0 * TANK_R - dist, 0.0) * pairs
        away = -delta / np.maximum(dist, 1e-9)[..., None]
        self.pos = self.pos + (away * overlap[..., None] / 2.0).sum(2)

    def _blow_up_strays(self) -> None:
        """A live tank whose centre has left the field dies. It takes nobody with it and nobody gets the credit."""
        strayed = self.alive & ((self.pos < 0.0) | (self.pos > FIELD)).any(2)
        self.alive &= ~strayed
        self.blasts += [Blast(int(w), tuple(self.pos[w, t])) for w, t in zip(*np.nonzero(strayed))]

    def _advance_shells(self) -> np.ndarray:
        """Move the shells in flight; returns (W, T), the ones that just reached full range."""
        flying = self.shell_ttl > 0.0
        self.shell_pos = self.shell_pos + self.shell_vel * DT
        self.shell_ttl = np.maximum(self.shell_ttl - DT, 0.0)
        return flying & (self.shell_ttl <= 0.0)

    def _fire(self, trigger: np.ndarray) -> None:
        can = trigger & (self.cooldown <= 0.0)
        nose = self.nose()
        self.shell_pos[can] = self.pos[can] + nose[can] * (TANK_R + MUZZLE_GAP)
        self.shell_vel[can] = nose[can] * SHELL_SPEED
        self.shell_ttl[can] = SHELL_LIFE
        self.shell_near[can] = False
        self.cooldown[can] = RELOAD
        self.fired[can] += 1

    def _resolve_shells(self, expired: np.ndarray) -> None:
        """Every shell going off this step acts on the same live tanks, so two shells that
        catch one tank both get the kill and neither depends on the order of the slots."""
        active = expired | (self.shell_ttl > 0.0)
        reach = length(self.pos[:, None] - self.shell_pos[:, :, None])   # (W, shell, tank)
        alive = self.alive[:, None]
        self.shell_near |= active & ((reach < NEAR_MISS_R) & alive & ENEMY).any(2)
        touched = active & ((reach < TANK_R) & alive).any(2)
        blown = expired | touched
        if not blown.any():
            return
        caught = (reach < SPLASH_R) & alive & blown[:, :, None]
        killed_enemy = caught & ENEMY
        self.kills += killed_enemy.sum(2)
        self.friendly_kills += (caught & ~ENEMY).sum(2)
        self.near_misses += blown & self.shell_near & ~killed_enemy.any(2)
        self.alive &= ~caught.any(1)
        self.shell_ttl[blown] = 0.0
        self.blasts += [Blast(int(w), tuple(self.shell_pos[w, s])) for w, s in zip(*np.nonzero(blown))]

"""A tiny policy network in numpy: what the lander sees in, thruster probabilities out.

Each of the three thrusters (main, left, right) is an independent on/off choice,
so the output is three sigmoid probabilities rather than one softmax.
"""
from pathlib import Path

import numpy as np

from lander.physics import Controls, World

INPUT = 4
HIDDEN = 32
OUTPUT = 3  # main, left, right
SPEED_SCALE = 100.0  # px/s that maps to 1.0


def encode(world: World) -> np.ndarray:
    """4 floats: pad offset (x, y) from the lander's feet, and velocity, all roughly in [-1, 1].
    Positive dx means the pad is to the right; positive dy means the pad is below."""
    c, lander, pad = world.config, world.lander, world.pad
    dx = (pad.x + pad.w / 2) - (lander.x + c.lander_size / 2)
    dy = pad.y - (lander.y + c.lander_size)
    return np.array([dx / c.width * 2, dy / c.height * 2,
                     lander.vx / SPEED_SCALE, lander.vy / SPEED_SCALE], dtype=np.float32)


def to_controls(bits: np.ndarray) -> Controls:
    return Controls(main=bool(bits[0]), left=bool(bits[1]), right=bool(bits[2]))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class PolicyNet:
    def __init__(self, rng: np.random.Generator) -> None:
        self.W1 = (rng.standard_normal((INPUT, HIDDEN)) * np.sqrt(2 / INPUT)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = (rng.standard_normal((HIDDEN, OUTPUT)) * np.sqrt(1 / HIDDEN)).astype(np.float32)
        self.b2 = np.zeros(OUTPUT, dtype=np.float32)
        self._m = [np.zeros_like(p) for p in self.params]
        self._v = [np.zeros_like(p) for p in self.params]
        self._t = 0

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X is (N, 4). Returns (probs (N, 3), logits (N, 3), hidden (N, H))."""
        H = np.tanh(X @ self.W1 + self.b1)
        Z = H @ self.W2 + self.b2
        return sigmoid(Z), Z, H

    def probs(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def greedy(self, world: World) -> Controls:
        """Fire each thruster whose probability is over one half."""
        return to_controls(self.probs(encode(world)[None, :])[0] > 0.5)

    def update(self, X: np.ndarray, actions: np.ndarray, advantages: np.ndarray,
               lr: float, entropy_beta: float) -> None:
        """One REINFORCE step. actions is (N, 3) of 0/1, advantages is (N,).

        For a sigmoid, d log pi(a) / d logit = a - p, and the entropy gradient is
        -p (1 - p) * logit, which nudges confident thrusters back toward 50/50."""
        N = X.shape[0]
        P, Z, H = self.forward(X)
        dZ = advantages[:, None] * (actions - P) + entropy_beta * (-P * (1 - P) * Z)
        dZ /= N
        gW2 = H.T @ dZ
        gb2 = dZ.sum(axis=0)
        dH = (dZ @ self.W2.T) * (1 - H * H)
        gW1 = X.T @ dH
        gb1 = dH.sum(axis=0)
        self._adam([gW1, gb1, gW2, gb2], lr)

    def _adam(self, grads: list[np.ndarray], lr: float) -> None:
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._t += 1
        for p, g, m, v in zip(self.params, grads, self._m, self._v):
            m[...] = b1 * m + (1 - b1) * g
            v[...] = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1 ** self._t)
            vhat = v / (1 - b2 ** self._t)
            p += lr * mhat / (np.sqrt(vhat) + eps)  # ascent

    def save(self, path: Path) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_from(self, path: Path) -> None:
        """Overwrite weights from a file; raises ValueError if the shapes don't match."""
        with np.load(path) as f:
            for name, p in zip(("W1", "b1", "W2", "b2"), self.params):
                if f[name].shape != p.shape:
                    raise ValueError(f"{path} has {name} shaped {f[name].shape}, this net uses {p.shape}")
            for name, p in zip(("W1", "b1", "W2", "b2"), self.params):
                p[...] = f[name]
        self._reset_adam()

    def copy_from(self, other: "PolicyNet") -> None:
        for p, q in zip(self.params, other.params):
            p[...] = q
        self._reset_adam()

    def _reset_adam(self) -> None:
        for m, v in zip(self._m, self._v):
            m[...] = 0
            v[...] = 0
        self._t = 0

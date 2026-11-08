"""A tiny policy network in numpy: board in, move probabilities out. No lookahead."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..board import Board

INPUT = 10
HIDDEN = 64
OUTPUT = 9


def encode(cells: list[str | None], player: str) -> np.ndarray:
    """10 floats from a fixed view. The first 9 are the cells: X = +1, O = -1,
    empty = 0, the same for both sides. The 10th is whose turn it is:
    +1 when X is to move, -1 when O is to move."""
    board = [1.0 if c == "X" else -1.0 if c == "O" else 0.0 for c in cells]
    return np.array(board + [1.0 if player == "X" else -1.0], dtype=np.float32)


class PolicyNet:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        self.W1 = (rng.standard_normal((INPUT, HIDDEN)) * np.sqrt(2 / INPUT)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, dtype=np.float32)
        self.W2 = (rng.standard_normal((HIDDEN, OUTPUT)) * np.sqrt(1 / HIDDEN)).astype(np.float32)
        self.b2 = np.zeros(OUTPUT, dtype=np.float32)
        # Adam state
        self._m = [np.zeros_like(p) for p in self.params]
        self._v = [np.zeros_like(p) for p in self.params]
        self._t = 0

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X is (N, 27). Returns (probs (N, 9), hidden (N, 64))."""
        H = np.tanh(X @ self.W1 + self.b1)
        logits = H @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        return P, H

    def probs(self, cells: list[str | None], player: str) -> np.ndarray:
        P, _ = self.forward(encode(cells, player)[None, :])
        return P[0]

    def choose(self, board: Board, greedy: bool = True, rng: np.random.Generator | None = None) -> int:
        """Pick a move with no lookahead. If the net names an occupied cell, try again
        with that cell excluded, so the caller always gets a legal move."""
        rng = rng or np.random.default_rng()
        p = self.probs(board.cells, board.turn).copy()
        while True:
            move = int(np.argmax(p)) if greedy else int(rng.choice(OUTPUT, p=p / p.sum()))
            if board.cells[move] is None:
                return move
            p[move] = 0.0
            if p.sum() <= 0:  # all remaining mass was on bad cells; fall back to uniform legal
                p = np.array([1.0 if c is None else 0.0 for c in board.cells], dtype=np.float32)

    def update(self, X: np.ndarray, actions: np.ndarray, advantages: np.ndarray,
               lr: float = 3e-3, entropy_beta: float = 0.01,
               illegal: np.ndarray | None = None) -> None:
        """One REINFORCE step. Each row is one pick, and `illegal` flags which picks
        named an occupied cell. The two kinds of row have separate jobs:

        - Legal picks are scored by the game result, and only reorder probability
          among the legal cells (the softmax renormalized over empty cells). A losing
          move therefore can't push probability onto occupied cells.
        - Illegal picks raise the total probability on legal cells, pushing every
          occupied cell down at once, weighted by the size of the penalty. This is the
          only signal that teaches the net to stop naming occupied cells.

        The entropy bonus also only counts legal cells, so it never rewards spreading
        probability onto occupied squares."""
        N = X.shape[0]
        P, H = self.forward(X)
        onehot = np.zeros_like(P)
        onehot[np.arange(N), actions] = 1.0
        legal = (X[:, :9] == 0).astype(np.float32)  # 1.0 where the cell is empty
        Q = P * legal
        Q /= np.maximum(Q.sum(axis=1, keepdims=True), 1e-8)
        logQ = np.log(Q + 1e-8) * legal
        ent = -(Q * logQ).sum(axis=1, keepdims=True)
        if illegal is None:
            illegal = np.zeros(N, dtype=bool)
        # gradient of the objective wrt logits (ascent direction)
        # Legal pick: gradient of log Q(a), which only moves probability among legal cells.
        # Illegal pick: gradient of log(total probability on legal cells), which is Q - P.
        # It pushes every occupied cell down at once rather than just the one named.
        # Penalties are stored as negative advantages, so flip the sign for that row.
        grad_legal = (onehot - Q) * legal
        grad_rules = Q - P
        dlogits = np.where(illegal[:, None], -advantages[:, None] * grad_rules, advantages[:, None] * grad_legal)
        dlogits = dlogits + entropy_beta * (-Q * (logQ + ent)) * legal
        dlogits /= N
        gW2 = H.T @ dlogits
        gb2 = dlogits.sum(axis=0)
        dH = dlogits @ self.W2.T
        dZ = dH * (1 - H * H)
        gW1 = X.T @ dZ
        gb1 = dZ.sum(axis=0)
        self._adam([gW1, gb1, gW2, gb2], lr)

    def _adam(self, grads: list[np.ndarray], lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self._t += 1
        for p, g, m, v in zip(self.params, grads, self._m, self._v):
            m[...] = b1 * m + (1 - b1) * g
            v[...] = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1 ** self._t)
            vhat = v / (1 - b2 ** self._t)
            p += lr * mhat / (np.sqrt(vhat) + eps)  # ascent

    def save(self, path: str | Path) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyNet":
        net = cls()
        with np.load(path) as f:
            if f["W1"].shape != net.W1.shape:
                raise ValueError(f"{path} was saved for a different board encoding "
                                 f"({f['W1'].shape[0]} inputs, this net uses {net.W1.shape[0]})")
            net.W1[...] = f["W1"]
            net.b1[...] = f["b1"]
            net.W2[...] = f["W2"]
            net.b2[...] = f["b2"]
        return net

    def copy_from(self, other: "PolicyNet") -> None:
        for p, q in zip(self.params, other.params):
            p[...] = q

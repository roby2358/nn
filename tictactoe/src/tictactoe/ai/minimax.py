"""Perfect-play oracle. Used only to measure the policy net, never to play the human."""

from __future__ import annotations

import random
from functools import lru_cache

from ..board import LINES, Board


def _winner(cells: tuple[str | None, ...]) -> str | None:
    for a, b, c in LINES:
        if cells[a] and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return None


@lru_cache(maxsize=None)
def value(cells: tuple[str | None, ...], player: str) -> int:
    """Game value from X's view (+1 X wins, -1 O wins, 0 draw), with `player` to move."""
    w = _winner(cells)
    if w:
        return 1 if w == "X" else -1
    if all(cells):
        return 0
    nxt = "O" if player == "X" else "X"
    vals = []
    for i, c in enumerate(cells):
        if c is None:
            child = cells[:i] + (player,) + cells[i + 1:]
            vals.append(value(child, nxt))
    return max(vals) if player == "X" else min(vals)


def best_moves(board: Board) -> list[int]:
    cells = tuple(board.cells)
    nxt = "O" if board.turn == "X" else "X"
    scored = []
    for i, c in enumerate(cells):
        if c is None:
            child = cells[:i] + (board.turn,) + cells[i + 1:]
            scored.append((value(child, nxt), i))
    best = max(s for s, _ in scored) if board.turn == "X" else min(s for s, _ in scored)
    return [i for s, i in scored if s == best]


def choose(board: Board, rng: random.Random | None = None) -> int:
    """A random choice among the perfect moves, so games against it vary."""
    return (rng or random).choice(best_moves(board))

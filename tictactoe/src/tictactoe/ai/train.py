"""Self-play REINFORCE. The net plays both sides; winners' moves go up, losers' go down."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..board import Board
from . import minimax
from .policy import PolicyNet, encode

WEIGHTS = Path("weights.npz")


@dataclass
class Stats:
    games: int = 0
    x_win: float = 0.0
    draw: float = 0.0
    o_win: float = 0.0
    invalid: float = 0.0          # fraction of picks that named an occupied cell
    vs_minimax: float | None = None  # draw rate against perfect play, when measured


@dataclass
class Trainer:
    net: PolicyNet = field(default_factory=PolicyNet)
    games_per_step: int = 64
    gamma: float = 0.95
    lr: float = 3e-3
    entropy_beta: float = 0.02
    illegal_penalty: float = -1.0  # advantage for picking an occupied cell
    epsilon: float = 0.05     # chance of a random legal move during self-play, for exploration
    seed: int | None = None
    history: list[Stats] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.pyrng = random.Random(self.seed)
        self.total_games = 0
        self._baseline = 0.0

    # ---- one self-play game -------------------------------------------------
    def _play_one(self) -> tuple[list[tuple[np.ndarray, int, str]], list[tuple[np.ndarray, int]], str | None, int]:
        """Play one self-play game.

        Returns (records, illegal, winner, picks):
          records  the legal moves played, as (state, move, mover)
          illegal  every illegal pick, as (state, move). Each one is penalized on its
                   own; the net then picks again with that cell excluded, so illegal
                   picks never end the game or change who wins.
          winner   'X', 'O' or None for a draw
          picks    total picks made, legal and illegal, for the illegal-pick rate
        """
        b = Board()
        records: list[tuple[np.ndarray, int, str]] = []
        illegal: list[tuple[np.ndarray, int]] = []
        picks = 0
        while not b.game_over:
            x = encode(b.cells, b.turn)
            if self.rng.random() < self.epsilon:
                move = int(self.rng.choice([i for i, c in enumerate(b.cells) if c is None]))
                picks += 1
            else:
                p = self.net.forward(x[None, :])[0][0].astype(np.float64)
                while True:
                    move = int(self.rng.choice(9, p=p / p.sum()))
                    picks += 1
                    if b.cells[move] is None:
                        break
                    illegal.append((x, move))
                    p[move] = 0.0
                    if p.sum() < 1e-12:
                        # Every legal cell rounded to zero probability: pick uniformly
                        # among legal cells. The illegal picks above are still penalized.
                        p = np.array([1.0 if c is None else 0.0 for c in b.cells])
            records.append((x, move, b.turn))
            b.play(move)
        return records, illegal, b.winner, picks

    # ---- one training step ---------------------------------------------------
    def step(self) -> Stats:
        X, A, R = [], [], []
        IX, IA = [], []
        counts = {"X": 0, "O": 0, None: 0}
        illegal_picks = 0
        total_picks = 0
        for _ in range(self.games_per_step):
            recs, bad, winner, picks = self._play_one()
            counts[winner] += 1
            illegal_picks += len(bad)
            total_picks += picks
            n = len(recs)
            for k, (x, a, who) in enumerate(recs):
                if winner is None:
                    r = 0.0
                else:
                    r = 1.0 if who == winner else -1.0
                r *= self.gamma ** (n - 1 - k)
                X.append(x)
                A.append(a)
                R.append(r)
            for x, a in bad:
                IX.append(x)
                IA.append(a)
        R = np.array(R, dtype=np.float32)
        # Running baseline cuts variance; advantage = return - baseline.
        # It covers legal moves only, so illegal picks can't drag it around.
        self._baseline = 0.9 * self._baseline + 0.1 * float(R.mean())
        adv = R - self._baseline
        # Illegal picks get a fixed penalty, independent of how the game turned out.
        if IX:
            X += IX
            A += IA
            adv = np.concatenate([adv, np.full(len(IX), self.illegal_penalty, dtype=np.float32)])
        flags = np.zeros(len(X), dtype=bool)
        flags[len(X) - len(IX):] = True
        X = np.stack(X)
        A = np.array(A)
        with self.lock:
            self.net.update(X, A, adv, lr=self.lr, entropy_beta=self.entropy_beta, illegal=flags)
        self.total_games += self.games_per_step
        g = self.games_per_step
        s = Stats(self.total_games, counts["X"] / g, counts[None] / g, counts["O"] / g, illegal_picks / total_picks)
        self.history.append(s)
        return s

    # ---- evaluation ----------------------------------------------------------
    def eval_vs_minimax(self, games: int = 20) -> float:
        """Draw rate of the greedy policy against perfect play, half as X, half as O."""
        draws = 0
        for i in range(games):
            b = Board()
            me = "X" if i % 2 == 0 else "O"
            while not b.game_over:
                if b.turn == me:
                    with self.lock:
                        mv = self.net.choose(b, greedy=True)
                else:
                    mv = minimax.choose(b, self.pyrng)
                b.play(mv)
            draws += b.winner is None
        rate = draws / games
        if self.history:
            self.history[-1].vs_minimax = rate
        return rate

    def save(self, path: Path = WEIGHTS) -> None:
        with self.lock:
            self.net.save(path)


class BackgroundTrainer(threading.Thread):
    """Runs trainer.step() in the background. `start_run(n)` trains for at most n
    steps, `halt()` stops early. The thread itself lives until `shutdown()`."""

    def __init__(self, trainer: Trainer, eval_every: int = 10) -> None:
        super().__init__(daemon=True)
        self.trainer = trainer
        self.eval_every = eval_every
        self._go = threading.Event()
        self._quit = threading.Event()
        self.remaining = 0
        self.steps_done = 0

    @property
    def running(self) -> bool:
        return self._go.is_set()

    def start_run(self, steps: int) -> None:
        self.remaining = steps
        self._go.set()

    def halt(self) -> None:
        self._go.clear()

    def shutdown(self) -> None:
        self.halt()
        self._quit.set()

    def run(self) -> None:
        while not self._quit.is_set():
            if not self._go.wait(timeout=0.1):
                continue
            if self.remaining <= 0:
                self._go.clear()
                continue
            self.trainer.step()
            self.remaining -= 1
            self.steps_done += 1
            if self.steps_done % self.eval_every == 0:
                self.trainer.eval_vs_minimax()


def main() -> None:
    """Headless training for the container: `just train [steps]`."""
    import sys

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    try:
        net = PolicyNet.load(WEIGHTS) if WEIGHTS.exists() else PolicyNet()
    except ValueError as e:
        print(f"{e}; starting from a fresh net")
        net = PolicyNet()
    t = Trainer(net=net)
    t0 = time.time()
    for i in range(1, steps + 1):
        s = t.step()
        if i % 10 == 0:
            d = t.eval_vs_minimax()
            print(f"step {i:4d} games {s.games:6d}  X {s.x_win:.2f} draw {s.draw:.2f} O {s.o_win:.2f} "
                  f"invalid {s.invalid:.2f}  draw-vs-minimax {d:.2f}  {time.time() - t0:5.1f}s")
    t.save()
    print(f"saved {WEIGHTS}")


if __name__ == "__main__":
    main()

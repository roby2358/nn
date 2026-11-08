"""REINFORCE with a shaped reward. Batches of episodes fly side by side; thruster
choices that led to better-than-average returns become more likely.

The shaped reward is the change in a "potential" (closer to the pad and slower
is better) plus a bonus or penalty when the round ends. Potential-based shaping
gives a signal on every decision without changing which policy is best.
"""
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lander.physics import CONFIG, DT, Outcome, World, hold, new_world
from lander.ai.policy import PolicyNet, encode, to_controls

WEIGHTS = Path("weights.npz")


@dataclass(frozen=True)
class TrainConfig:
    episodes_per_step: int
    decide_every: int  # frames the net holds each choice (4 frames = 15 decisions/s)
    max_decisions: int  # round length cap; running out counts as a timeout
    gamma: float
    lr: float
    entropy_beta: float
    eval_episodes: int


TRAIN = TrainConfig(
    episodes_per_step=32,
    decide_every=4,
    max_decisions=300,  # 20 s of flight
    gamma=0.99,
    lr=3e-3,
    entropy_beta=0.01,
    eval_episodes=20,
)

RESULTS = ("landed", "crashed", "lost", "timeout")
TERMINAL_REWARD = {"landed": 3.0, "crashed": -1.0, "lost": -1.0, "timeout": -1.0}


def potential(world: World) -> float:
    dx, dy, vx, vy = encode(world)
    return -(abs(dx) + abs(dy)) - 0.5 * (abs(vx) + abs(vy))


def result_of(world: World) -> str:
    """The round's result; a world still flying when time runs out is a timeout."""
    return "timeout" if world.outcome is Outcome.FLYING else world.outcome.value


@dataclass
class Episode:
    world: World
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def returns(self, gamma: float) -> np.ndarray:
        out = np.zeros(len(self.rewards), dtype=np.float32)
        g = 0.0
        for k in range(len(self.rewards) - 1, -1, -1):
            g = self.rewards[k] + gamma * g
            out[k] = g
        return out


@dataclass
class Stats:
    episodes: int
    rates: dict[str, float]  # fraction of this batch per result
    mean_return: float
    greedy_landed: float | None  # greedy landing rate, when measured


def sample(rng: np.random.Generator):
    return lambda P: (rng.random(P.shape) < P).astype(np.float32)


def greedy(P: np.ndarray) -> np.ndarray:
    return (P > 0.5).astype(np.float32)


class Trainer:
    def __init__(self, net: PolicyNet, train: TrainConfig, seed: int | None) -> None:
        self.net = net
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.lock = threading.Lock()
        self.history: list[Stats] = []
        self.total_episodes = 0

    def fly(self, count: int, choose) -> list[Episode]:
        """Fly `count` rounds side by side. choose maps probabilities (N, 3) to 0/1 actions."""
        eps = [Episode(new_world(CONFIG, self.pyrng)) for _ in range(count)]
        for _ in range(self.train.max_decisions):
            live = [e for e in eps if e.world.outcome is Outcome.FLYING]
            if not live:
                break
            X = np.stack([encode(e.world) for e in live])
            with self.lock:
                P = self.net.probs(X)
            A = choose(P)
            for e, x, a in zip(live, X, A):
                after = hold(e.world, to_controls(a), DT, self.train.decide_every)
                e.states.append(x)
                e.actions.append(a)
                e.rewards.append(potential(after) - potential(e.world))
                e.world = after
        for e in eps:
            e.rewards[-1] += TERMINAL_REWARD[result_of(e.world)]
        return eps

    def step(self) -> Stats:
        eps = self.fly(self.train.episodes_per_step, sample(self.rng))
        X = np.concatenate([np.stack(e.states) for e in eps])
        A = np.concatenate([np.stack(e.actions) for e in eps])
        G = np.concatenate([e.returns(self.train.gamma) for e in eps])
        adv = (G - G.mean()) / (G.std() + 1e-8)
        with self.lock:
            self.net.update(X, A, adv, self.train.lr, self.train.entropy_beta)
        self.total_episodes += len(eps)
        stats = Stats(self.total_episodes, rate_by_result(eps),
                      float(np.mean([sum(e.rewards) for e in eps])), None)
        self.history.append(stats)
        return stats

    def eval_greedy(self) -> float:
        """Landing rate of the greedy policy; recorded on the latest Stats."""
        eps = self.fly(self.train.eval_episodes, greedy)
        rate = rate_by_result(eps)["landed"]
        if self.history:
            self.history[-1].greedy_landed = rate
        return rate

    def reset(self, net: PolicyNet) -> None:
        with self.lock:
            self.net.copy_from(net)
        self.history.clear()
        self.total_episodes = 0

    def save(self, path: Path) -> None:
        with self.lock:
            self.net.save(path)

    def load(self, path: Path) -> None:
        with self.lock:
            self.net.load_from(path)


def rate_by_result(eps: list[Episode]) -> dict[str, float]:
    results = [result_of(e.world) for e in eps]
    return {r: results.count(r) / len(eps) for r in RESULTS}


class BackgroundTrainer(threading.Thread):
    """Runs trainer.step() in the background. start_run(n) trains for at most n
    steps, halt() stops early. The thread lives until shutdown()."""

    def __init__(self, trainer: Trainer, eval_every: int) -> None:
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
                self.trainer.eval_greedy()


def main() -> None:
    """Headless training for the container: `just train [steps]`."""
    steps = int(sys.argv[1])
    net = PolicyNet(np.random.default_rng())
    if WEIGHTS.exists():
        net.load_from(WEIGHTS)
        print(f"continuing from {WEIGHTS}")
    t = Trainer(net, TRAIN, None)
    t0 = time.time()
    for i in range(1, steps + 1):
        s = t.step()
        if i % 10 != 0:
            continue
        g = t.eval_greedy()
        rates = "  ".join(f"{r} {s.rates[r]:.2f}" for r in RESULTS)
        print(f"step {i:4d}  {rates}  return {s.mean_return:6.2f}  greedy-landed {g:.2f}  {time.time() - t0:5.1f}s",
              flush=True)
    t.save(WEIGHTS)
    print(f"saved {WEIGHTS}")


if __name__ == "__main__":
    main()

"""How a weights file plays on fixed fields: `just cc-stats <weights.npz> <n>`."""

import sys
from pathlib import Path

import numpy as np

from .config import CMD_FIRE, CMD_THRUST, CMD_TURN, FIRE_ABOVE
from .evolve import accuracy, fly
from .pilots import Brains
from .scenario import make_scenario
from .worlds import Worlds


class Recorder:
    """Wraps a pilot and keeps the commands of the ships still alive when each was issued."""

    def __init__(self, pilot: Brains, worlds: Worlds) -> None:
        self.pilot = pilot
        self.worlds = worlds
        self.commands: list[np.ndarray] = []

    def act(self, obs: np.ndarray) -> np.ndarray:
        out = self.pilot.act(obs)
        self.commands.append(out[self.worlds.alive])
        return out


def main() -> None:
    path, n = Path(sys.argv[1]), int(sys.argv[2])
    scenarios = [make_scenario(s) for s in range(n)]
    pilot = Brains.load(path)
    worlds = Worlds(scenarios, np.arange(n))
    recorder = Recorder(Brains(np.tile(pilot.genomes, (n, 1))), worlds)
    w = fly(worlds, recorder)
    commands = np.concatenate(recorder.commands)
    turn, thrust, fire = commands[:, CMD_TURN], commands[:, CMD_THRUST], commands[:, CMD_FIRE]
    print(f"{path}: {n} fields")
    print(f"  fitness    mean {w.fitness().mean():7.2f}   median {np.median(w.fitness()):7.2f}")
    print(f"  score      mean {w.score.mean():7.2f}   best {w.score.max():7.2f}")
    print(f"  survived   mean {w.survived.mean():7.2f}s  alive at the end {w.alive.mean():4.0%}")
    print(f"  hits       mean {w.hits.mean():7.2f}   waves cleared mean {w.wave.mean():5.2f}")
    print(f"  accuracy   {accuracy(w.hits.sum(), w.fired.sum()):4.0%} of {w.fired.mean():.1f} shots per round")
    print(f"  turn       mean |turn| {np.abs(turn).mean():4.2f}   mean signed {turn.mean():+5.2f}")
    print(f"  thrust     mean {thrust.mean():4.2f}")
    print(f"  trigger    held {(fire > FIRE_ABOVE).mean():4.0%} of the time")


if __name__ == "__main__":
    main()

"""Describe how a set of weights flies, for comparing piloting styles.

Run `python -m lander.ai.flight_stats WEIGHTS.npz ROUNDS`.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lander.ai.policy import SPEED_SCALE, PolicyNet
from lander.ai.train import TRAIN, Episode, Trainer, greedy, rate_by_result, result_of
from lander.physics import CONFIG

DECISIONS_PER_SECOND = 15
SETTLE_SPEED = 30.0  # px/s: slower than this counts as the final settling phase


@dataclass(frozen=True)
class FlightProfile:
    seconds: float
    peak_descent: float  # px/s
    touchdown_descent: float  # px/s at the last decision
    first_burn_height: float  # px above the pad when the main engine first fires
    burn_duty: float  # fraction of decisions with the engine on, from first burn onward
    burn_switches: int  # engine on/off changes after the first burn
    seconds_before_burn: float
    seconds_braking: float  # first burn until descent is down to SETTLE_SPEED
    seconds_settling: float  # the rest of the flight, at or below SETTLE_SPEED
    settle_height: float  # px above the pad when descent first reaches SETTLE_SPEED


def profile(ep: Episode) -> FlightProfile:
    """Measures one landed flight, which must have fired its main engine."""
    states, main = np.stack(ep.states), np.stack(ep.actions)[:, 0]
    start = int(np.argmax(main > 0))
    burn = main[start:]
    descent = states[:, 3] * SPEED_SCALE
    # The first decision after the burn starts at which descent has dropped to SETTLE_SPEED.
    settle = start + int(np.argmax(descent[start:] <= SETTLE_SPEED))
    return FlightProfile(
        seconds=len(main) / DECISIONS_PER_SECOND,
        peak_descent=float(descent.max()),
        touchdown_descent=float(descent[-1]),
        first_burn_height=float(states[start, 1] * CONFIG.height / 2),
        burn_duty=float(burn.mean()),
        burn_switches=int(np.count_nonzero(np.diff(burn))),
        seconds_before_burn=start / DECISIONS_PER_SECOND,
        seconds_braking=(settle - start) / DECISIONS_PER_SECOND,
        seconds_settling=(len(main) - settle) / DECISIONS_PER_SECOND,
        settle_height=float(states[settle, 1] * CONFIG.height / 2),
    )


def landed_profiles(net: PolicyNet, rounds: int, seed: int) -> tuple[dict[str, float], list[FlightProfile]]:
    eps = Trainer(net, TRAIN, seed).fly(rounds, greedy)
    return rate_by_result(eps), [profile(e) for e in eps if result_of(e.world) == "landed"]


def main() -> None:
    path, rounds = Path(sys.argv[1]), int(sys.argv[2])
    net = PolicyNet(np.random.default_rng(0))
    net.load_from(path)
    rates, profiles = landed_profiles(net, rounds, 4321)
    print(f"{path}: {rounds} rounds  " + "  ".join(f"{r} {v:.4f}" for r, v in rates.items()))
    for name in FlightProfile.__dataclass_fields__:
        values = [getattr(p, name) for p in profiles]
        print(f"  {name:20s} mean {np.mean(values):7.2f}  min {np.min(values):7.2f}  max {np.max(values):7.2f}")


if __name__ == "__main__":
    main()

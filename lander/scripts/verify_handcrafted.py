"""Verify handcrafted weights without training or requiring a display.

From the repository root: python scripts/verify_handcrafted.py
Each flight has its own seed, so a failure can be replayed with
--seed-start SEED --episodes 1. Exits nonzero on any failure.
"""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lander.ai.handcrafted import build_policy
from lander.ai.policy import PolicyNet
from lander.ai.train import TRAIN
from lander.physics import CONFIG, DT, Outcome, hold, new_world


def require_equal(actual: PolicyNet, expected: PolicyNet) -> None:
    for name, a, b in zip(("W1", "b1", "W2", "b2"), actual.params, expected.params):
        if not np.array_equal(a, b) or not np.isfinite(a).all():
            raise ValueError(f"{name}: weights differ or contain nonfinite values")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights.npz"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    expected = build_policy()
    require_equal(build_policy(), expected)
    net = PolicyNet(np.random.default_rng(123))
    net.load_from(args.weights)
    require_equal(net, expected)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "roundtrip.npz"
        net.save(path)
        restored = PolicyNet(np.random.default_rng(456))
        restored.load_from(path)
        require_equal(restored, net)
    print("PASS: deterministic construction, file weights, and save/load round trip", flush=True)

    counts = Counter()
    decisions = side_firings = reversals = total_frames = 0
    first_burn_times = []
    longest_burns = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        world = new_world(CONFIG, random.Random(seed))
        previous_side = 0
        flight_frames = burn_frames = longest_burn = 0
        first_burn = None
        for _ in range(TRAIN.max_decisions):
            controls = net.greedy(world)
            if controls.main and first_burn is None:
                first_burn = flight_frames * DT
            if controls.left and controls.right:
                raise ValueError(f"Both side thrusters fired at seed {seed}")
            side = int(controls.right) - int(controls.left)
            reversals += int(side * previous_side < 0)
            previous_side = side
            side_firings += int(side != 0)
            decisions += 1
            for _frame in range(TRAIN.decide_every):
                world = hold(world, controls, DT, 1)
                total_frames += 1
                flight_frames += 1
                burn_frames = burn_frames + 1 if controls.main else 0
                longest_burn = max(longest_burn, burn_frames)
                if world.outcome is not Outcome.FLYING:
                    break
            if world.outcome is not Outcome.FLYING:
                break
        outcome = "timeout" if world.outcome is Outcome.FLYING else world.outcome.value
        counts[outcome] += 1
        if first_burn is not None:
            first_burn_times.append(first_burn)
        longest_burns.append(longest_burn * DT)
        if outcome != "landed":
            print(f"FAIL seed={seed}: {outcome}; lander={world.lander}; pad={world.pad}")
        if sum(counts.values()) % 500 == 0:
            print(f"{sum(counts.values())} flights: {dict(counts)}", flush=True)
    require_equal(net, expected)
    print(f"Final: {dict(counts)}; weights unchanged during evaluation")
    print(f"Mean flight time: {total_frames * DT / args.episodes:.2f}s; "
          f"side thrust: {side_firings / decisions:.1%} of decisions; "
          f"immediate side reversals: {reversals}")
    if first_burn_times:
        print(f"Mean first burn: {np.mean(first_burn_times):.2f}s after release; "
              f"mean longest continuous burn: {np.mean(longest_burns):.2f}s")
    return 0 if counts["landed"] == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())

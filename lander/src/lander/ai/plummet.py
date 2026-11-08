"""Hand-built weights for a suicide burn: plummet, then brake at the last moment.

Greedy flight fires a thruster when its logit is above zero, so the weights only
have to draw on/off boundaries; no training is involved.

Main engine: burn once the descent speed is above a braking parabola
v^2 = touchdown^2 + 2 * brake * (height - buffer). The "don't burn yet" region
under that curve is convex, so it is an intersection of half-planes. Each hidden
unit is one steep half-plane, a tangent to the parabola, and the engine fires if
any unit trips. The plan assumes less braking than the engine really has.

Side jets: chase a target sideways speed that rises like sqrt(distance to pad),
the fastest speed that can still be braked away in time. That curve is a least
squares mix of tanh units at several widths.

Run `python -m lander.ai.plummet ROUNDS` to measure it and write weights.npz.
"""
import sys
from pathlib import Path

import numpy as np

from lander.ai.policy import SPEED_SCALE, PolicyNet
from lander.ai.train import RESULTS, TRAIN, WEIGHTS, Trainer, greedy, rate_by_result
from lander.physics import CONFIG

PLAN_BRAKE = 50.0  # px/s^2 the burn is planned around; the engine really nets 60
BUFFER = 10.0  # px of extra height the plan keeps in hand
TOUCHDOWN_SPEED = 25.0  # px/s planned at the pad; the limit is 40
TANGENT_SPEEDS = np.arange(30.0, 271.0, 30.0)
EDGE_PX = 1.0  # height error that fully flips a tangent unit
LOGIT_GAIN = 10.0

SIDE_PLAN_ACCEL = 35.0  # px/s^2 planned; the jets really give 45
SIDE_NEAR_SLOPE = 0.8  # per second: target speed per px once close to the pad
SIDE_DEAD_ZONE = 4.0  # px/s of error left alone, so the jets don't alternate
SIDE_WIDTHS = np.array([10.0, 30.0, 80.0, 200.0, 500.0])  # px
VX_INPUT_WEIGHT = 0.1  # small enough that this tanh unit stays linear
SIDE_LOGIT_PER_SPEED = 0.05  # keeps logits small enough for float32 exp


def side_target(distance: np.ndarray) -> np.ndarray:
    """Target speed toward the pad at each distance in px."""
    return np.minimum(SIDE_NEAR_SLOPE * distance, np.sqrt(2 * SIDE_PLAN_ACCEL * distance))


def fit_side_target() -> np.ndarray:
    """Coefficients of tanh(distance / width), one per SIDE_WIDTHS, approximating side_target."""
    distance = np.linspace(0.0, float(CONFIG.width), 2000)
    basis = np.tanh(distance[:, None] / SIDE_WIDTHS)
    return np.linalg.lstsq(basis, side_target(distance), rcond=None)[0]


def wire_main(net: PolicyNet) -> int:
    """Tangent units on hidden 0..n-1, firing the main engine if any trips. Returns n."""
    n = len(TANGENT_SPEEDS)
    for i, v in enumerate(TANGENT_SPEEDS):
        net.W1[1, i] = -(CONFIG.height / 2) / EDGE_PX
        net.W1[3, i] = (v / PLAN_BRAKE) * SPEED_SCALE / EDGE_PX
        net.b1[i] = (BUFFER - (v * v + TOUCHDOWN_SPEED ** 2) / (2 * PLAN_BRAKE)) / EDGE_PX
    net.W2[:n, 0] = LOGIT_GAIN
    net.b2[0] = LOGIT_GAIN * (n - 0.5)
    return n


def wire_sides(net: PolicyNet, first: int) -> None:
    """Target-speed units from hidden `first`, then one linear vx unit; left mirrors right."""
    coefficients = fit_side_target()
    for j, (width, c) in enumerate(zip(SIDE_WIDTHS, coefficients), first):
        net.W1[0, j] = (CONFIG.width / 2) / width
        net.W2[j, 2] = c
    vx_unit = first + len(SIDE_WIDTHS)
    net.W1[2, vx_unit] = VX_INPUT_WEIGHT
    net.W2[vx_unit, 2] = -SPEED_SCALE / VX_INPUT_WEIGHT
    net.W2[:, 2] *= SIDE_LOGIT_PER_SPEED
    net.W2[:, 1] = -net.W2[:, 2]
    net.b2[1:] = -SIDE_DEAD_ZONE * SIDE_LOGIT_PER_SPEED


def build_policy() -> PolicyNet:
    net = PolicyNet(np.random.default_rng(0))
    for param in net.params:
        param.fill(0)
    wire_sides(net, wire_main(net))
    return net


def landing_rates(net: PolicyNet, rounds: int, seed: int) -> dict[str, float]:
    return rate_by_result(Trainer(net, TRAIN, seed).fly(rounds, greedy))


def main() -> None:
    rounds = int(sys.argv[1])
    net = build_policy()
    rates = landing_rates(net, rounds, 1234)
    print("  ".join(f"{r} {rates[r]:.4f}" for r in RESULTS))
    net.save(Path(WEIGHTS))
    print(f"saved {WEIGHTS}")


if __name__ == "__main__":
    main()

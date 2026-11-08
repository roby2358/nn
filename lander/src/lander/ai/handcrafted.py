"""Build fixed landing weights: free-fall, brake, and settle onto the pad.

Run ``python -m lander.ai.handcrafted`` to write weights.npz. Seven
hidden units encode an analytic controller; no training is performed.
"""
from pathlib import Path

import numpy as np

from lander.ai.policy import PolicyNet, SPEED_SCALE
from lander.ai.train import WEIGHTS
from lander.physics import CONFIG


def build_policy() -> PolicyNet:
    net = PolicyNet(np.random.default_rng(0))
    for param in net.params:
        param.fill(0)

    # Approach quickly, then brake sideways. A 4 px/s dead zone lets
    # both side jets remain off while velocity tracks the target.
    # Target vx = 45*tanh(dx/20) + 130*tanh(dx/300).
    net.W1[0, 0] = CONFIG.width / 2 / 20
    net.W1[0, 6] = CONFIG.width / 2 / 300
    net.W1[2, 1] = 0.01
    net.W2[0, 2] = 4.5
    net.W2[6, 2] = 13
    net.W2[1, 2] = -1000
    net.W2[:, 1] = -net.W2[:, 2]
    net.b2[1:] = -0.4

    # Burn when stopping distance reaches the height above the pad.
    # The denominator is just above 2*net braking acceleration so that
    # the switching signal stays positive during the sustained burn.
    # Leave a settling allowance for sampled controls and lateral travel.
    denominator = 2 * (CONFIG.main_thrust - CONFIG.gravity) + 1
    margin = 25 + 200 / denominator
    logit_scale = 0.001
    net.W1[1, 2] = 0.001
    net.W2[2, 0] = -logit_scale * CONFIG.height / 2 / 0.001

    # A symmetric pair of shifted tanh units approximates vy squared:
    # 2*tanh(b) - tanh(b+k*v) - tanh(b-k*v)
    #   = 2*tanh(b)*(1-tanh(b)**2)*k**2*v**2 + O(k**4*v**4).
    # Small k keeps the approximation accurate over reachable speeds.
    bias, k = 0.65848, 0.05
    t = np.tanh(bias)
    quadratic = 2 * t * (1 - t * t) * k * k
    coefficient = logit_scale * SPEED_SCALE**2 / denominator / quadratic
    net.W1[3, 3:5] = (k, -k)
    net.b1[3:5] = bias
    net.W2[3:5, 0] = -coefficient
    net.b2[0] = 2 * coefficient * t + logit_scale * margin

    # Release thrust at low descent speeds, giving a short final settling
    # phase and preventing the squared-speed term from firing on ascent.
    net.W1[3, 5] = 20
    net.W2[5, 0] = 10
    net.b2[0] -= 10
    return net


def main() -> None:
    build_policy().save(Path(WEIGHTS))
    print(f"Saved handcrafted policy to {WEIGHTS}")


if __name__ == "__main__":
    main()


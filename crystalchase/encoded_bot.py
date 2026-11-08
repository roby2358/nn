"""Algebraically encode AutoPilot as a deeper, ordinary tanh neural network.

No training or teacher fitting: speed lookup coefficients come from the bot's
formula, products/norms from tanh finite differences, and gates from its limits.
Inference is exclusively affine layers, tanh, and final argmax.
"""

from pathlib import Path

import numpy as np

from .config import (BOT_AIM_TOLERANCE, BOT_ARRIVE_SPEED, BOT_BRAKE_SHARE,
                     BOT_DEADBAND, BOT_MAX_SPEED, BOT_MIN_ERROR, DIST_SCALE,
                     N_ACTIONS, N_INPUTS, SPEED_SCALE, THRUST)


class EncodedBot:
    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]) -> None:
        width = N_INPUTS
        if not layers:
            raise ValueError("Neural pilot has no layers")
        for w, b in layers:
            if w.ndim != 2 or b.ndim != 1 or w.shape != (width, len(b)) or not len(b):
                raise ValueError("Invalid neural pilot layer dimensions")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("Neural pilot weights must be finite")
            width = len(b)
        if width != N_ACTIONS:
            raise ValueError("Neural pilot must have six action outputs")
        self.layers = layers

    def act(self, obs: np.ndarray) -> np.ndarray:
        h = obs
        for w, b in self.layers[:-1]:
            h = np.tanh(h @ w + b)
        w, b = self.layers[-1]
        return np.argmax(h @ w + b, axis=1)

    def save(self, path: Path) -> None:
        arrays = {"neural_layers": np.array(len(self.layers))}
        for i, (w, b) in enumerate(self.layers):
            arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "EncodedBot":
        with np.load(path) as f:
            count = int(f["neural_layers"])
            if not 1 <= count <= 16:
                raise ValueError("Invalid neural pilot layer count")
            return cls([(f[f"w{i}"].copy(), f[f"b{i}"].copy()) for i in range(count)])


def encode_autopilot() -> EncodedBot:
    # Smooth staircase of the exact speed law, including its maximum-speed cap.
    bins = 256
    cap_distance = (BOT_MAX_SPEED**2 - BOT_ARRIVE_SPEED**2) / (2 * THRUST * BOT_BRAKE_SHARE)
    edges = np.linspace(0.0, cap_distance, bins + 1)
    speeds = np.sqrt(BOT_ARRIVE_SPEED**2 + 2 * THRUST * BOT_BRAKE_SHARE * edges)
    delta = np.diff(speeds) / SPEED_SCALE
    centers = (edges[:-1] + edges[1:]) / (2 * DIST_SCALE)
    steepness = 2 * DIST_SCALE / (edges[1] - edges[0])
    w1, b1 = np.zeros((N_INPUTS, bins + 4)), np.zeros(bins + 4)
    w1[2, :bins], b1[:bins] = steepness, -steepness * centers
    for j, source in enumerate((0, 1, 3, 4)):
        w1[source, bins + j] = 0.01
    speed_coeff = np.zeros(bins + 4)
    speed_coeff[:bins] = delta / 2
    speed_bias = BOT_ARRIVE_SPEED / SPEED_SCALE + delta.sum() / 2

    # Multiply speed by each bearing component via symmetric tanh differences.
    anchor = np.arctanh(1 / np.sqrt(3))
    second = -4 / (3 * np.sqrt(3))
    eps = 0.01
    w2, b2 = np.zeros((bins + 4, 10)), np.zeros(10)
    ex, ey = np.zeros(10), np.zeros(10)
    signs = ((1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1))
    for bearing, start, error in ((1, 0, ex), (0, 4, ey)):
        for j, (sv, sb, coefficient) in enumerate(signs):
            unit = start + j
            w2[:, unit] = eps * sv * speed_coeff
            w2[bins + bearing, unit] += eps * sb / 0.01
            b2[unit] = anchor + eps * sv * speed_bias
            error[unit] = coefficient / (4 * eps**2 * second)
    for unit, source, error in ((8, bins + 2, ex), (9, bins + 3, ey)):
        w2[source, unit] = 0.1
        error[unit] = -1000.0

    # Squared error magnitude and copies of ex/ey for the decision layer.
    w3, b3 = np.zeros((10, 6)), np.zeros(6)
    norm = np.zeros(6)
    for start, error in ((0, ex), (2, ey)):
        for j, sign in enumerate((1, -1)):
            unit = start + j
            w3[:, unit] = sign * eps * error
            b3[unit] = anchor
            norm[unit] = 1 / (eps**2 * second)
    norm_bias = -4 * np.tanh(anchor) / (eps**2 * second)
    w3[:, 4], w3[:, 5] = 0.01 * ex, 0.01 * ey
    x, y = np.zeros(6), np.zeros(6)
    x[4], y[5] = 100.0, 100.0

    # Angular cones avoid atan2. Their intersection also requires ex > 0.
    w4, b4 = np.zeros((6, 6)), np.zeros(6)
    w4[:, 0] = 10000 * y
    w4[:, 1] = 10000 * (np.tan(BOT_DEADBAND) * x - y)
    w4[:, 2] = 10000 * (np.tan(BOT_DEADBAND) * x + y)
    w4[:, 3] = 10000 * (np.tan(BOT_AIM_TOLERANCE) * x - y)
    w4[:, 4] = 10000 * (np.tan(BOT_AIM_TOLERANCE) * x + y)
    w4[:, 5] = 100000 * norm
    b4[5] = 100000 * (norm_bias - (BOT_MIN_ERROR / SPEED_SCALE)**2)
    w5, b5 = np.zeros((6, N_ACTIONS)), np.zeros(N_ACTIONS)
    for action in range(N_ACTIONS):
        turn, thrust = action // 2 - 1, action % 2
        if turn:
            w5[0, action] = 0.5 * turn
        else:
            w5[1:3, action] = 1
            b5[action] = -1
        if thrust:
            w5[3:6, action] = 1
            b5[action] -= 2
    return EncodedBot([(w1, b1), (w2, b2), (w3, b3), (w4, b4), (w5, b5)])


def main() -> None:
    path = Path("racers/encoded-autopilot.npz")
    path.parent.mkdir(exist_ok=True)
    bot = encode_autopilot()
    bot.save(path)
    print(f"Wrote {path}; architecture: 6 -> " + " -> ".join(str(len(b)) for _, b in bot.layers))


if __name__ == "__main__":
    main()

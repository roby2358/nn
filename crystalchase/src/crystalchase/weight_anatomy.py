"""Read a sector net's design back out of its weights: `just cc-weights <reference.npz> <evolved.npz>`.
Pure algebra, nothing is flown. The output layer is factored into the sector polygon and two
velocity-error readouts over the hidden units; each readout is then expanded to second
order in the inputs."""

import sys
from dataclasses import dataclass

import numpy as np

from .calculated import SECTOR_DESIGN, _velocity_error_units
from .config import DIST_SCALE, N_HIDDEN, N_INPUTS, SPEED_SCALE
from .pilots import Brains

INPUTS = ("sinb", "cosb", "dist", "vf", "vs", "closing")
OFF_AXIS_ACTIONS = (0, 1, 2, 4, 5)   # every action, measured from straight-burn (action 3)
BURN_RIGHT, COAST_RIGHT, BURN_LEFT, COAST_LEFT, COAST_STRAIGHT = 5, 4, 1, 0, 2
GROUPS = {"dist x bearing, forward": range(0, 4), "dist x bearing, sideways": range(4, 8),
          "velocity": range(8, 10), "arrival": range(10, 12), "spare": range(12, 16)}
TERMS_SHOWN = 8


@dataclass(frozen=True)
class Anatomy:
    corners: dict[int, np.ndarray]    # action -> polygon corner, relative to straight-burn
    deadband: tuple[float, float]     # (right, left), radians
    aim_tolerance: tuple[float, float]
    min_error: float                  # px/s
    outside_share: float              # share of the output layer outside the two-readout form
    forward: dict[str, float]         # term -> coefficient of the forward velocity error
    sideways: dict[str, float]
    w1: np.ndarray
    b1: np.ndarray
    readouts: np.ndarray              # (N_HIDDEN, 2)


def polygon(differences: np.ndarray) -> dict[int, np.ndarray]:
    """Corners from the output layer, using the design's own readouts as the basis."""
    _, _, error_x, error_y = _velocity_error_units(*SECTOR_DESIGN[:2])
    x = differences.T @ error_x / (error_x @ error_x)
    y = differences.T @ error_y / (error_y @ error_y)
    return {a: np.array([x[i], y[i]]) for i, a in enumerate(OFF_AXIS_ACTIONS)}


def term_names() -> list[str]:
    pairs = [f"{INPUTS[i]}*{INPUTS[j]}" for i in range(N_INPUTS) for j in range(i, N_INPUTS)]
    return ["const", *INPUTS, *pairs]


def expansion(w1: np.ndarray, b1: np.ndarray, readout: np.ndarray) -> dict[str, float]:
    """Second-order Taylor coefficients of sum_u readout[u]*tanh(w1[:, u].obs + b1[u])."""
    tanh, slope = np.tanh(b1), 1.0 / np.cosh(b1) ** 2
    curve = -2.0 * tanh * slope
    linear = w1 @ (readout * slope)
    hessian = (w1 * readout * curve) @ w1.T
    quadratic = [hessian[i, j] * (0.5 if i == j else 1.0) for i in range(N_INPUTS) for j in range(i, N_INPUTS)]
    return dict(zip(term_names(), [float(readout @ tanh), *map(float, linear), *map(float, quadratic)]))


def anatomy(genome: np.ndarray) -> Anatomy:
    w1, b1, w2, b2 = (layer[0] for layer in Brains(genome[None, :]).layers)
    differences = w2[:, OFF_AXIS_ACTIONS] - w2[:, 3:4]
    corners = polygon(differences)
    basis = np.array([corners[a] for a in OFF_AXIS_ACTIONS])          # (5, 2)
    readouts = np.linalg.lstsq(basis, differences.T, rcond=None)[0].T   # (N_HIDDEN, 2)
    outside = differences - readouts @ basis.T
    forward, sideways = expansion(w1, b1, readouts[:, 0]), expansion(w1, b1, readouts[:, 1])

    right, right_coast = corners[BURN_RIGHT], corners[COAST_RIGHT] - corners[BURN_RIGHT]
    left, left_coast = corners[BURN_LEFT], corners[COAST_LEFT] - corners[BURN_LEFT]
    coast = corners[COAST_STRAIGHT]
    offset = np.tanh(b1) @ (w2[:, 2] - w2[:, 3]) + b2[2] - b2[3]
    bias = offset - coast[0] * forward["const"] - coast[1] * sideways["const"]
    return Anatomy(corners,
                   (float(np.arctan2(-right[0], right[1])), float(np.arctan2(-left[0], -left[1]))),
                   (float(np.arctan2(-right_coast[0], right_coast[1])), float(np.arctan2(-left_coast[0], -left_coast[1]))),
                   float(-bias * SPEED_SCALE / coast[0]),
                   float(np.sum(outside ** 2) / np.sum(differences ** 2)),
                   forward, sideways, w1, b1, readouts)


def design_rows(a: Anatomy) -> list[tuple[str, float, float]]:
    """(name, forward or right value, sideways or left value)."""
    gain = DIST_SCALE / SPEED_SCALE
    return [("arrival speed (px/s)", a.forward["cosb"] * SPEED_SCALE, a.sideways["sinb"] * SPEED_SCALE),
            ("range gain", a.forward["cosb*dist"] / gain, a.sideways["sinb*dist"] / gain),
            ("velocity coefficient", a.forward["vf"], a.sideways["vs"]),
            ("deadband (rad)", *a.deadband), ("aim tolerance (rad)", *a.aim_tolerance),
            ("min error (px/s)", a.min_error, a.min_error)]


def print_design(reference: Anatomy, evolved: Anatomy) -> None:
    print(f"{'':24s}{'reference':>20s}{'evolved':>20s}")
    print(f"{'':24s}{'fwd/right':>10s}{'side/left':>10s}{'fwd/right':>10s}{'side/left':>10s}")
    for (name, a, b), (_, c, d) in zip(design_rows(reference), design_rows(evolved)):
        print(f"{name:24s}{a:10.3f}{b:10.3f}{c:10.3f}{d:10.3f}")
    print(f"output layer outside the two-readout form: {reference.outside_share:.2e} -> {evolved.outside_share:.2e}")


def print_new_terms(label: str, reference: dict[str, float], evolved: dict[str, float]) -> None:
    moved = sorted(term_names(), key=lambda t: -abs(evolved[t] - reference[t]))[:TERMS_SHOWN]
    print(f"{label} velocity error, largest changes (units of {SPEED_SCALE:.0f} px/s):")
    for term in moved:
        print(f"  {term:18s}{reference[term]:10.3f} ->{evolved[term]:10.3f}")


def print_units(reference: Anatomy, evolved: Anatomy) -> None:
    print(f"{'hidden units':28s}{'|w1| ref':>10s}{'dw1 rms':>10s}{'db1 rms':>10s}{'|readout| ref':>15s}{'evolved':>10s}")
    for name, units in GROUPS.items():
        u = list(units)
        before, after = np.abs(reference.readouts[u]).max(), np.abs(evolved.readouts[u]).max()
        print(f"{name:28s}{np.abs(reference.w1[:, u]).max():10.3f}{np.sqrt(np.mean((evolved.w1[:, u] - reference.w1[:, u]) ** 2)):10.3f}"
              f"{np.sqrt(np.mean((evolved.b1[u] - reference.b1[u]) ** 2)):10.3f}{before:15.3f}{after:10.3f}")


def main() -> None:
    reference = anatomy(Brains.load(sys.argv[1]).genomes[0])
    evolved = anatomy(Brains.load(sys.argv[2]).genomes[0])
    assert N_HIDDEN == 16
    print_design(reference, evolved)
    print_units(reference, evolved)
    print_new_terms("forward", reference.forward, evolved.forward)
    print_new_terms("sideways", reference.sideways, evolved.sideways)


if __name__ == "__main__":
    main()

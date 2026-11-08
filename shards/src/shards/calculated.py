"""A turret pilot built algebraically, with no training: `just cc-calculated <out.npz>`.

It sits still, leads the nearest rock and fires only when the shot would land. The result is an
ordinary Brains genome, the same 27 -> 24 -> 3 net that evolution uses, so only five hidden units
do anything and the other nineteen are zeros.

For a target at bearing b with sideways relative speed v, the shot that intercepts it leaves at
sin(aim) = v / SHOT_SPEED, whatever the distance. So near the target the aiming error is linear in
the inputs: e = sin(b) + v / SHOT_SPEED. A linear readout steers on e, and a soft AND of four
tanh half-planes pulls the trigger when |e| is inside a tolerance, the rock is ahead and in range.
"""

import sys
from pathlib import Path

import numpy as np

from .config import (CMD_FIRE, CMD_THRUST, CMD_TURN, N_HIDDEN, N_INPUTS, N_OUTPUTS, ROCK_CLOSE, ROCK_COS, ROCK_RADIUS,
                     ROCK_SIN, ROCK_V_SIDEWAYS, SHOT_SPEED, SPEED_SCALE, rock_input)
from .pilots import Brains

SIN, COS, CLOSE = (rock_input(0, field) for field in (ROCK_SIN, ROCK_COS, ROCK_CLOSE))
V_SIDE, SIZE = (rock_input(0, field) for field in (ROCK_V_SIDEWAYS, ROCK_RADIUS))
ERROR, UNDER, OVER, AHEAD, IN_RANGE = range(5)      # the hidden units in use

LEAD = SPEED_SCALE / SHOT_SPEED
LINEAR = 0.05           # small enough that tanh(LINEAR * x) is LINEAR * x
TURN_GAIN = 12.0        # turn command per radian of aiming error
SHARP = 100.0           # how hard the half-plane units switch
COS_MIN = 0.5           # the rock counts as ahead above this
CLOSE_MIN = 0.1         # closeness below this is out of shot range, and an empty slot is 0
TOL_BASE = 0.0          # aiming tolerance in sin units, an affine fit to the rock's angular radius
TOL_CLOSE = 0.05
TOL_SIZE = 0.05
FIRE_BIAS = -3.5        # four units at +1 clear this; any one at -1 does not
NO_THRUST = -6.0


def _half_plane(w1: np.ndarray, b1: np.ndarray, unit: int, sign: float) -> None:
    """Unit is +1 when sign * e is inside the tolerance for this rock's range and size."""
    w1[SIN, unit] = -sign * SHARP
    w1[V_SIDE, unit] = -sign * SHARP * LEAD
    w1[CLOSE, unit] = SHARP * TOL_CLOSE
    w1[SIZE, unit] = SHARP * TOL_SIZE
    b1[unit] = SHARP * TOL_BASE


def turret() -> Brains:
    w1, b1 = np.zeros((N_INPUTS, N_HIDDEN)), np.zeros(N_HIDDEN)
    w2, b2 = np.zeros((N_HIDDEN, N_OUTPUTS)), np.zeros(N_OUTPUTS)

    w1[SIN, ERROR], w1[V_SIDE, ERROR] = LINEAR, LINEAR * LEAD
    w2[ERROR, CMD_TURN] = TURN_GAIN / LINEAR

    _half_plane(w1, b1, UNDER, 1.0)
    _half_plane(w1, b1, OVER, -1.0)
    w1[COS, AHEAD], b1[AHEAD] = SHARP, -SHARP * COS_MIN
    w1[CLOSE, IN_RANGE], b1[IN_RANGE] = SHARP, -SHARP * CLOSE_MIN
    w2[[UNDER, OVER, AHEAD, IN_RANGE], CMD_FIRE] = 1.0
    b2[CMD_FIRE] = FIRE_BIAS

    b2[CMD_THRUST] = NO_THRUST
    return Brains(np.concatenate([w1.ravel(), b1, w2.ravel(), b2])[None, :])


def main() -> None:
    out = Path(sys.argv[1])
    turret().save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

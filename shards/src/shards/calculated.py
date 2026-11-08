"""A hunter pilot built algebraically, with no training: `just cc-calculated <out.npz>`.

It leads the nearest rock and fires only when the shot would land, and moves by reading a danger
field. The result is an ordinary Brains genome, the same 27 -> 24 -> 3 net that evolution uses, so
the spare hidden units are zeros.

For a target at bearing b with sideways relative speed v, the shot that intercepts it leaves at
sin(aim) = v / SHOT_SPEED, whatever the distance. So near the target the aiming error is linear in
the inputs: e = sin(b) + v / SHOT_SPEED. A linear readout steers on e, and a soft AND of four
tanh half-planes pulls the trigger when |e| is inside a tolerance, the rock is ahead and in range.

The danger field gives each of the four visible rocks two half-planes in closeness, bearing and
forward relative speed, so a moving rock counts from where it will be LOOK_AHEAD seconds on. One
says the rock blocks the space ahead, the other that it threatens from behind. Thrust is their
balance: advance while nothing blocks the way and the ship is under cruising speed, hold when
something does, and burn away when danger behind outweighs danger ahead. A left/right pair per
rock turns away from a close rock, outweighing the aim.
"""

import sys
from pathlib import Path

import numpy as np

from .config import (CMD_FIRE, CMD_THRUST, CMD_TURN, N_HIDDEN, N_INPUTS, N_OUTPUTS, NEAREST, ROCK_CLOSE, ROCK_COS,
                     ROCK_RADIUS, ROCK_SIN, ROCK_V_FORWARD, ROCK_V_SIDEWAYS, SENSE_RANGE, SHOT_SPEED, SPEED_SCALE,
                     V_FORWARD, rock_input)
from .pilots import Brains

SIN, COS, CLOSE = (rock_input(0, field) for field in (ROCK_SIN, ROCK_COS, ROCK_CLOSE))
V_SIDE, SIZE = (rock_input(0, field) for field in (ROCK_V_SIDEWAYS, ROCK_RADIUS))
ERROR, UNDER, OVER, AHEAD, IN_RANGE, CRUISING = range(6)   # the hidden units not tied to one rock of the field
BLOCKED, CHASED, DODGE_POS, DODGE_NEG = (6 + kind * NEAREST for kind in range(4))   # first of NEAREST units of each

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

LOOK_AHEAD = 1.0        # seconds; a moving rock counts from where it will be this much later
CLEAR_AHEAD = 160.0     # px of clear space a rock dead ahead must leave for the way to count as open
BLOCK_SLANT = 0.1       # how much less a rock counts as blocking per unit of bearing away from dead ahead
DANGER = 75.0           # px of clear space at which a rock abeam counts as a threat
CHASE_SLANT = 0.15      # how much sooner a rock behind is a threat, and later one ahead
CRUISE = 140.0          # px/s of forward speed above which the way ahead earns no more thrust
OPEN_WEIGHT = 3.0       # what an open way, or cruising speed, says to thrust
BLOCK_WEIGHT = 3.0      # one blocking rock cancels an open way
CHASE_WEIGHT = 4.0      # one threat behind outweighs one rock blocking the way
DODGE_WEIGHT = 8.0      # one dodge pair outvotes the aiming error

LOOK = LOOK_AHEAD * SPEED_SCALE / SENSE_RANGE   # closeness gained per unit of the speed input
CLEAR_CLOSE = 1.0 - CLEAR_AHEAD / SENSE_RANGE
DANGER_CLOSE = 1.0 - DANGER / SENSE_RANGE
DODGE_SLANT = 1.0 - DANGER_CLOSE    # a rock abeam trips at DANGER; dead ahead it never does


def _half_plane(w1: np.ndarray, b1: np.ndarray, unit: int, sign: float) -> None:
    """Unit is +1 when sign * e is inside the tolerance for this rock's range and size."""
    w1[SIN, unit] = -sign * SHARP
    w1[V_SIDE, unit] = -sign * SHARP * LEAD
    w1[CLOSE, unit] = SHARP * TOL_CLOSE
    w1[SIZE, unit] = SHARP * TOL_SIZE
    b1[unit] = SHARP * TOL_BASE


def _danger_field(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray) -> None:
    """Give each visible rock its blocked and chased units, weighed against each other on thrust."""
    for rock in range(NEAREST):
        cos, close, v_forward = (rock_input(rock, field) for field in (ROCK_COS, ROCK_CLOSE, ROCK_V_FORWARD))
        w1[close, BLOCKED + rock], w1[cos, BLOCKED + rock] = SHARP, SHARP * BLOCK_SLANT
        w1[v_forward, BLOCKED + rock] = -SHARP * LOOK       # coming at the nose counts as closer
        b1[BLOCKED + rock] = -SHARP * (CLEAR_CLOSE + BLOCK_SLANT)
        w1[close, CHASED + rock], w1[cos, CHASED + rock] = SHARP, -SHARP * CHASE_SLANT
        w1[v_forward, CHASED + rock] = SHARP * LOOK         # coming up from behind counts as closer
        b1[CHASED + rock] = -SHARP * DANGER_CLOSE
    w2[BLOCKED:BLOCKED + NEAREST, CMD_THRUST] = -BLOCK_WEIGHT
    w2[CHASED:CHASED + NEAREST, CMD_THRUST] = CHASE_WEIGHT


def _dodges(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray) -> None:
    """Give each visible rock a pair of units that turns away from it."""
    for rock in range(NEAREST):
        sin, close = rock_input(rock, ROCK_SIN), rock_input(rock, ROCK_CLOSE)
        for first, side in ((DODGE_POS, 1.0), (DODGE_NEG, -1.0)):
            w1[close, first + rock], w1[sin, first + rock] = SHARP, side * SHARP * DODGE_SLANT
            b1[first + rock] = -SHARP
            w2[first + rock, CMD_TURN] = -side * DODGE_WEIGHT     # the pair cancels while both are off


def hunter() -> Brains:
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

    w1[V_FORWARD, CRUISING], b1[CRUISING] = -SHARP, SHARP * CRUISE / SPEED_SCALE
    w2[CRUISING, CMD_THRUST] = OPEN_WEIGHT
    # cancels the quiet field, where each BLOCKED unit at -1 adds BLOCK_WEIGHT and each CHASED unit subtracts CHASE_WEIGHT
    b2[CMD_THRUST] = NEAREST * (CHASE_WEIGHT - BLOCK_WEIGHT)

    _danger_field(w1, b1, w2)
    _dodges(w1, b1, w2)
    return Brains(np.concatenate([w1.ravel(), b1, w2.ravel(), b2])[None, :])


def main() -> None:
    out = Path(sys.argv[1])
    hunter().save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""A charging tank built algebraically, with no training: `just cc-calculated <out.npz>`.

The result is an ordinary genome in the 49 -> 54 -> 17 -> 3 net that evolution uses. The first hidden layer
cuts the inputs with sharp half-planes, the second ANDs them into facts, and the outputs weigh the facts.

Shooting is (any enemy in the sights that can't answer) AND NOT (any teammate in the danger zone). An enemy can't
answer if its gun is reloading, if it points too far off this tank to swing back before the shell lands, or if it's
beyond the reach of a direct hit, where either tank's shell can only splash. Each
visible enemy gets one AND unit and each visible teammate gets another. One line of sight is enough to fire, and because a single
veto outweighs all four sights, one teammate is enough to hold. A shell only goes off on contact or at full
range, so the sights are as wide as a tank up close, and the danger zone is as wide as a blast the whole way
out. Bearing tolerance is an affine fit in closeness to the shape width / distance.

Driving is a charge at the nearest enemy that stops short of the distance where a hit would catch the shooter.
Only a pack charges: a tank with two teammates close by. Any other tank stops at firing range and shells from
there, just past the reach of a direct hit. It holds its ground rather than backing away, because it can't see the edge.
Turning is linear in the bearing, and the treads share one drive and one turn signal. A hard turn cancels the
drive, so a tank spins in place until it's pointing roughly the right way and only then rolls.

On the approach, teammates bias the heading to hold them just outside a blast's reach: steer away from any
that are too close, and toward the nearest if it has drifted off. Once an enemy is in range the bias switches
off so the gun stays on target, and spacing can only slow a tank that's about to run into a teammate ahead.

A tank in reach of a nearest enemy that could answer veers about 30 degrees off it and keeps closing. That tilt
holds it out of the firing rule at long range, but the allowance for swinging back shrinks with range. Near 90 px
each tank's tilt is too big to answer from, so both turn back in, and the first to come round keeps the other
tilted: a loaded gun on it is a reason to veer. That breaks the tie between two tanks locked head-on.
"""

import math
import sys
from pathlib import Path

import numpy as np

from .config import (CLOSE, CMD_LEFT, CMD_RIGHT, CMD_SHOOT, CONTACT_INPUTS, COS, GUN_COS, LOADED, N_HIDDEN_1,
                     N_HIDDEN_2, N_INPUTS, N_OUTPUTS, NEAREST, SELF_INPUTS, SENSE_RANGE, SHELL_BURST, SHELL_SPEED, SIN,
                     TANK_R, TURN_RATE)
from .pilots import save_champion

SHARP = 100.0           # how hard a first-layer half-plane switches
RANGE_SHARP = 8.0       # steeper still on closeness for a range gate, which must tell apart distances a few px apart
GATE = 5.0              # how hard a second-layer AND switches, on inputs of +-1
LINEAR = 0.05           # small enough that tanh(LINEAR * x) is LINEAR * x, for a signal passing through a layer

SIGHT_NEAR, SIGHT_NEAR_TOL = 60.0, 0.055     # px, and |sin bearing| that puts a contact on target at that range
SIGHT_FAR, SIGHT_FAR_TOL = 190.0, 0.02
LINE_NEAR, LINE_NEAR_TOL = 45.0, 0.30        # the same for the teammate danger zone, wider than the sights
LINE_FAR, LINE_FAR_TOL = 200.0, 0.075
AHEAD_SLOPE = 0.5       # closes the zones behind the tank: costs nothing dead ahead and everything dead astern
DIRECT_RANGE = SHELL_BURST + TANK_R          # px; beyond this a shell can't touch a tank, only splash it
STANDOFF_RANGE = DIRECT_RANGE + 5.0          # px; a tank without a pack stops here, where only splash can reach it
FIRE_RANGE = STANDOFF_RANGE + 3.0            # px; the longest shot, a little past the stand-off so that a stopped tank has it
PACK_R = 80.0           # px; a tank whose second-nearest teammate is this close is in a pack of three
FRIEND_REACH = FIRE_RANGE + 5.0
MIN_RANGE = 25.0        # px; hold fire closer than this, because a hit would catch the shooter
STOP_RANGE = 30.0       # px; the charge halts here and backs off inside it
STOP_SOFTNESS = 40.0    # drive per unit of closeness around the stop distance

DRIVE = 1.5             # tread signal at full charge
TURN_GAIN = 20.0        # turn signal per unit of sin bearing
ASTERN_COS = -0.9       # an enemy this far behind gets a fixed kick, because sin alone can't break the tie
VEER_DEGREES = 30.0     # the same kick, against TURN_GAIN, settles the heading this far off an enemy that could answer
KICK_TURN = TURN_GAIN * math.sin(math.radians(VEER_DEGREES))
ANSWER_NEAR, ANSWER_FAR = 60.0, 190.0        # px; the ranges the could-it-answer allowance is fitted through
UNLOADED_MARGIN = 3.0   # outweighs any gun angle, so a reloading gun can never answer

REPEL_R = 22.0          # px; just outside a blast, measured to a teammate 30 degrees off the nose
REPEL_SLANT = 0.042     # how much sooner a teammate abeam counts as too close
ATTRACT_R = 30.0        # px; a nearest teammate farther than this gets steered back toward
SPACE_TURN = 6.0        # against TURN_GAIN, this tilts the heading about 17 degrees off the enemy
BLOCK_R = 22.0          # px; a teammate this close ahead slows the tank
BLOCK_SLANT = 0.042
BLOCK_BRAKE = 1.5       # drive removed per blocking teammate

FIRE_WEIGHT = 2.0       # one sight on target clears the threshold
VETO_WEIGHT = 10.0      # one veto outweighs all four sights

# first hidden layer
SIGHT_L, SIGHT_R, REACH = (kind * NEAREST for kind in range(3))     # the first of NEAREST units of each
NOT_POINT_BLANK = 3 * NEAREST
LINE_L, LINE_R, FRIEND_NEAR = (NOT_POINT_BLANK + 1 + kind * NEAREST for kind in range(3))
AIM, RANGE, APPROACH, ASTERN = range(FRIEND_NEAR + NEAREST, FRIEND_NEAR + NEAREST + 4)
REPEL_POS, REPEL_NEG, BLOCK = (ASTERN + 1 + kind * NEAREST for kind in range(3))
DRIFTED, HAS_FRIEND, FRIEND_SIDE = range(BLOCK + NEAREST, BLOCK + NEAREST + 3)
CANT_ANSWER = FRIEND_SIDE + 1   # the first of NEAREST units
BEYOND_DIRECT = CANT_ANSWER + NEAREST   # the first of NEAREST units
STANDOFF, PACK = range(BEYOND_DIRECT + NEAREST, BEYOND_DIRECT + NEAREST + 2)

# second hidden layer
FIRE, VETO = 0, NEAREST     # the first of NEAREST units of each
AIM_2, RANGE_2, KICK, SPACE_POS, SPACE_NEG, ATTRACT_POS, ATTRACT_NEG, BLOCKED, HOLD = range(2 * NEAREST, N_HIDDEN_2)


def closeness(px: float) -> float:
    return 1.0 - px / SENSE_RANGE


def enemy(k: int, field: int) -> int:
    return SELF_INPUTS + k * CONTACT_INPUTS + field


def friend(j: int, field: int) -> int:
    return SELF_INPUTS + (NEAREST + j) * CONTACT_INPUTS + field


def tolerance(near_px: float, near_tol: float, far_px: float, far_tol: float) -> tuple[float, float]:
    """(slope, intercept) of a bearing tolerance that's linear in closeness, through two ranges."""
    slope = (near_tol - far_tol) / (closeness(near_px) - closeness(far_px))
    return slope, far_tol - slope * closeness(far_px)


def swing_back_cos(px: float) -> float:
    """Cos of the angle a gun can swing through in the time a shell takes to cover this range."""
    return math.cos(TURN_RATE * px / SHELL_SPEED)


class Wiring:
    """The six arrays of the net, and the few ways this design connects them."""

    def __init__(self) -> None:
        self.w1, self.b1 = np.zeros((N_INPUTS, N_HIDDEN_1)), np.zeros(N_HIDDEN_1)
        self.w2, self.b2 = np.zeros((N_HIDDEN_1, N_HIDDEN_2)), np.zeros(N_HIDDEN_2)
        self.w3, self.b3 = np.zeros((N_HIDDEN_2, N_OUTPUTS)), np.zeros(N_OUTPUTS)

    def genome(self) -> np.ndarray:
        return np.concatenate([a.ravel() for a in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)])

    def plane(self, unit: int, weights: dict[int, float], offset: float) -> None:
        """First-layer unit, +1 where sum(weight * input) + offset is positive and -1 where it's negative."""
        for source, weight in weights.items():
            self.w1[source, unit] = SHARP * weight
        self.b1[unit] = SHARP * offset

    def range_gate(self, unit: int, close: int, px: float, sign: float) -> None:
        """First-layer unit, +1 for a contact nearer than px (sign 1.0) or farther than it (sign -1.0)."""
        self.plane(unit, {close: sign * RANGE_SHARP}, -sign * RANGE_SHARP * closeness(px))

    def bearing_window(self, left: int, right: int, sin: int, cos: int, close: int,
                       shape: tuple[float, float]) -> None:
        """Two units, both +1 when a contact's |sin| is under the tolerance `shape` allows and it's ahead."""
        slope, intercept = shape
        for unit, side in ((left, -1.0), (right, 1.0)):
            self.plane(unit, {sin: side, close: slope, cos: AHEAD_SLOPE}, intercept - AHEAD_SLOPE)

    def all_of(self, unit: int, sources: list[tuple[int, float]]) -> None:
        """Second-layer unit, +1 only when every source is at its given sign."""
        for source, sign in sources:
            self.w2[source, unit] = GATE * sign
        self.b2[unit] = -GATE * (len(sources) - 1)

    def all_and_either(self, unit: int, sources: list[int], either: tuple[int, int]) -> None:
        """Second-layer unit, +1 only when every source is +1 and at least one of `either` is too."""
        self.w2[sources, unit] = 2.0 * GATE
        self.w2[list(either), unit] = GATE
        self.b2[unit] = -GATE * (2.0 * len(sources) - 1.0)

    def relay(self, source: int, unit: int) -> None:
        """Carry a first-layer signal through the second layer unchanged but for scale LINEAR."""
        self.w2[source, unit] = LINEAR

    def treads(self, unit: int, drive: float, turn: float) -> None:
        self.w3[unit, CMD_LEFT] += drive + turn
        self.w3[unit, CMD_RIGHT] += drive - turn

    def tread_bias(self, drive: float, turn: float) -> None:
        self.b3[CMD_LEFT] += drive + turn
        self.b3[CMD_RIGHT] += drive - turn


def wire_shooting(net: Wiring) -> None:
    sights = tolerance(SIGHT_NEAR, SIGHT_NEAR_TOL, SIGHT_FAR, SIGHT_FAR_TOL)
    line = tolerance(LINE_NEAR, LINE_NEAR_TOL, LINE_FAR, LINE_FAR_TOL)
    answer_slope, answer_intercept = tolerance(ANSWER_NEAR, swing_back_cos(ANSWER_NEAR),
                                               ANSWER_FAR, swing_back_cos(ANSWER_FAR))
    net.plane(NOT_POINT_BLANK, {enemy(0, CLOSE): -1.0}, closeness(MIN_RANGE))
    for k in range(NEAREST):
        net.bearing_window(SIGHT_L + k, SIGHT_R + k, enemy(k, SIN), enemy(k, COS), enemy(k, CLOSE), sights)
        net.range_gate(REACH + k, enemy(k, CLOSE), FIRE_RANGE, 1.0)
        net.plane(CANT_ANSWER + k, {enemy(k, GUN_COS): -1.0, enemy(k, CLOSE): answer_slope,
                                    enemy(k, LOADED): -UNLOADED_MARGIN}, answer_intercept + UNLOADED_MARGIN)
        net.range_gate(BEYOND_DIRECT + k, enemy(k, CLOSE), DIRECT_RANGE, -1.0)
        net.all_and_either(FIRE + k, [SIGHT_L + k, SIGHT_R + k, REACH + k, NOT_POINT_BLANK],
                           (CANT_ANSWER + k, BEYOND_DIRECT + k))
    for j in range(NEAREST):
        net.bearing_window(LINE_L + j, LINE_R + j, friend(j, SIN), friend(j, COS), friend(j, CLOSE), line)
        net.plane(FRIEND_NEAR + j, {friend(j, CLOSE): 1.0}, -closeness(FRIEND_REACH))
        net.all_of(VETO + j, [(LINE_L + j, 1.0), (LINE_R + j, 1.0), (FRIEND_NEAR + j, 1.0)])
    net.w3[FIRE:FIRE + NEAREST, CMD_SHOOT] = FIRE_WEIGHT / 2.0
    net.w3[VETO:VETO + NEAREST, CMD_SHOOT] = -VETO_WEIGHT / 2.0
    net.b3[CMD_SHOOT] = NEAREST * (FIRE_WEIGHT - VETO_WEIGHT) / 2.0 - FIRE_WEIGHT / 2.0


def wire_charge(net: Wiring) -> None:
    net.w1[enemy(0, SIN), AIM] = LINEAR
    net.relay(AIM, AIM_2)
    net.treads(AIM_2, 0.0, TURN_GAIN / LINEAR ** 2)

    net.w1[enemy(0, CLOSE), RANGE], net.b1[RANGE] = -STOP_SOFTNESS, STOP_SOFTNESS * closeness(STOP_RANGE)
    net.relay(RANGE, RANGE_2)
    net.treads(RANGE_2, DRIVE / LINEAR, 0.0)

    net.plane(ASTERN, {enemy(0, COS): -1.0}, ASTERN_COS)
    wire_kick(net)
    wire_hold(net)


def wire_kick(net: Wiring) -> None:
    """One unit for a fixed turn: enemy astern, OR (nearest enemy in direct reach AND it could answer)."""
    net.w2[ASTERN, KICK] = 2.0 * GATE
    net.w2[BEYOND_DIRECT, KICK] = -GATE
    net.w2[CANT_ANSWER, KICK] = -GATE
    net.b2[KICK] = GATE
    net.treads(KICK, 0.0, KICK_TURN / 2.0)
    net.tread_bias(0.0, KICK_TURN / 2.0)


def wire_hold(net: Wiring) -> None:
    """Without a pack, the nearest enemy being inside the stand-off cancels the charge's drive, and nothing more."""
    net.range_gate(STANDOFF, enemy(0, CLOSE), STANDOFF_RANGE, 1.0)
    net.plane(PACK, {friend(1, CLOSE): 1.0}, -closeness(PACK_R))
    net.all_of(HOLD, [(PACK, -1.0), (STANDOFF, 1.0)])
    net.treads(HOLD, -DRIVE / 2.0, 0.0)
    net.tread_bias(-DRIVE / 2.0, 0.0)


def wire_spacing(net: Wiring) -> None:
    net.plane(APPROACH, {enemy(0, CLOSE): -1.0}, closeness(FIRE_RANGE))
    repel_at = closeness(REPEL_R) + 0.5 * REPEL_SLANT
    block_at = closeness(BLOCK_R) + 0.5 * BLOCK_SLANT
    for j in range(NEAREST):
        net.plane(REPEL_POS + j, {friend(j, CLOSE): 1.0, friend(j, SIN): REPEL_SLANT}, -repel_at)
        net.plane(REPEL_NEG + j, {friend(j, CLOSE): 1.0, friend(j, SIN): -REPEL_SLANT}, -repel_at)
        net.plane(BLOCK + j, {friend(j, CLOSE): 1.0, friend(j, COS): BLOCK_SLANT}, -block_at)
    wire_repulsion(net)
    wire_attraction(net)
    wire_blocking(net)


def wire_repulsion(net: Wiring) -> None:
    """Two units vote on the net count of too-close teammates per side, and only on the approach."""
    for unit, toward, away in ((SPACE_POS, REPEL_NEG, REPEL_POS), (SPACE_NEG, REPEL_POS, REPEL_NEG)):
        net.w2[toward:toward + NEAREST, unit] = GATE / 2.0
        net.w2[away:away + NEAREST, unit] = -GATE / 2.0
        net.w2[APPROACH, unit] = GATE * 3.0
        net.b2[unit] = -GATE * 3.5
    net.treads(SPACE_POS, 0.0, SPACE_TURN / 2.0)
    net.treads(SPACE_NEG, 0.0, -SPACE_TURN / 2.0)


def wire_attraction(net: Wiring) -> None:
    net.plane(DRIFTED, {friend(0, CLOSE): -1.0}, closeness(ATTRACT_R))
    net.plane(HAS_FRIEND, {friend(0, CLOSE): 1.0}, -0.01)
    net.plane(FRIEND_SIDE, {friend(0, SIN): 1.0}, 0.0)
    for unit, side, turn in ((ATTRACT_POS, 1.0, SPACE_TURN), (ATTRACT_NEG, -1.0, -SPACE_TURN)):
        net.all_of(unit, [(DRIFTED, 1.0), (HAS_FRIEND, 1.0), (FRIEND_SIDE, side), (APPROACH, 1.0)])
        net.treads(unit, 0.0, turn / 2.0)


def wire_blocking(net: Wiring) -> None:
    """A count of blocking teammates, scaled through the second layer and zero when there are none."""
    net.w2[BLOCK:BLOCK + NEAREST, BLOCKED] = LINEAR
    net.b2[BLOCKED] = NEAREST * LINEAR
    net.treads(BLOCKED, -BLOCK_BRAKE / (2.0 * LINEAR), 0.0)


def charger() -> np.ndarray:
    net = Wiring()
    wire_shooting(net)
    wire_charge(net)
    wire_spacing(net)
    return net.genome()


def main() -> None:
    out = Path(sys.argv[1])
    save_champion(out, charger())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

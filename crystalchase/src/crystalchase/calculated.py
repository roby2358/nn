"""Construct a velocity-chasing neural pilot algebraically, without training.

Run ``python -m crystalchase.calculated`` to write weights.npz. The resulting
network uses the ordinary Brains architecture and needs no custom runtime.
"""

from pathlib import Path

import numpy as np

from .config import DIST_SCALE, N_ACTIONS, N_HIDDEN, N_INPUTS, PICKUP_R, SPEED_SCALE, TURN_RATE
from .pilots import Brains


def long_arc_brain(*, arrival_speed: float = 5 * PICKUP_R,
                   range_gain: float = 0.4) -> Brains:
    """Aim broad full-burn arcs using desired speed 130 + 0.4*distance.

    At long range the ship aims farther ahead; near a crystal the lower target
    speed makes it turn back to brake. All weights are constructed algebraically.
    Twelve units approximate (base + gain*distance)*bearing - velocity.
    Four tanh ridge functions approximate each distance*bearing product using
    a symmetric second difference at tanh's maximum-curvature point. Four
    more units pass through the two bearings and two velocity components.
    The outputs select turn direction, or straight thrust inside an error cone.
    """
    w1, b1, error_x, error_y = _velocity_error_units(arrival_speed, range_gain)
    w2 = np.zeros((N_HIDDEN, N_ACTIONS))
    b2 = np.zeros(N_ACTIONS)
    for action in range(N_ACTIONS):
        turn, thrust = action // 2 - 1, action % 2
        w2[:, action] = turn * error_y if turn else 0.06 * error_x
        b2[action] = (0.01 if turn == 0 else 0.0) + thrust
    return Brains(np.concatenate([w1.ravel(), b1, w2.ravel(), b2])[None, :])


def _velocity_error_units(arrival_speed: float, range_gain: float
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Twelve hidden units whose readouts approximate the velocity error in ship axes,
    (arrival_speed + range_gain*distance)*bearing - velocity, in SPEED_SCALE units.
    Returns (w1, b1, error_x, error_y); error_x and error_y are readouts over hidden units."""
    w1 = np.zeros((N_INPUTS, N_HIDDEN))
    b1 = np.zeros(N_HIDDEN)
    error_x, error_y = np.zeros(N_HIDDEN), np.zeros(N_HIDDEN)
    anchor = np.arctanh(1.0 / np.sqrt(3.0))
    epsilon = 0.1
    second_derivative = -4.0 / (3.0 * np.sqrt(3.0))
    gain = range_gain * DIST_SCALE / SPEED_SCALE
    signs = ((1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1))
    for bearing, start, error in ((1, 0, error_x), (0, 4, error_y)):
        for j, (sd, sb, coefficient) in enumerate(signs):
            unit = start + j
            w1[2, unit] = epsilon * sd
            w1[bearing, unit] = epsilon * sb
            b1[unit] = anchor
            error[unit] = gain * coefficient / (4 * epsilon**2 * second_derivative)
    # Small tanh arguments approximate identity without another hidden layer.
    for source, unit, error, coefficient in (
        (3, 8, error_x, -1.0), (4, 9, error_y, -1.0),
        (1, 10, error_x, arrival_speed / SPEED_SCALE),
        (0, 11, error_y, arrival_speed / SPEED_SCALE),
    ):
        w1[source, unit] = 0.05
        error[unit] = coefficient / 0.05
    return w1, b1, error_x, error_y


def sector_brain(arrival_speed: float, range_gain: float, deadband: float, aim_tolerance: float,
                 min_error: float) -> Brains:
    """The autopilot's control law, compiled into the ordinary 6 -> 16 -> 6 net.

    Hidden units compute the velocity error (ex, ey). Each action's logit is then linear
    in that error, so argmax picks the action from a fan of angular sectors: the normal fan
    of a convex polygon whose vertices are the actions. Its edges are perpendicular to the
    sector boundaries at +-deadband, +-aim_tolerance and 180 degrees, so the autopilot's
    turn and thrust cones are drawn exactly, with no hidden units spent on them. Coasting
    straight has only a bias, so it wins when the error is smaller than min_error.
    """
    w1, b1, error_x, error_y = _velocity_error_units(arrival_speed, range_gain)
    straight = np.array([0.0, 0.0])
    right_burn = straight + np.array([-np.sin(deadband), np.cos(deadband)])
    right_coast = right_burn + np.array([-np.sin(aim_tolerance), np.cos(aim_tolerance)])
    flip = np.array([1.0, -1.0])
    corners = {1: flip * right_burn, 3: straight, 5: right_burn, 0: flip * right_coast, 4: right_coast}
    center = np.mean(list(corners.values()), axis=0)
    w2 = np.zeros((N_HIDDEN, N_ACTIONS))
    b2 = np.zeros(N_ACTIONS)
    for action, corner in corners.items():
        x, y = corner - center
        w2[:, action] = x * error_x + y * error_y
    b2[2] = (straight - center)[0] * min_error / SPEED_SCALE
    return Brains(np.concatenate([w1.ravel(), b1, w2.ravel(), b2])[None, :])


def braking_brain() -> Brains:
    """Approach under thrust, coast while rotating, then burn facing backward.

    The soft switch is distance + 154 - 2*closing_speed: an affine approximation
    to turning travel plus stopping distance over normal approach speeds.
    The steering speed 77 + 0.5*distance crosses the measured closing speed at
    the same boundary. Four final units combine phase and nose alignment into
    a thrust decision; all sixteen units have algebraically constructed weights.
    """
    brain = long_arc_brain(arrival_speed=77.0, range_gain=0.5)
    w1, b1, w2, b2 = brain.layers
    # A pair of half-plane tests implements OR: burn facing the target outside
    # braking range, or facing away inside it. A second pair requires alignment
    # with either direction; ANDing the pairs cuts thrust during the rotation.
    for unit, sign in ((12, 1), (13, -1)):
        w1[0, 1, unit] = 20.0 * sign
        w1[0, 2, unit] = 20.0 * sign * DIST_SCALE / 600.0
        w1[0, 5, unit] = -20.0 * sign * 2.0 * SPEED_SCALE / 600.0
        b1[0, unit] = 20.0 * (sign * 154.0 / 600.0 - 0.9)
    for unit, sign in ((14, 1), (15, -1)):
        w1[0, 1, unit] = 12.0 * sign
        b1[0, unit] = -12.0 * 0.65
    for action in (1, 3, 5):
        w2[0, 12:16, action] = 1.0
        # The existing +1 thrust bias is also the soft logical AND's offset.
    return brain


def calculated_brain(*, turnaround: bool = True, full_burn: bool | None = None,
                     long_arcs: bool = True) -> Brains:
    """Compile burn/coast/brake approaches, or an earlier pursuit controller.

    The default (full_burn=None) selects range-triggered braking. Explicit True
    selects continuous thrust; False selects the earlier five-unit coast pilot.

    With full_burn and long_arcs enabled, use the distance-aware twelve-unit
    controller. The description below applies to the earlier controllers,
    selected with long_arcs=False or full_burn=False.

    In ship coordinates, desired velocity is v*(cos(bearing), sin(bearing)).
    Subtract measured velocity to get forward/sideways errors ex and ey.
    Steering uses a lower target speed than thrust. Once the ship exceeds the
    steering target, steering tends to face back against its incoming velocity
    while inertia carries it into the pickup. No position or next-target data
    is used: this is a statistical inward bias, not a center-facing guarantee.
    Thrust requires ex > 0 and |ey| < 0.35*ex, using its higher target speed.
    Tanh units soften these three half-plane tests into an approximate AND.
    Turn and thrust scores add independently, matching the six action indices.

    The manually checked design uses seven pickup radii/s for steering and
    twelve for thrust (182 and 312 px/s). These are control setpoints, not
    promised travel speeds or optimized values. With turnaround=False, both
    use eight radii/s (208 px/s), preserving the previous faster pursuit mode.
    Distance and closing speed are unused. Full burn uses only the two steering
    units; each thrust action scores exactly one above its coast counterpart,
    so coasting cannot win argmax. Set full_burn=False for the five-unit pilot
    that can coast. Neither variant changes the simulation's action rules.
    """
    if full_burn is None:
        if long_arcs and turnaround:
            return braking_brain()
        full_burn = True
    if full_burn and long_arcs:
        return long_arc_brain()
    steer_speed = (7.0 if turnaround else 8.0) * PICKUP_R
    burn_speed = (12.0 if turnaround else 8.0) * PICKUP_R

    def errors(speed: float) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([0, speed, 0, -SPEED_SCALE, 0, 0], dtype=float),
                np.array([speed, 0, 0, 0, -SPEED_SCALE, 0], dtype=float))

    steer_x, steer_y = errors(steer_speed)
    ex, ey = errors(burn_speed)
    w1 = np.zeros((N_INPUTS, N_HIDDEN))
    b1 = np.zeros(N_HIDDEN)
    w2 = np.zeros((N_HIDDEN, N_ACTIONS))
    b2 = np.zeros(N_ACTIONS)

    w1[:, 0] = steer_y / 30.0
    w1[:, 1] = steer_x / 30.0
    if not full_burn:
        w1[:, 2] = ex / 15.0
        w1[:, 3] = (0.35 * ex - ey) / 15.0
        w1[:, 4] = (0.35 * ex + ey) / 15.0
    for action in range(N_ACTIONS):
        turn, thrust = action // 2 - 1, action % 2
        if turn:
            w2[0, action] = turn
        else:
            # A deadband when facing the error, but keep turning if it is aft.
            w2[1, action] = 0.2
            b2[action] = 0.1
        if thrust:
            if full_burn:
                b2[action] += 1.0
            else:
                w2[2:5, action] = 1.0
                b2[action] -= 2.4
    genome = np.concatenate([w1.ravel(), b1, w2.ravel(), b2])
    return Brains(genome[None, :])


# (arrival speed, range gain, deadband, aim tolerance, min error), from `just cc-sector-search`.
SECTOR_DESIGN = (60.0, 0.8, 0.15, 0.9, 5.0)

FLYTHROUGH_SPEED = 300.0
FLYTHROUGH_TURN_RANGE = PICKUP_R + FLYTHROUGH_SPEED * np.pi / TURN_RATE
FLYTHROUGH_ALIGNMENT_ANGLE = np.deg2rad(15.0)


def flythrough_brain(cruise_speed: float = FLYTHROUGH_SPEED, *, orient_first: bool = False) -> Brains:
    """Accelerate toward cruise speed, coast, then turn near an outer range.

    Original 6/16/6 network. The nominal range is 262 px at 300,
    allowing half a turn at cruise before pickup. An affine approach guard
    delays the turn by one pixel per px/s below cruise, and releases it on
    receding approaches. This prevents an endless spin after a missed pickup.
    No phase state or runtime controller is added. Next-target acquisition
    naturally permits braking thrust again. With orient_first enabled, ten
    units are active: thrust additionally requires bearing within 15 degrees,
    and bearing correction takes priority outside that cone. This limits the
    ability to correct sideways momentum; orient_first=False restores the
    preceding seven-unit controller.
    """
    if not np.isfinite(cruise_speed) or cruise_speed <= 0:
        raise ValueError("Cruise speed must be positive and finite")
    turn_range = PICKUP_R + cruise_speed * np.pi / TURN_RATE
    brain = calculated_brain(full_burn=False)
    w1, b1, w2, b2 = brain.layers
    # Align velocity accurately before entering the coasting leg. A small
    # 10 px/s headroom keeps thrust available until the cruise cutoff.
    reference = cruise_speed + 10.0
    w1[0, :2, :2] *= reference / (7 * PICKUP_R)
    w1[0, :, :2] *= 3.0
    w1[0, :2, 2:5] *= reference / (12 * PICKUP_R)
    w1[0, :, 2:5] *= 5.0
    w1[0, 5, 5] = SPEED_SCALE / 5.0
    b1[0, 5] = -cruise_speed / 5.0
    for action in (1, 3, 5):
        w2[0, 5, action] = -0.6
        b2[0, action] -= 0.6
    # Soft approach/range gate. At cruise it crosses zero at the fixed range.
    w1[0, 2, 6] = DIST_SCALE / 5.0
    w1[0, 5, 6] = -SPEED_SCALE / 5.0
    b1[0, 6] = (cruise_speed - turn_range) / 5.0
    for action in range(N_ACTIONS):
        if action // 2 == 2:
            w2[0, 6, action] -= 2.0
            b2[0, action] += 2.0
        if action % 2:
            w2[0, 6, action] += 2.0
            b2[0, action] -= 2.0
    if orient_first:
        # Permit a broader velocity-error cone once the nose faces the target;
        # the independent bearing gate below still strictly bounds thrust.
        w1[0, :, 3:5] += (np.tan(np.pi / 3) - 0.35) * w1[0, :, 2, None]
        w1[0, 1, 7] = 1000.0
        b1[0, 7] = -1000.0 * np.cos(FLYTHROUGH_ALIGNMENT_ANGLE)
        for action in (1, 3, 5):
            w2[0, 7, action] = 0.6
            b2[0, action] -= 0.6
        # Prioritize turning toward the bearing when outside the alignment cone.
        for unit, sign, actions in ((8, 1, (4, 5)), (9, -1, (0, 1))):
            w1[0, 0, unit] = 50.0 * sign
            b1[0, unit] = -50.0 * np.sin(FLYTHROUGH_ALIGNMENT_ANGLE)
            for action in actions:
                w2[0, unit, action] = 1.5
                b2[0, action] += 1.5
    return brain


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pursuit", action="store_true",
                        help="use the 208 px/s steering target without turnaround bias")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--allow-coast", action="store_true",
                        help="restore the previous pilot's alignment-based thrust gate")
    modes.add_argument("--full-burn", action="store_true",
                       help="restore continuous thrust instead of range-triggered braking")
    modes.add_argument("--fly-through", action="store_true",
                       help="accelerate toward cruise speed, coast, then turn near the pickup")
    modes.add_argument("--sector", action="store_true",
                       help="the autopilot's control law, compiled as angular action sectors")
    parser.add_argument("--cruise-speed", type=float, default=FLYTHROUGH_SPEED,
                        help="fly-through cruise target in px/s (default: 300)")
    parser.add_argument("--orient-first", action="store_true",
                        help="experimental strict 15-degree crystal-alignment gate")
    parser.add_argument("--tight-curls", action="store_true",
                        help="restore the earlier constant-speed full-burn steering")
    args = parser.parse_args()
    path = Path("weights.npz")
    full_burn = (False if args.allow_coast else
                 True if args.full_burn or args.tight_curls or args.pursuit else None)
    brain = (sector_brain(*SECTOR_DESIGN) if args.sector else
             flythrough_brain(args.cruise_speed, orient_first=args.orient_first)
             if args.fly_through else
             calculated_brain(turnaround=not args.pursuit, full_burn=full_burn,
                              long_arcs=not (args.tight_curls or args.pursuit)))
    brain.save(path)
    print(f"Wrote {path}: calculated weights, no training or random sampling.")


if __name__ == "__main__":
    main()

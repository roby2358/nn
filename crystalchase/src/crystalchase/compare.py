"""Fly the autopilot and saved pilots over the same N courses and show where the time goes:
`just cc-compare N file.npz ...`. Score tells you who won; these rows tell you why."""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DT
from .course import make_course
from .flock import Flock, decode, solo_rule
from .pilots import AutoPilot, Pilot, load_pilot

STRAIGHT_ON = 0.5      # cos of the velocity-to-next-crystal angle: above this, under 60 degrees
DOUBLING_BACK = -0.5   # below this, over 120 degrees


def mean_of(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


@dataclass(frozen=True)
class Report:
    name: str
    crashes: int
    crystals: float
    survivor_crystals: float
    score: float
    hop_p50: float
    hop_p90: float
    hop_straight_on: float
    hop_doubling_back: float
    speed: float
    pickup_speed: float
    nose_at_pickup: float   # mean cos(nose, velocity) at pickup; -1 means flying tail-first
    thrusting: float
    turning: float
    receding: float         # share of flight time spent moving away from the crystal
    path_ratio: float       # distance flown per straight-line pixel, completed hops only
    crash_speed: float
    crash_after: float      # seconds from the last pickup to the crash


def fly_report(name: str, pilot: Pilot, courses: list) -> Report:
    n = len(courses)
    flock = Flock(courses, np.arange(n), solo_rule)
    hop_start, leg_flown = np.zeros(n), np.zeros(n)
    entry_cos = np.full(n, math.nan)          # velocity vs next crystal at the start of each hop
    hops, straight_on, doubling_back, pickup_speeds, nose_dots = [], [], [], [], []
    crash_speeds, crash_after = [], []
    speeds, thrusts, turns, recedes = [], [], [], []
    total_flown, total_straight = 0.0, 0.0
    while not flock.done:
        obs = flock.observe()
        actions = pilot.act(obs)
        turn, thrust = decode(actions)
        live, leg, before = flock.alive.copy(), flock.start_dist.copy(), np.hypot(*flock.vel.T)
        flock.step(actions)
        speed = np.hypot(*flock.vel.T)
        leg_flown[live] += speed[live] * DT
        speeds.append(speed[live])
        thrusts.append(thrust[live])
        turns.append(turn[live] != 0)
        recedes.append(obs[live, 5] < 0)
        for e in flock.events:
            k = e.ship
            if e.kind == "crash":
                crash_speeds.append(float(before[k]))
                crash_after.append(flock.seconds - hop_start[k])
                continue
            hop = flock.seconds - hop_start[k]
            hops.append(hop)
            if entry_cos[k] > STRAIGHT_ON:
                straight_on.append(hop)
            if entry_cos[k] < DOUBLING_BACK:
                doubling_back.append(hop)
            hop_start[k] = flock.seconds
            total_flown += leg_flown[k]
            total_straight += leg[k]
            leg_flown[k] = 0.0
            nose = np.array([np.cos(flock.ang[k]), np.sin(flock.ang[k])])
            to_next = flock.targets()[k] - flock.pos[k]
            pickup_speeds.append(float(speed[k]))
            nose_dots.append(float(nose @ flock.vel[k]) / max(float(speed[k]), 1e-9))
            entry_cos[k] = float(flock.vel[k] @ to_next) / max(float(speed[k] * np.hypot(*to_next)), 1e-9)
    return Report(name, int((~flock.alive).sum()), float(flock.crystals.mean()),
                  float(flock.crystals[flock.alive].mean()), float(flock.fitness().mean()),
                  float(np.median(hops)), float(np.percentile(hops, 90)), mean_of(straight_on),
                  mean_of(doubling_back), float(np.concatenate(speeds).mean()), mean_of(pickup_speeds),
                  mean_of(nose_dots), float(np.concatenate(thrusts).mean()), float(np.concatenate(turns).mean()),
                  float(np.concatenate(recedes).mean()), total_flown / total_straight,
                  mean_of(crash_speeds), mean_of(crash_after))


ROWS = [("crashes", "crashes", "{:.0f}"), ("crystals / 30 s", "crystals", "{:.2f}"),
        ("  survivors only", "survivor_crystals", "{:.2f}"), ("score", "score", "{:.2f}"),
        ("hop median (s)", "hop_p50", "{:.2f}"), ("hop p90 (s)", "hop_p90", "{:.2f}"),
        ("hop, next <60 deg", "hop_straight_on", "{:.2f}"), ("hop, next >120 deg", "hop_doubling_back", "{:.2f}"),
        ("mean speed", "speed", "{:.0f}"), ("pickup speed", "pickup_speed", "{:.0f}"),
        ("nose.vel at pickup", "nose_at_pickup", "{:+.2f}"), ("thrusting", "thrusting", "{:.0%}"),
        ("turning", "turning", "{:.0%}"), ("moving away", "receding", "{:.0%}"),
        ("path / straight", "path_ratio", "{:.2f}"), ("crash speed", "crash_speed", "{:.0f}"),
        ("crash after pickup s", "crash_after", "{:.2f}")]


def main() -> None:
    n = int(sys.argv[1])
    courses = [make_course(seed) for seed in range(n)]
    entries: list[tuple[str, Pilot]] = [("autopilot", AutoPilot())]
    entries += [(Path(p).stem, load_pilot(Path(p))) for p in sys.argv[2:]]
    reports = [fly_report(name, pilot, courses) for name, pilot in entries]
    width = max(len(r.name) for r in reports) + 2
    print(f"{n} courses\n{'':22}" + "".join(f"{r.name:>{width}}" for r in reports))
    for label, attr, fmt in ROWS:
        print(f"{label:22}" + "".join(f"{fmt.format(getattr(r, attr)):>{width}}" for r in reports))


if __name__ == "__main__":
    main()

import numpy as np

from crystalchase.calculated import FLYTHROUGH_ALIGNMENT_ANGLE, flythrough_brain
from crystalchase.config import BOT_ARRIVE_SPEED, PICKUP_R, TURN_RATE
from crystalchase.course import Course, make_course
from crystalchase.flock import Flock, solo_rule
from crystalchase.pilots import Brains, GENOME


def test_flythrough_original_architecture_and_roundtrip(tmp_path):
    brain = flythrough_brain()
    assert GENOME == 214
    assert brain.genomes.shape == (1, 214)
    assert brain.layers[0].shape == (1, 6, 16)
    assert brain.layers[2].shape == (1, 16, 6)
    assert np.count_nonzero(np.any(brain.layers[0][0] != 0, axis=0)) == 7
    assert np.array_equal(brain.genomes, flythrough_brain().genomes)
    path = tmp_path / "weights.npz"
    brain.save(path)
    assert np.array_equal(Brains.load(path).genomes, brain.genomes)
    assert BOT_ARRIVE_SPEED == 140.0


def test_flythrough_accelerates_then_coasts_and_retargets():
    brain = flythrough_brain(200, orient_first=False)
    for cos, forward, closing, thrust in (
        (1, 0, 0, 1),                  # accelerate toward the crystal
        (1, 204, 204, 0),              # cruise: cut thrust
        (-1, -204, 204, 0),            # nose reversed: keep coasting
        (1, -204, -204, 1),            # next target: burn to redirect
    ):
        obs = np.array([[0, cos, 1, forward / 300, 0, closing / 300]])
        assert brain.act(obs)[0] % 2 == thrust


def test_centered_approach_coasts_then_turns_at_range_without_braking():
    course = Course(0, np.array([100., 360.]), 0., np.tile([600., 360.], (80, 1)))
    flock = Flock([course], np.array([0]), solo_rule)
    brain = flythrough_brain(200, orient_first=False)
    turn_range = PICKUP_R + 200 * np.pi / TURN_RATE
    cruising = False
    coast_before_range = False
    turned_inside_range = False
    while not flock.done and flock.crystals[0] == 0:
        action = brain.act(flock.observe())
        if cruising:
            assert action[0] % 2 == 0
            distance = flock.observe()[0, 2] * 400
            if distance > turn_range + 25:
                assert action[0] // 2 == 1
                coast_before_range = True
            if distance < turn_range:
                turned_inside_range |= action[0] // 2 != 1
        flock.step(action)
        cruising |= np.linalg.norm(flock.vel[0]) >= 198
    assert flock.crystals[0] == 1
    assert coast_before_range and turned_inside_range
    assert abs(np.linalg.norm(flock.vel[0]) - 200) < 5
    nose = np.array([np.cos(flock.ang[0]), np.sin(flock.ang[0])])
    assert nose @ flock.vel[0] < 0


def test_flythrough_seeded_courses():
    n = 100
    flock = Flock([make_course(seed) for seed in range(n)], np.arange(n), solo_rule)
    brain = Brains(np.repeat(flythrough_brain(200, orient_first=False).genomes, n, axis=0))
    backward, pickup_speeds = [], []
    while not flock.done:
        flock.step(brain.act(flock.observe()))
        for event in flock.events:
            if event.kind == "pickup":
                k = event.ship
                nose = np.array([np.cos(flock.ang[k]), np.sin(flock.ang[k])])
                backward.append(nose @ flock.vel[k] < 0)
                pickup_speeds.append(np.linalg.norm(flock.vel[k]))
    assert flock.alive.all()
    assert flock.crystals.mean() >= 8.5
    assert np.mean(backward) > 0.95
    assert np.mean(pickup_speeds) > 190
    assert 195 < np.median(pickup_speeds) < 205


def test_400_cruise_on_a_long_clear_approach():
    # Long enough to accelerate and turn, still inside the physical crash walls.
    course = Course(0, np.array([-100., 360.]), 0., np.tile([700., 360.], (80, 1)))
    flock = Flock([course], np.array([0]), solo_rule)
    brain = flythrough_brain(400, orient_first=False)
    while not flock.done and flock.crystals[0] == 0:
        flock.step(brain.act(flock.observe()))
    assert flock.alive[0] and flock.crystals[0] == 1
    assert abs(np.linalg.norm(flock.vel[0]) - 400) < 5


def test_orient_first_turns_toward_the_crystal_without_thrust():
    brain = flythrough_brain(orient_first=True)
    for bearing, turn in ((np.pi / 2, 1), (-np.pi / 2, -1)):
        obs = np.array([[np.sin(bearing), np.cos(bearing), 1.5, 0, 0, 0]])
        action = brain.act(obs)[0]
        assert action % 2 == 0
        assert action // 2 - 1 == turn
    assert brain.act(np.array([[0., 1., 1.5, 0., 0., 0.]]))[0] == 3


def test_orient_first_never_burns_outside_the_bearing_cone():
    rng = np.random.default_rng(902)
    n = 10000
    bearing = rng.uniform(-np.pi, np.pi, n)
    obs = np.column_stack((np.sin(bearing), np.cos(bearing),
                           rng.uniform(0, 2.5, n), rng.uniform(-2, 2, (n, 3))))
    actions = flythrough_brain(orient_first=True).act(obs)
    assert not np.any((actions % 2 == 1) &
                      (np.abs(bearing) >= FLYTHROUGH_ALIGNMENT_ANGLE))

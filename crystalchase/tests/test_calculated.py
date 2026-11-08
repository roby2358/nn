import numpy as np

from crystalchase.calculated import calculated_brain
from crystalchase.config import DT, FIELD_H, FIELD_W, TURN_RATE
from crystalchase.course import make_course
from crystalchase.evolve import fly
from crystalchase.flock import Flock, solo_rule
from crystalchase.pilots import Brains, GENOME


def test_calculated_pilot_controls_and_roundtrip(tmp_path):
    brain = calculated_brain(full_burn=False)
    assert brain.genomes.shape == (1, GENOME)
    assert np.array_equal(brain.genomes, calculated_brain(full_burn=False).genomes)
    path = tmp_path / "weights.npz"
    brain.save(path)
    loaded = Brains.load(path)
    assert np.array_equal(loaded.genomes, brain.genomes)
    # Target ahead: accelerate, initiate the turn above the steering setpoint,
    # and coast while turning at the higher thrust setpoint.
    for vf, expected in [(0.0, 3), (208.0 / 300.0, 1), (312.0 / 300.0, 0)]:
        obs = np.array([[0.0, 1.0, 1.0, vf, 0.0, vf]])
        assert loaded.act(obs)[0] == expected
    for bearing, expected in [(1.0, 4), (-1.0, 0)]:
        obs = np.array([[bearing, 0.0, 1.0, 0.0, 0.0, 0.0]])
        assert loaded.act(obs)[0] == expected


def test_calculated_pilot_flies_seeded_courses():
    courses = [make_course(seed) for seed in range(20)]
    brain = calculated_brain(full_burn=False)
    pilots = Brains(np.repeat(brain.genomes, len(courses), axis=0))
    flock = fly(Flock(courses, np.arange(len(courses)), solo_rule), pilots)
    assert flock.alive.all()
    assert (flock.crystals >= 5).all()


def test_turnaround_improves_inward_pickup_headings():
    # Measure the actual field-center direction only in the test; the pilots
    # still receive the original six observations, with no center information.
    courses = [make_course(seed) for seed in range(100)]
    fractions = []
    for turnaround in (False, True):
        brain = calculated_brain(turnaround=turnaround, full_burn=False)
        pilots = Brains(np.repeat(brain.genomes, len(courses), axis=0))
        flock = Flock(courses, np.arange(len(courses)), solo_rule)
        inward = []
        while not flock.done:
            flock.step(pilots.act(flock.observe()))
            for event in flock.events:
                if event.kind == "pickup":
                    k = event.ship
                    nose = np.array([np.cos(flock.ang[k]), np.sin(flock.ang[k])])
                    to_center = np.array([FIELD_W / 2, FIELD_H / 2]) - flock.pos[k]
                    inward.append(nose @ to_center > 0)
        assert flock.alive.all()
        assert flock.crystals.mean() > 7
        fractions.append(np.mean(inward))
    assert fractions[1] > 0.8
    assert fractions[1] > fractions[0] + 0.1


def test_full_burn_coast_actions_are_strictly_dominated(tmp_path):
    brain = calculated_brain(full_burn=True)
    path = tmp_path / "weights.npz"
    brain.save(path)
    loaded = Brains.load(path)
    assert np.array_equal(brain.genomes, loaded.genomes)
    assert np.array_equal(brain.genomes, calculated_brain(full_burn=True).genomes)
    w1, _, w2, b2 = loaded.layers
    assert np.count_nonzero(np.any(w1[0] != 0, axis=0)) == 12
    # Identical output columns, with a strict positive bias difference, prove
    # every coast action loses to its own burning counterpart for any finite h.
    assert np.array_equal(w2[:, :, 0::2], w2[:, :, 1::2])
    assert np.allclose(b2[:, 1::2] - b2[:, 0::2], 1.0)


def test_full_burn_collects_without_ever_coasting():
    n = 100
    flock = Flock([make_course(seed) for seed in range(n)], np.arange(n), solo_rule)
    brain = calculated_brain(full_burn=True)
    pilots = Brains(np.repeat(brain.genomes, n, axis=0))
    while not flock.done:
        actions = pilots.act(flock.observe())
        assert (actions % 2 == 1).all()
        flock.step(actions)
        assert flock.thrusting[flock.alive].all()
    assert flock.crystals.mean() >= 6
    assert (~flock.alive).sum() <= 10


def test_long_arcs_make_more_progress_with_less_turning():
    results = []
    n = 100
    for long_arcs in (False, True):
        flock = Flock([make_course(seed) for seed in range(n)], np.arange(n), solo_rule)
        brain = calculated_brain(full_burn=True, long_arcs=long_arcs)
        pilots = Brains(np.repeat(brain.genomes, n, axis=0))
        distance, turned, frames = 0.0, 0, 0
        while not flock.done:
            actions = pilots.act(flock.observe())
            turned += np.count_nonzero((actions // 2 != 1) & flock.alive)
            frames += flock.alive.sum()
            flock.step(actions)
            distance += np.linalg.norm(flock.vel[flock.alive], axis=1).sum() * DT
        # Distance per equivalent full turn measures stretched-out arcs;
        # pickup count ensures those arcs still progress toward the crystals.
        results.append((distance / (turned * DT * TURN_RATE / (2 * np.pi)),
                        turned / frames, flock.crystals.mean(), (~flock.alive).sum()))
    old, new = results
    assert new[0] > old[0] * 1.1
    assert new[1] < old[1]
    assert new[2] > old[2] + 1
    assert new[3] <= old[3]


def test_long_arcs_turn_back_earlier_near_the_target():
    brain = calculated_brain(full_burn=True)
    # Same pose and velocity, different distance: continue a long approach,
    # but turn to brake on the shorter one. Both actions still include thrust.
    near = np.array([[0.0, 1.0, 100.0 / 400.0, 230.0 / 300.0, 0.0, 230.0 / 300.0]])
    far = near.copy()
    far[0, 2] = 600.0 / 400.0
    assert brain.act(near)[0] in (1, 5)
    assert brain.act(far)[0] == 3


def test_braking_pilot_roundtrip_and_approach_phases(tmp_path):
    brain = calculated_brain()
    path = tmp_path / "weights.npz"
    brain.save(path)
    brain = Brains.load(path)
    assert np.array_equal(brain.genomes, calculated_brain().genomes)
    assert np.count_nonzero(np.any(brain.layers[0][0] != 0, axis=0)) == 16
    # Far ahead: straight full burn. Close and fast: coast to rotate.
    # Same closing motion with the nose reversed: straight braking thrust.
    for cos, distance, forward, closing, expected in (
        (1, 600, 200, 200, 3), (1, 200, 250, 250, 0),
        (-1, 200, -250, 250, 3),
    ):
        observation = np.array([[0, cos, distance / 400, forward / 300, 0, closing / 300]])
        assert brain.act(observation)[0] == expected


def test_braking_pilot_coasts_during_turns_and_arrives_backward():
    n = 100
    flock = Flock([make_course(seed) for seed in range(n)], np.arange(n), solo_rule)
    pilots = Brains(np.repeat(calculated_brain().genomes, n, axis=0))
    rotating = coasting_turns = 0
    backward, pickup_speeds = [], []
    while not flock.done:
        actions = pilots.act(flock.observe())
        turning = (actions // 2 != 1) & flock.alive
        rotating += np.count_nonzero(turning)
        coasting_turns += np.count_nonzero(turning & (actions % 2 == 0))
        flock.step(actions)
        for event in flock.events:
            if event.kind == "pickup":
                k = event.ship
                nose = np.array([np.cos(flock.ang[k]), np.sin(flock.ang[k])])
                backward.append(nose @ flock.vel[k] < 0)
                pickup_speeds.append(np.linalg.norm(flock.vel[k]))
    assert flock.alive.all()
    assert flock.crystals.mean() >= 9
    assert coasting_turns / rotating > 0.6
    assert np.mean(backward) > 0.7
    assert np.mean(pickup_speeds) < 110

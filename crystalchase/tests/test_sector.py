import numpy as np

from crystalchase.calculated import SECTOR_DESIGN, sector_brain
from crystalchase.config import DIST_SCALE, SPEED_SCALE
from crystalchase.course import make_course
from crystalchase.evolve import Evolver, fly, population_flight
from crystalchase.flock import encode_action
from crystalchase.pilots import GENOME, Brains


def test_sector_net_is_the_ordinary_architecture_and_round_trips(tmp_path):
    brain = sector_brain(*SECTOR_DESIGN)
    assert brain.genomes.shape == (1, GENOME)
    brain.save(tmp_path / "weights.npz")
    assert np.array_equal(Brains.load(tmp_path / "weights.npz").genomes, brain.genomes)


def test_argmax_draws_the_control_law_sectors():
    arrival, gain, deadband, tolerance, _ = SECTOR_DESIGN
    rng = np.random.default_rng(3)
    n = 40000
    bearing = rng.uniform(-np.pi, np.pi, n)
    sinb, cosb = np.sin(bearing), np.cos(bearing)
    dist, vf, vs = rng.uniform(0.05, 2.0, n), rng.uniform(-1.5, 1.5, n), rng.uniform(-1.5, 1.5, n)
    obs = np.column_stack([sinb, cosb, dist, vf, vs, vf * cosb + vs * sinb])
    want = arrival + gain * dist * DIST_SCALE
    ex, ey = cosb * want - vf * SPEED_SCALE, sinb * want - vs * SPEED_SCALE
    aim = np.arctan2(ey, ex)
    turn = np.where(np.abs(aim) < deadband, 0, np.sign(aim)).astype(np.int64)
    expected = encode_action(turn, (np.abs(aim) < tolerance).astype(np.int64))
    margin = 0.03
    clear = ((np.hypot(ex, ey) > 60.0) & (np.abs(np.abs(aim) - deadband) > margin)
             & (np.abs(np.abs(aim) - tolerance) > margin) & (np.abs(aim) < np.pi - margin))
    assert clear.sum() > n / 2
    # The sector edges are exact; the tanh units only approximate the error fed into them.
    assert np.mean(sector_brain(*SECTOR_DESIGN).act(obs)[clear] == expected[clear]) > 0.999


def test_sector_net_loaded_over_a_random_swarm_stays_the_leader_through_evolution(tmp_path):
    swarm, weights = tmp_path / "swarm.npz", tmp_path / "weights.npz"
    Evolver(1).save(swarm, weights)
    brain = sector_brain(*SECTOR_DESIGN)
    brain.save(weights)
    evolver = Evolver(2)
    evolver.load(swarm, weights)
    for _ in range(3):
        evolver.step()
    assert np.array_equal(evolver.champion().genomes, brain.genomes)


def test_sector_net_outscores_the_autopilot_without_crashing():
    courses = [make_course(seed) for seed in range(80)]
    flock = fly(*population_flight(sector_brain(*SECTOR_DESIGN).genomes, courses))
    fit, c = flock.fitness(), len(courses)
    assert flock.alive.all()
    assert fit[:c].mean() > fit[c:].mean()

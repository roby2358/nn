import numpy as np

from crystalchase.config import ELITES, POP_SIZE
from crystalchase.course import make_course
from crystalchase.evolve import Evolver, evaluate, fly, population_flight, reproduce
from crystalchase.flock import Flock, solo_rule
from crystalchase.pilots import GENOME, AutoPilot, Brains, random_genomes


def test_autopilot_collects_crystals_without_crashing():
    f = Flock([make_course(3)], np.zeros(1, dtype=np.int64), solo_rule)
    while not f.done:
        f.step(AutoPilot().act(f.observe()))
    assert f.alive[0] and f.crystals[0] >= 10


def test_replaying_one_course_matches_the_evaluation():
    """The GUI replays course 0 alone; it must reproduce training's scores exactly."""
    genomes = random_genomes(np.random.default_rng(0), 12)
    courses = [make_course(11), make_course(12)]
    full = fly(*population_flight(genomes, courses))
    alone = fly(*population_flight(genomes, courses[:1]))
    assert np.array_equal(full.fitness()[:12], alone.fitness()[:12])
    assert full.fitness()[24] == alone.fitness()[12]  # the bot on course 0


def test_evaluate_reports_the_bot_as_a_yardstick():
    fit, stats = evaluate(random_genomes(np.random.default_rng(1), 8), [make_course(5)])
    assert fit.shape == (8,) and stats.best == fit.max() and stats.bot > 5


def test_reproduce_keeps_elites_unchanged_and_best_first():
    rng = np.random.default_rng(2)
    g = random_genomes(rng, POP_SIZE)
    fit = rng.random(POP_SIZE)
    nxt = reproduce(rng, g, fit)
    order = np.argsort(-fit)
    assert nxt.shape == g.shape
    assert np.array_equal(nxt[:ELITES], g[order[:ELITES]])


def saved(tmp_path, seed):
    ev = Evolver(seed)
    ev.step()
    ev.save(tmp_path / "swarm.npz", tmp_path / "weights.npz")
    return ev


def test_save_and_load_roundtrip(tmp_path):
    ev = saved(tmp_path, 4)
    other = Evolver(5)
    report = other.load(tmp_path / "swarm.npz", tmp_path / "weights.npz")
    assert report.swarm_found and report.leader_found and report.gen == 1
    assert np.array_equal(other.genomes, ev.genomes) and other.history == ev.history
    assert Brains.load(tmp_path / "weights.npz").genomes.shape == (1, GENOME)


def test_weights_overwrite_the_swarm_leader(tmp_path):
    saved(tmp_path, 4)
    swarm = Evolver(6).genomes  # a different population's file supplies the swarm
    np.savez(tmp_path / "swarm.npz", population=swarm, history=np.zeros((0, 6)))
    leader = Brains.load(tmp_path / "weights.npz").genomes[0]
    ev = Evolver(7)
    ev.load(tmp_path / "swarm.npz", tmp_path / "weights.npz")
    assert np.array_equal(ev.genomes[0], leader) and np.array_equal(ev.genomes[1:], swarm[1:])


def test_missing_swarm_means_random_swarm_under_the_saved_leader(tmp_path):
    saved(tmp_path, 4)
    (tmp_path / "swarm.npz").unlink()
    ev = Evolver(8)
    report = ev.load(tmp_path / "swarm.npz", tmp_path / "weights.npz")
    assert not report.swarm_found and report.leader_found and ev.history == []
    assert ev.genomes.shape == (POP_SIZE, GENOME)
    assert np.array_equal(ev.genomes[0], Brains.load(tmp_path / "weights.npz").genomes[0])


def test_missing_weights_keeps_the_swarm_leader(tmp_path):
    original = saved(tmp_path, 4)
    (tmp_path / "weights.npz").unlink()
    ev = Evolver(9)
    report = ev.load(tmp_path / "swarm.npz", tmp_path / "weights.npz")
    assert report.swarm_found and not report.leader_found
    assert np.array_equal(ev.genomes, original.genomes)

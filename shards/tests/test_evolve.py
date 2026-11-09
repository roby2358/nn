"""Nets, selection and persistence."""

import sys

import numpy as np

from shards import evolve
from shards.config import CMD_FIRE, CMD_THRUST, CMD_TURN, ELITES, N_INPUTS, POP_SIZE
from shards.evolve import Evolver, evaluate, reproduce
from shards.pilots import GENOME, Brains, random_genomes
from shards.scenario import make_scenario


def test_outputs_stay_in_their_ranges():
    rng = np.random.default_rng(0)
    out = Brains(random_genomes(rng, 50) * 30).act(rng.normal(0.0, 5.0, (50, N_INPUTS)))
    assert out.shape == (50, 3)
    assert (-1.0 <= out[:, CMD_TURN]).all() and (out[:, CMD_TURN] <= 1.0).all()
    squashed = out[:, [CMD_THRUST, CMD_FIRE]]
    assert (0.0 <= squashed).all() and (squashed <= 1.0).all()


def test_headless_run_saves_along_the_way_and_at_the_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)                     # keeps the real swarm.npz and weights.npz out of it
    monkeypatch.setattr(evolve, "SAVE_EVERY", 2)
    monkeypatch.setattr(sys, "argv", ["evolve", "3"])
    evolve.main()
    assert capsys.readouterr().out.count("saved") == 2
    assert (tmp_path / evolve.SWARM).exists() and (tmp_path / evolve.WEIGHTS).exists()


def test_evaluate_is_repeatable_and_fair():
    rng = np.random.default_rng(1)
    genomes = random_genomes(rng, 6)
    scenarios = [make_scenario(3), make_scenario(4)]
    a, _ = evaluate(genomes, scenarios, 1)
    b, _ = evaluate(genomes, scenarios, 1)
    assert np.array_equal(a, b)
    twins, _ = evaluate(np.vstack([genomes[:1], genomes[:1]]), scenarios, 1)
    assert twins[0] == twins[1]


def test_elites_survive_unchanged_and_best_is_first():
    rng = np.random.default_rng(2)
    genomes = random_genomes(rng, POP_SIZE)
    fitness = rng.random(POP_SIZE)
    next_gen = reproduce(rng, genomes, fitness)
    assert next_gen.shape == (POP_SIZE, GENOME)
    assert np.array_equal(next_gen[:ELITES], genomes[np.argsort(-fitness)[:ELITES]])


def test_save_and_load_round_trip(tmp_path):
    swarm, weights = tmp_path / "swarm.npz", tmp_path / "weights.npz"
    a = Evolver(3)
    a.step()
    a.save(swarm, weights)
    b = Evolver(4)
    report = b.load(swarm, weights)
    assert report.swarm_found and report.leader_found and b.gen == 1
    assert np.array_equal(a.genomes, b.genomes)
    assert a.history == b.history

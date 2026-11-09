"""Nets, seating, selection and persistence."""

import numpy as np

from tenks import evolve
from tenks.config import CMD_LEFT, CMD_RIGHT, CMD_SHOOT, ELITES, N_INPUTS, N_OUTPUTS, POP_SIZE, RUN_STEPS, TANKS
from tenks.evolve import Evolver, deal, evaluate, fight, reproduce
from tenks.pilots import GENOME, Brains, load_roster, random_genomes, save_champion, seat
from tenks.scenario import make_scenario
from tenks.worlds import BLUE, GREEN, Worlds


def test_outputs_stay_in_their_ranges():
    rng = np.random.default_rng(0)
    out = Brains(random_genomes(rng, 50) * 30).act(rng.normal(0.0, 5.0, (50, N_INPUTS)))
    assert out.shape == (50, N_OUTPUTS)
    treads = out[:, [CMD_LEFT, CMD_RIGHT]]
    assert (-1.0 <= treads).all() and (treads <= 1.0).all()
    assert (0.0 <= out[:, CMD_SHOOT]).all() and (out[:, CMD_SHOOT] <= 1.0).all()


def test_each_deal_seats_every_net_once():
    seats = deal(np.random.default_rng(1), POP_SIZE, 3)
    assert seats.shape == (3 * POP_SIZE // TANKS, TANKS)
    assert (np.bincount(seats.ravel(), minlength=POP_SIZE) == 3).all()


def test_evaluate_is_repeatable():
    rng = np.random.default_rng(1)
    genomes = random_genomes(rng, POP_SIZE)
    seats = deal(rng, POP_SIZE, 1)
    scenarios = [make_scenario(s) for s in range(len(seats))]
    a, _ = evaluate(genomes, seats, scenarios, 1)
    b, _ = evaluate(genomes, seats, scenarios, 1)
    assert a.shape == (POP_SIZE,)
    assert np.array_equal(a, b)


def test_neither_side_has_an_edge():
    rng = np.random.default_rng(6)
    genomes = random_genomes(rng, POP_SIZE)
    seats = deal(rng, POP_SIZE, 40)
    worlds = fight(Worlds([make_scenario(s) for s in range(len(seats))], RUN_STEPS), Brains(genomes[seats.ravel()]))
    gap = worlds.alive[:, GREEN].sum(1).mean() - worlds.alive[:, BLUE].sum(1).mean()
    assert abs(gap) < 0.5


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
    b.load(swarm, weights)
    assert np.array_equal(a.genomes, b.genomes)
    assert a.history == b.history


def test_a_weights_file_alone_fills_every_seat_with_the_leader(tmp_path):
    weights = tmp_path / "weights.npz"
    champion = random_genomes(np.random.default_rng(8), 1)[0]
    save_champion(weights, champion)
    ev = Evolver(9)
    report = ev.load(tmp_path / "missing.npz", weights)
    assert np.array_equal(ev.genomes, np.tile(champion, (POP_SIZE, 1)))
    assert np.array_equal(ev.leaders(), np.tile(champion, (TANKS, 1)))
    assert (not report.swarm_found, report.leader_found, ev.gen) == (True, True, 0)


def test_a_weights_file_with_a_swarm_takes_only_the_top_spot(tmp_path):
    swarm, weights = tmp_path / "swarm.npz", tmp_path / "weights.npz"
    a = Evolver(3)
    a.step()
    a.save(swarm, weights)
    champion = random_genomes(np.random.default_rng(8), 1)[0]
    save_champion(weights, champion)
    b = Evolver(4)
    b.load(swarm, weights)
    assert np.array_equal(b.genomes[0], champion)
    assert np.array_equal(b.genomes[1:], a.genomes[1:])


def test_a_generation_under_way_when_the_population_is_replaced_is_dropped(monkeypatch):
    ev = Evolver(7)
    real_evaluate = evolve.evaluate

    def evaluate_then_reset(*args):
        result = real_evaluate(*args)
        ev.reset()
        return result

    monkeypatch.setattr(evolve, "evaluate", evaluate_then_reset)
    ev.step()
    assert ev.gen == 0


def test_a_roster_comes_from_a_swarm_or_a_single_champion(tmp_path):
    genomes = random_genomes(np.random.default_rng(5), POP_SIZE)
    path = tmp_path / "weights.npz"
    save_champion(path, genomes[3])
    assert np.array_equal(seat(load_roster(path), TANKS), np.tile(genomes[3], (TANKS, 1)))
    swarm = tmp_path / "swarm.npz"
    np.savez(swarm, population=genomes)
    assert np.array_equal(seat(load_roster(swarm), TANKS), genomes[:TANKS])

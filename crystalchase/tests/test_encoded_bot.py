import numpy as np
import pytest

from crystalchase.course import make_course
from crystalchase.encoded_bot import EncodedBot, encode_autopilot
from crystalchase.flock import Flock, solo_rule
from crystalchase.pilots import AutoPilot, Brains, load_pilot


def test_encoded_bot_agrees_on_unseen_observations():
    # Sampling is validation only: the compiler never consumes these samples.
    rng = np.random.default_rng(777)
    n = 10000
    angle = rng.uniform(-np.pi, np.pi, n)
    obs = np.column_stack([
        np.sin(angle), np.cos(angle), rng.uniform(0, 2, n),
        rng.uniform(-1.5, 1.5, n), rng.uniform(-1.5, 1.5, n), np.zeros(n),
    ])
    assert np.mean(encode_autopilot().act(obs) == AutoPilot().act(obs)) > 0.999


def test_encoded_bot_roundtrip_and_no_scripted_runtime(tmp_path, monkeypatch):
    bot = encode_autopilot()
    again = encode_autopilot()
    for (w, b), (w2, b2) in zip(bot.layers, again.layers):
        assert np.array_equal(w, w2) and np.array_equal(b, b2)
    path = tmp_path / "encoded.npz"
    bot.save(path)
    loaded = load_pilot(path)
    assert isinstance(loaded, EncodedBot)
    obs = Flock([make_course(1)], np.array([0]), solo_rule).observe()
    expected = bot.act(obs)

    def unavailable(*args):
        raise AssertionError("Scripted bot called at inference")

    monkeypatch.setattr(AutoPilot, "act", unavailable)
    assert np.array_equal(loaded.act(obs), expected)


def test_standard_racer_loading_still_works(tmp_path):
    from crystalchase.calculated import calculated_brain
    path = tmp_path / "standard.npz"
    brain = calculated_brain()
    brain.save(path)
    loaded = load_pilot(path)
    assert isinstance(loaded, Brains)
    assert np.array_equal(loaded.genomes, brain.genomes)


def test_encoded_bot_flies_like_the_original():
    n = 100
    courses = [make_course(seed) for seed in range(n)]
    neural, scripted = encode_autopilot(), AutoPilot()
    agrees = observations = 0
    crystals = []
    for pilot in (scripted, neural):
        flock = Flock(courses, np.arange(n), solo_rule)
        while not flock.done:
            obs = flock.observe()
            actions = pilot.act(obs)
            if pilot is scripted:
                agrees += np.count_nonzero(neural.act(obs) == actions)
                observations += n
            flock.step(actions)
        assert flock.alive.all()
        crystals.append(flock.crystals.mean())
    assert agrees / observations > 0.995
    assert abs(crystals[1] - crystals[0]) < 0.1


def test_layer_shapes_are_checked():
    with pytest.raises(ValueError, match="dimensions"):
        EncodedBot([(np.zeros((5, 6)), np.zeros(6))])

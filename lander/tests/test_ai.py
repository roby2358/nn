from dataclasses import replace

import numpy as np

from lander.ai.policy import INPUT, PolicyNet, encode, to_controls
from lander.ai.train import RESULTS, TRAIN, Trainer, potential
from lander.physics import CONFIG, NO_CONTROLS, Lander, Outcome, Rect, Terrain, World

PAD = Rect(400, 500, CONFIG.pad_width, CONFIG.pad_height)
SMALL = replace(TRAIN, episodes_per_step=4, max_decisions=20, eval_episodes=3)


def world_at(lander: Lander) -> World:
    return World(CONFIG, lander, PAD, Terrain(((0, CONFIG.ground_y), (CONFIG.width, CONFIG.ground_y))), Outcome.FLYING, NO_CONTROLS)


def test_encode_points_at_pad():
    left_of_pad = world_at(Lander(100, 100, 0, 0))
    dx, dy, vx, vy = encode(left_of_pad)
    assert dx > 0 and dy > 0 and vx == 0 and vy == 0
    assert encode(left_of_pad).shape == (INPUT,)


def test_potential_prefers_close_and_slow():
    far = world_at(Lander(100, 100, 0, 0))
    near = world_at(Lander(PAD.x + 28, PAD.y - 40, 0, 0))
    fast = world_at(Lander(PAD.x + 28, PAD.y - 40, 0, 90))
    assert potential(near) > potential(far)
    assert potential(near) > potential(fast)


def test_to_controls_maps_bits():
    c = to_controls(np.array([1, 0, 1]))
    assert c.main and not c.left and c.right


def test_update_raises_probability_of_rewarded_action():
    net = PolicyNet(np.random.default_rng(0))
    X = np.random.default_rng(1).standard_normal((16, INPUT)).astype(np.float32)
    A = np.tile(np.array([1, 0, 0], dtype=np.float32), (16, 1))
    before = net.probs(X)[:, 0].mean()
    for _ in range(20):
        net.update(X, A, np.ones(16, dtype=np.float32), 1e-2, 0.0)
    assert net.probs(X)[:, 0].mean() > before


def test_save_load_round_trip(tmp_path):
    a, b = PolicyNet(np.random.default_rng(0)), PolicyNet(np.random.default_rng(1))
    a.save(tmp_path / "w.npz")
    b.load_from(tmp_path / "w.npz")
    assert all(np.array_equal(p, q) for p, q in zip(a.params, b.params))


def test_trainer_step_records_stats():
    t = Trainer(PolicyNet(np.random.default_rng(0)), SMALL, 0)
    s = t.step()
    assert s.episodes == SMALL.episodes_per_step
    assert abs(sum(s.rates[r] for r in RESULTS) - 1.0) < 1e-9
    assert 0.0 <= t.eval_greedy() <= 1.0
    assert t.history[-1].greedy_landed is not None

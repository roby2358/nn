import numpy as np

from lander.ai.flight_stats import landed_profiles
from lander.ai.plummet import TANGENT_SPEEDS, build_policy, side_target
from lander.ai.policy import PolicyNet
from lander.physics import CONFIG


def rates_and_profiles():
    return landed_profiles(build_policy(), 300, 77)


def test_lands_nearly_every_round():
    rates, _ = rates_and_profiles()
    assert rates["landed"] >= 0.99


def test_plummets_before_it_brakes():
    _, profiles = rates_and_profiles()
    assert np.mean([p.first_burn_height for p in profiles]) < 300
    assert np.mean([p.peak_descent for p in profiles]) > 100


def test_touches_down_under_the_speed_limit():
    _, profiles = rates_and_profiles()
    assert max(p.touchdown_descent for p in profiles) < CONFIG.safe_vy


def test_weights_fit_the_network_and_use_only_spare_units():
    net = build_policy()
    assert isinstance(net, PolicyNet)
    used = len(TANGENT_SPEEDS) + 6
    assert np.count_nonzero(net.W1.any(axis=0)) == used


def test_side_target_is_zero_at_the_pad_and_grows():
    speeds = side_target(np.array([0.0, 50.0, 200.0, 800.0]))
    assert speeds[0] == 0
    assert np.all(np.diff(speeds) > 0)

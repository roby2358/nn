import numpy as np

from tictactoe.ai import minimax
from tictactoe.ai.policy import PolicyNet, encode
from tictactoe.ai.train import Trainer
from tictactoe.board import Board


def test_encode_fixed_view_with_turn():
    cells = ["X", None, "O"] + [None] * 6
    x = encode(cells, "X")
    o = encode(cells, "O")
    assert x.shape == (10,)
    assert list(x[:3]) == [1.0, 0.0, -1.0]
    # Board values are the same for both sides; only the turn input differs.
    assert (x[:9] == o[:9]).all()
    assert x[9] == 1.0 and o[9] == -1.0


def test_load_rejects_old_encoding(tmp_path):
    import pytest
    path = tmp_path / "old.npz"
    np.savez(path, W1=np.zeros((27, 64)), b1=np.zeros(64), W2=np.zeros((64, 9)), b2=np.zeros(9))
    with pytest.raises(ValueError):
        PolicyNet.load(path)


def test_update_lowers_prob_of_penalized_move():
    net = PolicyNet(np.random.default_rng(0))
    x = encode(["X"] + [None] * 8, "O")[None, :]
    before = net.forward(x)[0][0, 4]
    for _ in range(20):
        net.update(x, np.array([4]), np.array([-1.0], dtype=np.float32), entropy_beta=0.0)
    assert net.forward(x)[0][0, 4] < before


def test_choose_retries_on_occupied():
    net = PolicyNet(np.random.default_rng(0))
    b = Board()
    for i in range(8):
        b.cells[i] = "X" if i % 2 == 0 else "O"
    b.winner = None  # not a real game state, just forcing one empty cell
    assert net.choose(b, greedy=True) == 8


def test_minimax_blocks_and_wins():
    b = Board()
    for m in (0, 4, 1):  # X at 0,1 threatens 2; O must block
        b.play(m)
    assert minimax.best_moves(b) == [2]
    b = Board()
    for m in (0, 3, 1, 4):  # X can win at 2
        b.play(m)
    assert minimax.best_moves(b) == [2]


def test_minimax_never_loses_to_itself():
    b = Board()
    while not b.game_over:
        b.play(minimax.choose(b))
    assert b.winner is None


def test_training_step_runs_and_records():
    t = Trainer(seed=0, games_per_step=8)
    s = t.step()
    assert s.games == 8
    assert abs(s.x_win + s.draw + s.o_win - 1) < 1e-9
    assert 0 <= t.eval_vs_minimax(4) <= 1


def test_illegal_pick_is_retried_not_game_ending():
    t = Trainer(seed=0, epsilon=0.0)
    # Force the net to always name cell 0, so every pick after the first is illegal.
    t.net.W2[...] = 0.0
    t.net.b2[...] = 0.0
    t.net.b2[0] = 20.0
    recs, bad, winner, picks = t._play_one()
    # The game still runs to a normal finish with only legal moves recorded.
    assert len(recs) >= 5
    assert len({a for _, a, _ in recs}) == len(recs)
    assert len(bad) > 0 and bad[0][1] == 0  # the first illegal pick is the forced cell
    assert picks == len(recs) + len(bad)


def test_training_reduces_illegal_picks():
    t = Trainer(seed=0, epsilon=0.0)
    t.net.b2[0] = 3.0  # start with a strong habit of naming cell 0
    first = t.step().invalid
    for _ in range(40):
        last = t.step().invalid
    assert last < first


def test_retry_survives_zero_probability_on_legal_cells():
    t = Trainer(seed=0, epsilon=0.0)
    # All probability on cells 0 and 1; legal cells underflow to exactly 0.
    t.net.W2[...] = 0.0
    t.net.b2[...] = -200.0
    t.net.b2[0] = t.net.b2[1] = 200.0
    recs, bad, winner, picks = t._play_one()
    assert len(recs) >= 5


def test_losing_legal_move_does_not_raise_occupied_cells():
    net = PolicyNet(np.random.default_rng(0))
    cells = ["X", "O"] + [None] * 7
    x = encode(cells, "X")[None, :]
    occupied_before = net.forward(x)[0][0, :2].sum()
    for _ in range(20):
        net.update(x, np.array([4]), np.array([-1.0], dtype=np.float32), entropy_beta=0.0)
    occupied_after = net.forward(x)[0][0, :2].sum()
    assert occupied_after <= occupied_before + 1e-6

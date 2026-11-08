from tictactoe.board import Board


def test_x_moves_first_then_alternates():
    b = Board()
    assert b.turn == "X"
    assert b.play(0)
    assert b.turn == "O"
    assert not b.play(0)  # occupied
    assert b.turn == "O"


def test_row_win():
    b = Board()
    for i in (0, 3, 1, 4, 2):
        b.play(i)
    assert b.winner == "X"
    assert b.winning_line == (0, 1, 2)
    assert b.game_over
    assert not b.play(5)


def test_draw():
    b = Board()
    for i in (0, 1, 2, 4, 3, 5, 7, 6, 8):
        b.play(i)
    assert b.winner is None
    assert b.is_draw


def test_reset():
    b = Board()
    b.play(4)
    b.reset()
    assert b.cells == [None] * 9 and b.turn == "X"

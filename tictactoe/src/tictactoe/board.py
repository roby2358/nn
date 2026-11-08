"""Pure game logic, no pygame dependency."""

from __future__ import annotations

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


class Board:
    def __init__(self) -> None:
        self.cells: list[str | None] = [None] * 9
        self.turn = "X"
        self.winner: str | None = None
        self.winning_line: tuple[int, int, int] | None = None

    @property
    def is_draw(self) -> bool:
        return self.winner is None and all(self.cells)

    @property
    def game_over(self) -> bool:
        return self.winner is not None or self.is_draw

    def play(self, index: int) -> bool:
        """Place the current player's mark. Returns True if the move was legal."""
        if self.game_over or not 0 <= index < 9 or self.cells[index] is not None:
            return False
        self.cells[index] = self.turn
        for line in LINES:
            a, b, c = line
            if self.cells[a] and self.cells[a] == self.cells[b] == self.cells[c]:
                self.winner = self.turn
                self.winning_line = line
                break
        if not self.game_over:
            self.turn = "O" if self.turn == "X" else "X"
        return True

    def reset(self) -> None:
        self.__init__()

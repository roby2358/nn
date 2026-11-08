# tictactoe

Pygame tic-tac-toe with a self-play policy network that learns while you watch.

Board on the left, controls and a live chart on the right:

- **Play** starts a game against the current net (you are X; `S` swaps sides).
- **Clear** resets the net to random weights and clears the chart.
- **Train** runs 1000 steps of 64 self-play games each in the background, capped per press. **Stop** halts early.
- **Save** / **Load** write and read `weights.npz` in the project dir.

The chart stacks self-play outcomes per batch: X wins at the bottom, draws in the middle, O wins on top.
The thin yellow line is the greedy net's draw rate against a perfect minimax player, measured every 10 steps.
Rising draws and a yellow line approaching the top mean it has learned the game.

The net has no lookahead: it emits a probability over the 9 cells and the top one is played.
If that cell is occupied it tries the next. Training does the same retry, and keeps two lessons apart:

- Game results only change which legal cell the net prefers. A lost game can't push it toward occupied cells.
- Each illegal pick is penalized on its own, separate from the game result. The penalty pushes every occupied
  cell in that position down at once, so an illegal move is always worse than any legal one, even for a losing side.

Expect illegal picks to fall to a few percent and the draw rate against perfect play to approach 100% within about
2000 training steps, which is two presses of Train.

`Esc`/`Q` quits.

The project dir is shared between container and host, so each side gets its own venv at `.venv-<hostname>/` (gitignored). Override with `VENV=path just setup`.

```sh
just setup   # create .venv and install (run on host AND in container as needed)
just test   # headless tests, fine inside the container
just up      # launch the GUI, from the host
just train 300   # headless training in the container, saves weights.npz
```

Requires `uv` and `just`. If you don't have `uv` on the host, equivalent commands:

```sh
python -m venv .venv-$(hostname) && .venv-$(hostname)/bin/pip install -e ".[dev]"
.venv-$(hostname)/bin/python -m tictactoe.game
```

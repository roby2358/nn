# nn

Small pygame games with tiny neural networks written in numpy. Each game trains its net while you watch. Afterwards
you can play against it (`tictactoe`), watch it fly (`lander`, `shards`) or race it (`crystalchase`).

Each game except `shards`, so far, also has weights built without training, calculated directly from the rules or the physics and loaded into
the same kind of net as the learned ones. That makes it easy to compare what a net can learn with what it can
represent.

| Project | Game | How the net learns | Weights built without training |
| --- | --- | --- | --- |
| [`tictactoe`](tictactoe/README.md) | Tic-tac-toe against the net | Self-play | [`weights-1`](weights-1/README.md) compiles minimax results into the net and checks it never loses |
| [`lander`](lander/README.md) | Lunar lander, fly it yourself or watch | REINFORCE | A handcrafted controller and a suicide-burn pilot |
| [`crystalchase`](crystalchase/README.md) | Race neural pilots to crystals in a starfield | Evolution, no gradients | The autopilot's control law compiled into the net |
| [`shards`](shards/README.md) | Blast drifting crystal rocks, fly it yourself or watch | Evolution, no gradients | Not built yet |

Sample runs: [`lander/landings.mp4`](lander/landings.mp4), [`lander/landings-plummet.mp4`](lander/landings-plummet.mp4),
[`crystalchase/crystal-loop.mp4`](crystalchase/crystal-loop.mp4), [`crystalchase/crystal-swarm.mp4`](crystalchase/crystal-swarm.mp4).

## Running

You need Python 3.11+, [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/). Each game is its own
package, so run these from inside its directory:

```sh
cd lander         # or tictactoe, crystalchase, shards
just setup        # create this host's venv (.venv-<hostname>) and install
just test         # headless tests, no display needed
just up           # launch the game, needs a display
just              # list every recipe
```

## Weights are not in the repo

All `*.npz` files are gitignored, so a fresh clone has no saved weights. Train new ones or rebuild the calculated ones
from inside the project directory, after `just setup`. Run the `python` commands with the venv active. `weights-1`
has no venv of its own, so use the one from `tictactoe`.

| Project | Train | Build without training |
| --- | --- | --- |
| `tictactoe` | `just train 2000` | From `weights-1/`: `python build_weights.py`, then copy `weights.npz` into `tictactoe/` |
| `lander` | `just train 1000` | `just cc-plummet` or `python -m lander.ai.handcrafted` |
| `crystalchase` | `just evolve 300` | `python -m crystalchase.calculated --sector` |
| `shards` | `just evolve 400` | Not built yet |

Each of these writes `weights.npz` in the project directory, replacing any file already there. `crystalchase` and
`shards` also save their whole population to `swarm.npz` when evolving, and `python -m crystalchase.encoded_bot` writes the extra
racer `racers/encoded-autopilot.npz`.

# tenks

Eight green tanks fight eight blue tanks on a wrapped field, every tank driven by its own small neural net. The nets
learn by evolution. You can take over one of the tanks yourself.

The battlefield is on the left. On the right are the controls and a live chart:

- **Drive** puts you in the ringed green tank: up/down drive, left/right turn, space fires, `R` restarts. The battle has
  no time limit and ends when you die or a team is wiped out. **Watch** goes back to the leaders fighting each other.
- **Evolve** runs up to 500 generations in the background. **Stop** halts early.
- **Clear** starts over from a random population.
- **Save** / **Load** write and read `swarm.npz` (all 64 nets and the chart history) and `weights.npz` (the leader).
  Both load automatically at startup.

`F` cycles watch speed (x1, x4, x16). `Esc`/`Q` quits.

## The game

- **Treads:** each net sets a left and a right tread in -1..1. Speed follows the treads directly, with no momentum.
  Equal treads drive straight, opposite treads spin in place, and both negative backs up.
- **Gun:** fixed to the hull, so aiming means turning the tank. A shoot output above 0.5 fires, then the gun reloads for
  2 s.
- **Shells:** they fly at 450 px/s for 180 px, a quarter of the field. A shell goes off when it touches any live tank or
  reaches full range.
- **Blast:** every tank within 30 px of the detonation dies outright. That includes teammates and the shooter.
- **Tanks** block each other and wrecks don't block anything. Shell range is under half the field, so nothing can shoot
  itself around the wrap.
- A battle lasts 30 s in training. If a team is wiped out, the survivors hold still and keep earning time.

## The net

- **Inputs (25, all in the tank's own frame and across the wrap):** gun ready, then the four nearest live enemies and
  the four nearest live teammates, each as bearing (sin, cos) and closeness. An absent contact is zeros. The net sees
  positions only: no enemy heading or speed, and no shells.
- **Net:** 32 tanh hidden units, then left tread, right tread and shoot.
- **Genome:** 931 weights, evolved without gradients.

## Fitness

Each net drives one tank. A shell's owner gets the credit for it, even if the owner has died by the time it lands.

| Event | Points |
| --- | --- |
| Each second alive | +0.1 |
| Enemy killed by your shell | +5 |
| Teammate (or yourself) killed by your shell | -5 |
| Near miss: your shell came within 60 px of a live enemy and killed no enemy | +0.5 |

Each generation the population of 64 is dealt at random into four battles, three times over, so every net fights three
battles in random seats alongside random teammates. The top 8 by mean fitness survive unchanged and are re-tested. The
other 56 are mutated copies of them, a quarter of them crossed with a second elite.

Fitness is relative to whoever else is alive, so a rising number in the log doesn't by itself mean improvement.
`just cc-stats` is the fixed yardstick: it plays a file's nets against fresh random nets on the same battles every time.

All tuning numbers are in `src/tenks/config.py`.

## Running

The project dir is shared between container and host, so each side gets its own venv at `.venv-<hostname>/`
(gitignored).

```sh
just setup        # create this host's venv and install
just test         # headless tests
just up           # launch the GUI, needs a display
just evolve 300   # evolve headless, resumes from and saves swarm.npz + weights.npz
just cc-stats swarm.npz 200                  # win rate and rates per tank against random nets
just cc-video swarm.npz 0 30 tenks.mp4       # record one battle
```

Requires `uv` and `just`.

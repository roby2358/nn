# shards

Pygame arcade shooter where small neural nets learn, by evolution, to fly a ship through a field of drifting crystal
rocks and blast them apart. You can take the controls yourself.

The field is on the left. On the right are the controls and a live chart:

- **Fly** gives you the ship: left/right turn, up thrusts, space fires, `R` restarts. **Watch** goes back to the champion.
- **Evolve** runs up to 500 generations in the background. **Stop** halts early.
- **Clear** starts over from a random population.
- **Save** / **Load** write and read `swarm.npz` (all 64 nets and the chart history) and `weights.npz` (the leader).
  Both load automatically at startup.

`F` cycles watch speed (x1, x4, x16). `Esc`/`Q` quits.

## The game

The ship turns and thrusts with inertia, bleeds a little speed as it coasts, has a top speed, and wraps at every edge.
Shots add the ship's velocity, last about a second, and only four can be in the air at once.

A shot breaks a big rock into two medium ones, a medium into two small, and a small rock is gone. Smaller rocks drift
faster and score more: 1, 2 and 3 points for big, medium and small. Clearing the field brings a new wave with one more
big rock, up to eight. Touching a rock ends the round, and a round lasts at most 45 s.

## The pilot

- **Inputs (27, all in the ship's own frame and across the wrap):** the ship's forward and sideways velocity, whether
  the gun is ready, and for each of the four nearest rocks its bearing (sin, cos), closeness, relative velocity
  (forward, sideways) and radius. A slot with no rock is zeros.
- **Net:** 24 tanh hidden units, then three continuous outputs: turn in -1..1, thrust in 0..1, and fire in 0..1
  (the gun fires above 0.5).
- **Fitness:** points scored, plus 0.1 per second alive, minus 0.5 per shot fired. A typical hit is worth about 2.4 points,
  so a shot pays for itself only if it has better than about a 20% chance of landing.
- **Genome:** 747 weights, evolved without gradients. Every net flies the same three freshly seeded fields each
  generation. The top 8 survive unchanged and are re-tested on new fields, so luck doesn't stick. The other 56 are
  mutated copies of them, a quarter of them crossed with a second elite.

All tuning numbers are in `src/shards/config.py`.

## Running

The project dir is shared between container and host, so each side gets its own venv at `.venv-<hostname>/` (gitignored).

```sh
just setup        # create this host's venv and install
just test         # headless tests, fine inside the container
just up           # launch the GUI, needs a display
just evolve 300   # evolve headless, resumes from and saves swarm.npz + weights.npz
just cc-stats weights.npz 100   # how a weights file plays on 100 fixed fields
just cc-shot 6    # headless screenshots shot-watch.png / shot-fly.png, 6 s in
```

Requires `uv` and `just`.

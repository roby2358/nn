# lander

Pygame lunar lander with a small numpy policy network that learns to land while you watch.

Scene on the left, controls and a live chart on the right:

- **Fly**: you have the controls. `W`/`S` main thrust, `A`/`D` sideways, `R` restart.
- **Watch**: the current net flies greedily, round after round.
- **Train** runs 1000 steps of 32 rounds each in the background. **Stop** halts early.
- **Clear** resets to a random net. **Save** / **Load** write and read `weights.npz` (loaded on startup if present).

`Esc` or the window's close button quits; Ctrl+C in the terminal kills it.

## How it learns

The net sees 4 numbers: the pad's offset from the lander's feet (x, y) and the lander's velocity (vx, vy).
It outputs an independent on/off probability for each thruster, and chooses every 4 frames (15 times a second),
holding the choice in between.

Training is REINFORCE. Batches of 32 rounds fly side by side. Each decision earns the change in a
"potential" (closer to the pad and slower is better), and the round ends with +3 for landing or -1 for
a crash, flying off the side, or a 20 s timeout. Choices with better-than-average returns become more likely.

The chart stacks each batch's results: landed at the bottom, then crashed, lost and timeout.
The yellow line is the greedy net's landing rate, measured every 10 steps. Expect it to approach
100% within about 1000 steps, roughly a minute headless.

## Running

To create a deterministic, handcrafted controller without training, run
`python -m lander.ai.handcrafted` in the installed environment. This writes
`weights.npz` using fixed weights that steer toward the pad and regulate
descent speed. Select **Watch** in the game to see it fly.
The controller free-falls until its estimated stopping distance reaches
the pad, makes a sustained braking burn, then settles onto it. Sideways
control includes a coasting dead zone to avoid alternating jets.
Verify the saved weights and 2,000 independently seeded flights with
`python scripts/verify_handcrafted.py`. Failures report a seed that can be
replayed with `--seed-start SEED --episodes 1`. This check needs NumPy,
but no Pygame or display, and never updates the weights.

The project dir is shared between container and host, so each side gets its own venv at `.venv-<hostname>/`.

```sh
just setup        # create this host's venv and install
just test         # headless tests, fine inside the container
just up           # launch the GUI (needs a display)
just train 1000   # headless training; continues from and saves weights.npz
just cc-record 8 landings.mp4   # render 8 landings to video, scene only, no display needed
just cc-plummet 5000            # build hand-wired suicide-burn weights, measure, overwrite weights.npz
just cc-flight-stats weights.npz 5000   # how a weights file flies: landing rate, burn timing, speeds
```

# crystalchase

Pygame starfield where small neural nets learn, by evolution, to fly a ship to crystals.
The controls are rotate and thrust, with pure inertia: no drag, so a ship has to turn around and burn to brake.
Then you race them.

The field is on the left. On the right are the controls and a live chart:

- **Race** puts you (green, arrow keys), the current champion (cyan), the scripted autopilot (gold) and any snapped
  racers on one course. There is one shared crystal, and the first to touch it takes it. **Watch** goes back.
- **Evolve** runs up to 500 generations in the background. **Stop** halts early.
- **Clear** starts over from a random population.
- **Save** writes `swarm.npz` (all 64 nets plus the chart history) and `weights.npz` (the leader alone).
  **Load** reads the swarm, then writes the leader from `weights.npz` over the swarm's top spot. With no
  `swarm.npz` you get a random swarm under that leader. With no `weights.npz`, the swarm's own leader stays.
  Both files load automatically at startup, and `weights.npz` is in the same format as a `racers/` file.
- **Snap** saves the current champion to `racers/gen-NNNN.npz`. Every file in `racers/` joins the next race, so old
  champions can race new ones.

`F` cycles replay speed (x1, x4, x16). `Esc`/`Q` quits.

## What you're watching

Each generation, all 64 nets fly the same two freshly seeded courses, solo. While evolving, the left side replays
the newest generation on its first course. Every net is a faint ghost with its own next crystal drawn faintly. The current
leader is bright, with a trail, and the autopilot flies along in gold as a yardstick. An arrow on the border marks a
ship that is off-screen. The crash wall is 120 px outside the window, out of view.

When evolution stops, the view switches straight to **free flight**: the champion alone on an endless course. Every
30 s a new lap begins. The crystal count and score reset, while the ship and its crystal carry on, and the previous
lap's result stays on screen. Only a crash ends free flight, and then a new course starts.

The chart shows best, mean and autopilot score per generation. Raw values are faint and the moving average is bold.
Each generation flies new courses, so the raw lines are noisy.

## Scoring

A run lasts 30 s. Each crystal is worth `1 + speed bonus`. The bonus is +1 if the pickup took 2 s or less, fading to 0
at 10 s. On top of that, up to +0.5 goes for progress toward the crystal being chased when the run ends. Without it,
early nets that never touch a crystal would all score 0 and evolution would have nothing to select on. A crash ends
the run and keeps whatever was earned.

## The pilot

- **Inputs (6, all in the ship's own frame):** bearing to the crystal (sin, cos), distance, forward and sideways
  velocity, and closing speed.
- **Net:** 16 tanh hidden units, then 6 outputs, one per action: {left, none, right} × {coast, thrust}. The
  highest output is flown.
- **Genome:** 214 weights, evolved without gradients. The top 8 nets survive unchanged and are re-tested on new
  courses each generation, so luck doesn't stick. The other 56 are copies of elites, picked with a bias toward the
  better ranks. A quarter of them are uniformly crossed with a second elite. Every copy gets Gaussian noise at one of
  three scales (fine-tune, explore, leap).

The autopilot sees exactly the same six inputs. It works out the velocity it wants (toward the crystal, slow enough to
brake into it), points along the velocity error, and burns when roughly aligned.

All tuning numbers are in `src/crystalchase/config.py`.

## Running

The project dir is shared between container and host, so each side gets its own venv at `.venv-<hostname>/`
(gitignored).

```sh
just setup        # create this host's venv and install
just test         # headless tests, fine inside the container
just up           # launch the GUI, from the host
just evolve 300   # evolve headless (~5 gen/s), resumes from and saves swarm.npz + weights.npz
just cc-shot 6    # headless screenshots shot-watch.png / shot-race.png, 6 s in
```

Requires `uv` and `just`.

## Current compact fly-through pilot

The active champion in `weights.npz` is now the **sector net**: the autopilot's
control law compiled into the ordinary 6 → 16 → 6 net with no training. Twelve
hidden units compute the velocity error for a target speed of
`60 + 0.8 * distance`. Each action's logit is linear in that error, so `argmax`
selects an action by the error's angle, and the turn and thrust cones need no
hidden units at all. Regenerate it with `python -m crystalchase.calculated --sector`.
`just cc-sector-search` chooses its five constants by evaluation only.

The notes below describe the previous 300 px/s champion, which
`racers/encoded-autopilot.npz` still holds.

The previous champion and neural racer were set to **300 px/s**, with
velocity-correction alignment before thrust. The generator's cruise default
is also 300. The strict crystal-facing mode remains optional and inactive.
Both saved files use the original 6 → 16 → 6 architecture and the successful
262 px nominal turn range. Measurements below identify their cruise settings.

An optional strict crystal-alignment experiment is available with
`--fly-through --orient-first`: it forbids thrust outside a 15-degree bearing
cone and prioritizes pointing toward the crystal. It uses ten hidden units in
the same 6 → 16 → 6 architecture. This performed much worse: 873 crashes in
900 additional courses (97%), averaging 1.88 crystals per run. The constraint
prevents many useful sideways momentum corrections. It is preserved as
`pilot-archive/flythrough-300-orient-first.npz`; the working 300 px/s weights
remain active. The strict mode is not the generator's default.

The saved pilots now use a **300 px/s** cruise target with a nominal 262 px turn
range, in the unchanged 6 → 16 → 6 network. On seeds 0–99, 37 of 100 runs crashed;
on additional seeds 100–999, 250 of 900 crashed (27.8%), averaging 7.23 crystals
per run. Pickup speed averaged 258.5 px/s with a 297.2 px/s median on that
additional set. Both `weights.npz` and the neural racer have been updated.
Regenerate the champion with
`python -m crystalchase.calculated --fly-through --cruise-speed 300`.
Backups of the prior variants are in `pilot-archive/flythrough-200.npz` and
`pilot-archive/flythrough-400.npz`.

The previous experimental **400 px/s** pilot used a
nominal 340 px turn range. The original 6 → 16 → 6 architecture is unchanged.
On an 800 px straight approach the pilot collected at 398.7 px/s. On normal
development courses 0–99, however, 92 of 100 runs crashed, with 3.44 crystals
per run and 275.6 px/s mean pickup speed. Many legs do not offer enough space
to accelerate and flip; even ideal braking from 400 takes about 308 px.
The reliable 200 px/s weights are preserved at `pilot-archive/flythrough-200.npz`.

`python -m crystalchase.calculated --fly-through --cruise-speed 400` regenerates
the experimental champion; use `--cruise-speed 200` for the previous behavior.
The measurements below describe that previous 200 px/s version.

The current `weights.npz` and `racers/encoded-autopilot.npz` both use the original
**6 → 16 → 6** architecture (214 parameters), with seven active hidden units.
They accelerate toward the crystal, cut thrust near 200 px/s closing speed,
coast while aligned, then turn near a nominal fixed range of 183 px. A newly acquired target changes the closing-speed
input and permits thrust again to redirect the old velocity. The scripted bot
remains unchanged at its 140 px/s arrival target. A precise 210 px/s reference
aligns the velocity before the 200 px/s thrust cutoff. At cruise the turn range
is `pickup radius + speed * pi / turn rate`, enough for a half-turn before
contact. An approach guard delays the turn one pixel per px/s below cruise
and releases it on receding approaches; the soft boundary is
`distance - 183 + (200 - closing_speed) = 0`. Thus 183 px is a nominal outer
range, not a hard distance-only switch. Thrust is suppressed during the final
turn; braking is left to reorienting for the next target.

`python -m crystalchase.calculated --fly-through` regenerates the compact
champion's `weights.npz`. **Load** selects it as the champion; starting a new
**Race** reloads the compact neural racer under its existing label.

This reactive controller has no phase memory and cannot guarantee every pickup
at 200 px/s or facing away. A centered 500 px approach collected at 199.3 px/s,
facing backward, after coasting straight and then turning without thrust.
On development seeds 0–99 it averaged 8.95 crystals with no crashes and
194.5 px/s mean pickup speed. On additional seeds 100–999 it averaged 8.94
crystals with one crash, 99.2% backward-facing pickups, 194.6 px/s mean pickup
speed, and 199.9 px/s median. Regression tests check the coasting leg, delayed
turn, real pickup speeds, and the ideal straight-line case.

The previous deeper racer is preserved in
`pilot-archive/encoded-autopilot-layered-180.npz`, outside the active racers folder.
The section below describes its exporter; running that exporter replaces the
compact neural racer with a larger network again.

## Scripted autopilot encoded as a deeper neural racer

`python -m crystalchase.encoded_bot` constructs
`racers/encoded-autopilot.npz` without training. Restart the game, then start a **Race** to include
it alongside the scripted autopilot and the current champion. The champion's
`weights.npz` is preserved. The new racer loader accepts both the original
genome files and this layered-network format.

This is a larger network: **6 → 260 → 10 → 6 → 6 → 6**, with tanh hidden
layers, linear output logits, and 4,580 parameters. The first layer approximates
the bot's capped square-root speed curve using 256 smooth steps. Subsequent
layers approximate bearing products and squared error magnitude through tanh
finite differences, then encode the bot's steering deadband, thrust cone, and
minimum velocity-error threshold. Coefficients come directly from equations
and configuration constants; no optimizer or teacher-data fitting is used.
Runtime uses only matrix operations, tanh, and argmax, without calling AutoPilot.
The archived layered racer targets 180 px/s through the crystal; the scripted
bot remains at 140 px/s. Running this compiler normally regenerates its 140 px/s
version. The measurements below describe that matching layered version.

This is an approximation, not an exact compilation of the original branching
program. Actions can differ near thresholds, at degenerate zero-error poses,
and beyond the normal simulation range. On 10,000 fixed validation observations
(distance 0–800 px, forward and sideways speeds ±450 px/s), action agreement was
99.99%. On courses seeded 0–99, agreement on the scripted bot's observations
was 99.675%; independent flights averaged 11.04 crystals for the neural racer
and 11.03 for the scripted bot, with no crashes for either.
An additional check on seeds 100–999 gave 99.678% action agreement; independent
neural and scripted flights averaged 11.106 and 11.110 crystals respectively,
with zero crashes for either across those 900 courses.

The layered racer cannot be inserted into the existing 214-parameter evolving
population. It is a separate racer; regenerating it reads the current bot
constants. Changing the scripted bot does not automatically change saved weights.

## Calculated pilot (no training)

`python -m crystalchase.calculated` writes `weights.npz` deterministically
using algebraically constructed weights, with no training or randomness.
Run it in an environment where the project is installed (or set `PYTHONPATH=src`).
It replaces the saved leader; press **Load** in an open game or restart to use it.

The default now **burns toward the crystal, coasts to turn, then burns backward
to brake into the pickup**. It uses all sixteen hidden units, with the same six
inputs and ordinary saved-file format. Twelve approximate steering toward a
distance-dependent velocity target (`77 + 0.5 * distance` px/s). Four combine
range, closing speed, and nose alignment to decide when thrust helps.

The approximate phase boundary is `distance + 154 - 2 * closing_speed = 0`,
with distances in pixels and speeds in pixels/s. Faster approaches therefore
start turning farther out. This affine rule approximates turning travel plus
stopping distance over typical approach speeds; it is not an exact intercept
solution. The gates allow enough angular freedom to correct sideways drift.
This is a reactive policy, without phase memory: it may revise the maneuver
as its approach changes, and cannot guarantee a backward-facing pickup.

On development seeds 0–99 it averaged 9.12 crystals per 30-second run, with no
crashes. On additional seeds 100–999, it averaged 9.18 crystals, also with no
crashes; it coasted during 64.6% of turning frames, faced against its incoming
velocity at 75.7% of pickups, and averaged 96.3 px/s at pickup. Backward-facing
here means the nose opposes the actual velocity; it does not measure exact
alignment with the crystal or field center. All weights remain calculated,
with manual design checks and no training.

`python -m crystalchase.calculated --full-burn` restores the preceding pilot:
**full burn, never coast**, with broader arcs aimed at the
crystal. Twelve hidden neurons approximate a distance-dependent velocity target:
`desired speed = 130 + 0.4 * distance` in pixels per second. Long approaches allow
longer arcs; the lower target near a crystal turns the ship back earlier to
brake. Eight units approximate distance times bearing, and four handle bearing
and velocity directly. These weights come from a finite-difference identity
for tanh, without fitting data. Every burning action has the same output weights as
its coasting counterpart and a bias exactly one higher. Thus the ordinary
network's argmax always selects thrust, without changing physics or overriding
actions outside the network. The six original inputs and saved-file format
remain unchanged.

On development seeds 0–99, the broader arcs averaged 7.98 crystals per 30-second
run with 6 crashes. On additional seeds 100–999, they averaged 7.99 crystals
with 81 crashes out of 900. The previous two-unit full-burn pilot averaged 6.78
with 50 crashes on those same courses. Travel per equivalent full turn rose
from 371 to 438 pixels (18%), and time spent turning fell from 64.1% to 59.6%.
Thus the arcs are longer and collect more, with a higher crash rate. Inward
pickups fell from 73.7% to 68.1%; center-facing behavior remains approximate.

Use `python -m crystalchase.calculated --tight-curls` to restore the previous
two-unit full-burn pilot with its constant 182 px/s steering target.

Use `python -m crystalchase.calculated --allow-coast` to regenerate the previous
five-unit turnaround pilot described below. Add `--pursuit` to that command to
restore the earlier 208 px/s coasting pursuit pilot. Without `--allow-coast`,
`--pursuit` also keeps full burn, using the higher steering target.

**Previous pilot, with coasting enabled:**

The controller subtracts the ship's velocity from a desired velocity toward the
crystal. Five tanh units encode sideways steering, a steering deadband, and
three soft conditions for thrust: the velocity error points forward and lies
inside an alignment cone. The six output scores combine turn and thrust
decisions. All other hidden units have zero weights.

Steering uses a 182 px/s target while thrust uses 312 px/s. The higher thrust
target permits harder burns; exceeding the lower steering target biases the
nose back against the incoming motion while inertia carries the ship toward
the crystal. These are controller setpoints, not actual speed guarantees. This
is an approximate turnaround, with no center or next-crystal information.
Distance and closing speed remain unused, and the input format is unchanged.

On courses seeded 0–99, the coasting turnaround pilot completed all 30-second runs without crashing,
averaged 7.49 crystals, and faced into the center's hemisphere at 84.4% of pickups.
The earlier 156 px/s pilot averaged 6.8 crystals. The faster 208 px/s pursuit
version averaged 8.03 crystals but faced inward at 69.5% of pickups; regenerate
that version with `python -m crystalchase.calculated --allow-coast --pursuit`.

These are manually designed and checked controllers; no optimizer, evolution,
or gradient training was used. Several manual designs were evaluated, so these
course results are development measurements, not an independent generalization
claim. Facing the exact center cannot be guaranteed from the six inputs.
An additional check on seeds 100–999, performed after choosing the final design,
gave 7.62 crystals per run, 83.0% inward-facing pickups, and 3 crashes in 900 runs.

"""Every tuning number lives here, so physics, scoring, evolution and UI agree."""

import math

import numpy as np

TAU = 2 * math.pi

# ---- field & physics (pixels, seconds) ------------------------------------------
FIELD_W = 720.0
FIELD_H = 720.0
FIELD = np.array([FIELD_W, FIELD_H])
DT = 1.0 / 60.0           # fixed physics step, identical headless and in the GUI
RUN_SECONDS = 30.0        # the limit on a round in training
RUN_STEPS = int(round(RUN_SECONDS / DT))
FREE_PLAY_STEPS = math.inf  # a round on screen, flown by you or the champion, ends only in a crash

# ---- ship: coasts on inertia, bleeds a little speed, and wraps at the edges ------
THRUST = 340.0            # px/s^2 at full thrust
TURN_RATE = 4.4           # rad/s at full turn
DRAG = 0.45               # fraction of speed lost per second, as a continuous decay rate
MAX_SPEED = 460.0
SHIP_R = 5.5

# ---- gun ----------------------------------------------------------------------------
SHOT_SPEED = 520.0        # added to the ship's own velocity
SHOT_LIFE = 1.05          # seconds in flight
SHOT_SLOTS = 4            # most shots in the air at once
COOLDOWN = 0.18           # seconds between shots
FIRE_ABOVE = 0.5          # the fire command pulls the trigger above this

# ---- rocks: index by size, where 0 is an empty slot and each hit drops a size -----
BIG = 3
RADIUS = np.array([0.0, 6.5, 12.5, 22.0])
SPEED_LO = np.array([0.0, 80.0, 50.0, 30.0])
SPEED_HI = np.array([0.0, 150.0, 105.0, 70.0])
POINTS = np.array([0.0, 3.0, 2.0, 1.0])

# ---- waves --------------------------------------------------------------------------
WAVE_BASE = 6             # big rocks in the first wave; each later wave adds one
WAVE_MAX = 12
ROCK_SLOTS = WAVE_MAX * 4  # a big rock is at most four small rocks at once
WAVE_TABLE = 16           # distinct waves stored per scenario, then they repeat
DRIFT_TABLE = 256         # stored rock drifts per scenario, then they repeat
SPAWN_CLEAR = 170.0       # a new rock this close to the ship moves to the far side of the field

# ---- scoring ----------------------------------------------------------------------
SURVIVAL_POINTS = 0.1     # fitness per second alive, so dodging is worth something
SHOT_COST = 0.5           # fitness lost per shot fired, so a shot has to be likely to hit to be worth taking

# ---- observation ------------------------------------------------------------------
NEAREST = 4               # rocks the net can see, closest surface first
ROCK_INPUTS = 6           # sin bearing, cos bearing, closeness, v forward, v sideways, radius
SELF_INPUTS = 3           # v forward, v sideways, gun ready
N_MOODS = 2               # outputs that come back as inputs on the next step: the net's only memory
N_INPUTS = SELF_INPUTS + NEAREST * ROCK_INPUTS + N_MOODS
SENSE_RANGE = 500.0
SPEED_SCALE = 300.0
RADIUS_SCALE = 22.0

# ---- net: turn in -1..1, thrust in 0..1, fire in 0..1, and the moods in -1..1 ------
N_HIDDEN = 24
N_OUTPUTS = 3 + N_MOODS
CMD_TURN, CMD_THRUST, CMD_FIRE, CMD_PASSION, CMD_PATIENCE = range(N_OUTPUTS)

# ---- input layout: the ship's own values, then NEAREST rocks of ROCK_INPUTS values each
V_FORWARD, V_SIDEWAYS, GUN_READY = range(SELF_INPUTS)
PASSION, PATIENCE = range(N_INPUTS - N_MOODS, N_INPUTS)
MOOD_PASSION, MOOD_PATIENCE = range(N_MOODS)    # columns of Worlds.mood
ROCK_SIN, ROCK_COS, ROCK_CLOSE, ROCK_V_FORWARD, ROCK_V_SIDEWAYS, ROCK_RADIUS = range(ROCK_INPUTS)


def rock_input(rock: int, field: int) -> int:
    """Where one value of the rock in position `rock` (0 is nearest) sits in the input vector."""
    return SELF_INPUTS + rock * ROCK_INPUTS + field


# ---- evolution --------------------------------------------------------------------
POP_SIZE = 64
ELITES = 8
SCENARIOS_PER_GEN = 3     # every net flies these many fields; fitness is the mean
CROSSOVER_RATE = 0.25
CROSSOVER_GENE_ODDS = 0.5  # a crossed child takes each weight from the second elite this often
MUTATION_SIGMAS = (0.02, 0.06, 0.2)   # each child draws one: fine-tune, explore, leap
GENS_PER_PRESS = 500
SAVE_EVERY = 100          # headless evolution writes swarm and weights this often, in generations

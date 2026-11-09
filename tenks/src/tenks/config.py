"""Every tuning number lives here, so physics, scoring, evolution and video agree."""

import math

import numpy as np

TAU = 2 * math.pi

# ---- field & time (pixels, seconds) -----------------------------------------------
FIELD_W = 720.0
FIELD_H = 720.0
FIELD = np.array([FIELD_W, FIELD_H])
DT = 1.0 / 60.0
RUN_SECONDS = 30.0
RUN_STEPS = int(round(RUN_SECONDS / DT))
FREE_PLAY_STEPS = math.inf   # a battle you drive in yourself has no clock
GENS_PER_PRESS = 500         # one press of Evolve in the GUI

# ---- teams: tanks 0..7 are green, 8..15 are blue -----------------------------------
TEAM_SIZE = 8
TANKS = 2 * TEAM_SIZE
SPAWN_LINE = 0.1 * FIELD_W  # green's vertical line, and blue's the same distance from the far side
SPAWN_MARGIN = 40.0       # no tank starts closer than this to any edge
SPAWN_HALF_BAND = SPAWN_LINE - SPAWN_MARGIN   # each team starts within this of its own line
SPAWN_GAP = 40.0          # least distance between two tanks of a team at the start

# ---- tanks: speed follows the treads directly, with no momentum --------------------
MAX_SPEED = 35.0          # px/s with both treads full forward
TURN_RATE = 2.2           # rad/s with the treads full and opposite
TANK_R = 5.0

# ---- gun: fixed to the hull, one shell in flight per tank --------------------------
RELOAD = 2.0              # seconds between shots
FIRE_ABOVE = 0.5          # the shoot command pulls the trigger above this
SHELL_SPEED = 450.0
SHELL_RANGE = FIELD_W / 4
SHELL_LIFE = SHELL_RANGE / SHELL_SPEED
MUZZLE_GAP = 2.0          # a new shell starts this far outside its own hull
SPLASH_R = 15.0           # every tank this close to a detonation dies
NEAR_MISS_R = 60.0        # a shell that came this close to an enemy and killed none is a near miss

# ---- scoring, per tank ------------------------------------------------------------
SURVIVAL_POINTS = 1.0     # per second alive
KILL_POINTS = 5.0         # per enemy your shell kills
FRIENDLY_KILL_COST = 5.0  # per tank of your own team your shell kills, yourself included
NEAR_MISS_POINTS = 0.5    # per near miss

# ---- observation ------------------------------------------------------------------
NEAREST = 4               # enemies the net can see, and friends too, closest first
# one contact: its bearing as sin and cos, its closeness, how far its gun points off me as sin and cos, gun ready
SIN, COS, CLOSE, GUN_SIN, GUN_COS, LOADED = range(6)
CONTACT_INPUTS = 6
SELF_INPUTS = 1           # gun ready
N_INPUTS = SELF_INPUTS + 2 * NEAREST * CONTACT_INPUTS
SENSE_RANGE = FIELD_W / 2

# ---- net: left tread -1..1, right tread -1..1, shoot 0..1 --------------------------
N_HIDDEN_1 = 48
N_HIDDEN_2 = 16
N_OUTPUTS = 3
CMD_LEFT, CMD_RIGHT, CMD_SHOOT = range(N_OUTPUTS)

# ---- evolution: a population deals evenly into battles of TANKS seats ---------------
POP_SIZE = 64
ELITES = 8
DEALS_PER_GEN = 3         # every net fights this many battles, in a fresh random seat each time
CROSSOVER_RATE = 0.25
CROSSOVER_GENE_ODDS = 0.5  # a crossed child takes each weight from the second elite this often
MUTATION_SIGMAS = (0.02, 0.06, 0.2)   # each child draws one: fine-tune, explore, leap

"""Every tuning number lives here, so physics, scoring, evolution and UI agree."""

import math

# ---- field & physics (pixels, seconds) ------------------------------------------
FIELD_W = 720.0
FIELD_H = 720.0
MARGIN = 120.0            # the crash wall sits this far outside the visible field
DT = 1.0 / 60.0           # fixed physics step, identical headless and in the GUI
RUN_SECONDS = 30.0
RUN_STEPS = int(round(RUN_SECONDS / DT))
THRUST = 260.0            # px/s^2 while thrusting
TURN_RATE = 4.0           # rad/s while turning
SHIP_R = 10.0
CRYSTAL_R = 16.0
PICKUP_R = SHIP_R + CRYSTAL_R

# ---- courses ----------------------------------------------------------------------
SPAWN_PAD = 50.0          # crystals spawn at least this far inside the visible edge
MIN_HOP = 220.0           # consecutive crystals (and the first, from the start) are this far apart
COURSE_LEN = 80           # more crystals than anyone can collect in one run

# ---- scoring ----------------------------------------------------------------------
BONUS_FAST = 2.0          # a pickup this quick (s) earns the full +1 speed bonus
BONUS_SLOW = 10.0         # the bonus has faded to zero by this many seconds
CLOSENESS_MAX = 0.5       # partial credit for progress toward the crystal being chased

# ---- controls: action = (turn + 1) * 2 + thrust ------------------------------------
N_ACTIONS = 6

# ---- observation scaling ------------------------------------------------------------
DIST_SCALE = 400.0
SPEED_SCALE = 300.0
N_INPUTS = 6              # sin bearing, cos bearing, distance, v forward, v sideways, closing speed

# ---- net --------------------------------------------------------------------------
N_HIDDEN = 16

# ---- evolution --------------------------------------------------------------------
POP_SIZE = 64
ELITES = 8
COURSES_PER_GEN = 2       # every net flies these many courses; fitness is the mean
CROSSOVER_RATE = 0.25
MUTATION_SIGMAS = (0.02, 0.06, 0.2)   # each child draws one: fine-tune, explore, leap
GENS_PER_PRESS = 500
SAVE_EVERY = 100          # headless evolution writes swarm and weights this often, in generations

# ---- scripted autopilot -------------------------------------------------------------
BOT_ARRIVE_SPEED = 140.0  # speed it aims to carry through a crystal
BOT_BRAKE_SHARE = 0.4     # fraction of thrust budgeted for braking on approach
BOT_MAX_SPEED = 420.0
BOT_AIM_TOLERANCE = 0.35  # rad: thrust only when the nose is this close to the wanted direction
BOT_DEADBAND = 0.06       # rad: stop turning inside this
BOT_MIN_ERROR = 15.0      # px/s: velocity error worth correcting

TAU = 2 * math.pi

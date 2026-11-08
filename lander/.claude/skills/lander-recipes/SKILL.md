---
name: lander-recipes
description: Project-local just recipes for the lander game. Use when asked to record or render a video of the lander flying.
---

# Lander recipes

Run `just setup` first on a new host; the venv is per-hostname.

- `just cc-plummet ROUNDS`: builds the hand-wired suicide-burn weights (`lander.ai.plummet`), measures them,
  and overwrites `weights.npz`. Tune by editing the constants at the top of that module.
- `just cc-flight-stats WEIGHTS.npz ROUNDS`: landing rate plus burn timing and speeds, for comparing pilots.
- `just cc-record COUNT OUT.mp4`: renders COUNT landings to an mp4, scene only, no display needed.
  Flies whatever is in `weights.npz` (the same pilot as Watch), skips rounds that don't land,
  and prints how many rounds it took. Fails if the net lands fewer than 1 in 25 rounds.

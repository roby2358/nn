---
name: tenks-recipes
description: Project helper recipes for tenks (cc- prefixed justfile recipes). Use when you need to check how trained tank nets actually fight, or to see a battle from the headless container.
---

# tenks helper recipes

- `just evolve <gens>` (not cc-, but useful): headless evolution with one log line per generation. Resumes from and
  saves `swarm.npz` + `weights.npz`. Per-tank numbers in the log are against the current population, so they don't
  measure progress on their own.
- `just cc-stats <file.npz> <n>`: the file's nets take green against fresh random blue nets on battles seeded 0..n-1.
  Prints win rates and per-tank kills, friendly kills, near misses and shots. Use the same n to compare two files.
- `just cc-video <file.npz> <seed> <seconds> <out.mp4>`: records one battle at real speed. A swarm file seats its first
  16 nets (8 elites and 8 children) in a shuffled order; a weights file seats its one champion in all 16 tanks.
- `just cc-shot <seconds> <dir>`: writes `shot-watch.png` and `shot-drive.png` of the whole GUI window, headless, for
  checking the layout without a display. It loads `swarm.npz` / `weights.npz` like `just up` does.
- To look at a video from the container, extract frames from it and read the PNGs.
- In a new container, pass an existing venv: `VENV=.venv-<host> just cc-stats ...`.

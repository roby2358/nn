---
name: shards-recipes
description: Project helper recipes for shards (cc- prefixed justfile recipes). Use when you need to check how a trained pilot actually plays, or see the game visually from the headless container.
---

# shards helper recipes

- `just evolve <gens>` (not cc-, but useful): headless evolution with one log line per generation showing the best
  net's hits, survival and accuracy. Resumes from and saves `swarm.npz` + `weights.npz`.
- `just cc-stats <weights.npz> <n>`: flies a weights file on fields seeded 0..n-1 and prints fitness, score, survival,
  hits, waves cleared, accuracy, and how it uses the controls while alive (mean turn, thrust, trigger time). Use the same n to
  compare two weights files. High turn with low accuracy means it's spinning and spraying rather than aiming.
- `just cc-shot <seconds>`: runs the App headless and saves `shot-watch.png` (the champion, that many seconds in)
  and `shot-fly.png` (a held burn with the trigger down, to see the flame). Read the PNGs to check visuals. They are
  gitignored.
- In a new container, pass an existing venv: `VENV=.venv-<host> just cc-stats ...`.

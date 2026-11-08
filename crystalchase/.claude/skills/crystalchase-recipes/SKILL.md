---
name: crystalchase-recipes
description: Project helper recipes for crystalchase (cc- prefixed justfile recipes). Use when you need to see the game visually from the headless container, or check training progress.
---

# crystalchase helper recipes

- `just cc-shot <seconds>`: runs the App headless (SDL dummy driver), steps one generation, then
  saves `shot-watch.png` (a generation replay `<seconds>` in) and `shot-race.png` (a race `<seconds>` past the
  countdown) into the project root. Read the PNGs to check visuals. It uses `weights.npz` if present.
  The PNGs are gitignored.
- `just evolve <gens>` (not cc-, but useful): headless evolution with one log line per generation.
  Resumes from and saves `swarm.npz` + `weights.npz`.
- `just cc-laps <n>`: flies the saved champion (`weights.npz`) solo on courses seeded 0..n-1 and prints
  per-lap crystals, score and crash time/position, then a summary. Use it to check whether the champion
  crashes or to compare champions.
- `just cc-compare <n> <file.npz> ...`: flies the autopilot and each pilot file on the same n courses and
  prints a side-by-side table: crashes, crystals, score, hop-time median/p90/worst, speeds, nose direction
  at pickup, thrust and turn share, and path length per straight-line pixel. Use it to see why one pilot
  beats another.
- `just cc-sector-search`: picks `sector_brain`'s five design constants by evaluation only (grid, then leaders
  on more courses, then the winner on held-out courses next to the autopilot). Prints the leaders; copy the
  chosen values to `SECTOR_DESIGN` in `calculated.py`. `python -m crystalchase.calculated --sector` writes them
  to `weights.npz`.
- `just cc-loop-video <n> <out.mp4>`: films the whole window (chart, stars, saved champion) at 60 fps as a looping
  MP4: ten captures on a nine-crystal ring, from one capture of its first crystal to the next. Rehearses n rings
  and films the one whose two captures match best. Needs `imageio-ffmpeg` (in the dev extras; rerun `just setup`).
- `just cc-swarm-video <seed> <out.mp4>`: films one 30-second training course in the replay view (flight field only), the saved champion
  among random nets with the autopilot alongside. The seed picks the course and the random nets.
- `just cc-lineage <n> <reference.npz>`: did the saved swarm evolve from a reference net? Shows the chart history,
  how far `weights.npz` and the top slots moved from the reference weight by weight, and their scores on n unseen
  courses next to the autopilot.
- `just cc-weights <reference.npz> <evolved.npz>`: pure weight algebra, nothing flown. Reads the sector design
  (arrival speed, range gain, deadband, aim tolerance, min error, left/right separately) out of each net, and lists
  the second-order terms that changed most.
- In a new container, pass an existing venv: `VENV=.venv-<host> just cc-compare ...`.

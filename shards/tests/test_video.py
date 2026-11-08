"""The video recipe writes a playable file of the field."""

import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from shards.calculated import turret  # noqa: E402
from shards.game import App, Round  # noqa: E402
from shards.video import record  # noqa: E402


def test_record_writes_a_video(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = App()
    app.scene = Round("turret", turret(), 0, np.random.default_rng(0))
    out = tmp_path / "clip.mp4"
    record(app, 6, out)
    app.bg.shutdown()
    assert out.stat().st_size > 0
    assert app.scene.world.steps == 6

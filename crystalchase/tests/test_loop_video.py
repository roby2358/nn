import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from crystalchase import art  # noqa: E402
from crystalchase.config import MIN_HOP  # noqa: E402
from crystalchase.course import make_ring_course  # noqa: E402
from crystalchase.loop_video import END_IDX, START_IDX, SPOTS, LoopingStars, seam_px  # noqa: E402


def test_ring_keeps_hop_spacing_all_the_way_around():
    for seed in range(50):
        ring = make_ring_course(seed, SPOTS).crystals
        assert len(ring) == SPOTS
        assert np.all(np.hypot(*(ring - np.roll(ring, 1, axis=0)).T) >= MIN_HOP)


def test_film_spans_ten_captures_returning_to_the_same_crystal():
    assert (START_IDX - 1) % SPOTS == (END_IDX - 1) % SPOTS
    assert END_IDX - START_IDX + 1 == 10


def test_stars_at_the_period_match_the_opening_frame():
    pygame.display.init()
    size = (200, 120)
    stars = art.Starfield(np.random.default_rng(5), size)
    looping = LoopingStars(stars, 20.0, 2.0, size)
    opening, closing = pygame.Surface(size), pygame.Surface(size)
    stars.draw(opening, 0.0)
    looping.draw(closing, 20.0)
    assert np.array_equal(pygame.surfarray.array3d(opening), pygame.surfarray.array3d(closing))


def test_seam_counts_position_heading_and_velocity():
    zero = np.zeros(2)
    assert seam_px((zero, zero, 0.1), (zero, zero, 0.1 + 2 * np.pi)) < 1e-9
    assert seam_px((zero, zero, 0.0), (np.array([3.0, 4.0]), zero, 0.0)) == 5.0

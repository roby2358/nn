import pytest

from crystalchase.calculated import SECTOR_DESIGN, sector_brain
from crystalchase.config import DIST_SCALE, SPEED_SCALE
from crystalchase.weight_anatomy import anatomy


def test_reads_the_five_design_constants_back_out_of_the_weights():
    arrival, gain, deadband, tolerance, min_error = SECTOR_DESIGN
    a = anatomy(sector_brain(*SECTOR_DESIGN).genomes[0])
    assert a.forward["cosb"] * SPEED_SCALE == pytest.approx(arrival)
    assert a.sideways["sinb"] * SPEED_SCALE == pytest.approx(arrival)
    assert a.forward["cosb*dist"] * SPEED_SCALE / DIST_SCALE == pytest.approx(gain, rel=0.01)
    assert a.forward["vf"] == pytest.approx(-1.0)
    assert a.deadband == pytest.approx((deadband, deadband))
    assert a.aim_tolerance == pytest.approx((tolerance, tolerance))
    assert a.min_error == pytest.approx(min_error)
    assert a.outside_share < 1e-12

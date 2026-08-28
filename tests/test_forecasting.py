import pytest
import math
from xp_model import GammaPoissonFilter, POINTS_GOAL, POINTS_CLEAN_SHEET

def test_gamma_poisson_filter():
    gpf = GammaPoissonFilter(half_life=5.0, prior_weight=1.0)
    assert gpf.prior_weight == 1.0
    assert gpf.pos_priors[1]["saves"] == 2.5
    assert gpf.pos_priors[2]["cbit"] == 7.45
    assert gpf.pos_priors[3]["cbirt"] == 7.86

def test_points_mapping():
    assert POINTS_GOAL[1] == 10
    assert POINTS_GOAL[2] == 6
    assert POINTS_CLEAN_SHEET[1] == 4

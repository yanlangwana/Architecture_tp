import math

import pytest

from fraudlake.features import build_model_features, is_night, priority_for
from serving.schemas import EXAMPLE


@pytest.mark.parametrize("hour,expected", [(0, 1), (3, 1), (4, 0), (12, 0), (21, 0), (22, 1), (23, 1)])
def test_night_window_matches_silver_definition(hour, expected):
    assert is_night(hour) == expected


def test_build_model_features_reproduces_silver_formulas():
    f = build_model_features(EXAMPLE)
    assert f["amt_log"] == pytest.approx(math.log1p(962.4), abs=1e-5)
    assert f["city_pop_log"] == pytest.approx(math.log1p(2380), abs=1e-5)
    assert f["is_night"] == 1 and f["is_weekend"] == 1 and f["is_online"] == 1
    assert f["category"] == "shopping_net" and f["gender"] == "F"


def test_build_model_features_normalises_text():
    f = build_model_features({**EXAMPLE, "category": " Grocery_POS ", "gender": "m", "weekday_num": 3})
    assert f["category"] == "grocery_pos" and f["gender"] == "M"
    assert f["is_online"] == 0 and f["is_weekend"] == 0


@pytest.mark.parametrize("score,expected", [(0.99, "critique"), (0.92, "élevée"), (0.6, "à surveiller"), (0.1, "faible")])
def test_priority_levels(score, expected):
    assert priority_for(score, threshold=0.9) == expected

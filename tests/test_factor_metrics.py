"""Compound arithmetic and initial-NAV regression tests for factor metrics."""

import numpy as np
import pandas as pd
import pytest

from core.apps.factor.services import return_growth


@pytest.mark.parametrize("values,kind,growth,drawdown", [
    ([-0.1, 0, 0.02], "simple", 0.918, 0.1),
    ([np.log(0.9), 0, np.log(1.02)], "log", 0.918, 0.1),
    ([-1, 0, 0.02], "simple", 0, 1),
    ([0.1, 0.2, -0.1], "simple", 1.188, 0.1),
    ([-1.1, 0, 0], "simple", -0.1, 1.1),
])
def test_return_growth_matches_browser_metrics_and_chart_sql(values, kind, growth, drawdown):
    actual = return_growth(pd.Series(values), kind)
    assert actual[0] == pytest.approx(growth)
    assert actual[1] == pytest.approx(drawdown)

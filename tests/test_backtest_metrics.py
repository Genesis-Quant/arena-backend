"""Core result metrics must agree with Runtime and the browser on boundaries."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core.apps.backtest.services import backtest_metrics
from core.utils.dsl_source import BacktestApplicationRequest


def metrics(returns, risk_free_rate=0):
    return backtest_metrics(
        {"annual_trading_days": 252, "risk_free_rate": risk_free_rate},
        pd.DataFrame({"tradeDate": pd.date_range("2025-01-01", periods=len(returns)), "ratio": returns}),
    )


def test_two_rows_do_not_report_zero_volatility_from_one_observation():
    assert metrics([0.01, 0.02])["volatility"] is None
    assert metrics([0.01, 0.02])["sharpe"] is None
    assert metrics([0.01, 0.02, 0.02])["volatility"] == 0


@pytest.mark.parametrize("risk_free_rate", [-0.04, 0, 0.04])
def test_sortino_includes_negative_risk_free_rates(risk_free_rate):
    values = np.array([-0.02, 0.01, 0.03])
    excess = values - ((1 + risk_free_rate) ** (1 / 252) - 1)
    expected = excess.mean() / np.sqrt(np.square(np.minimum(excess, 0)).mean()) * np.sqrt(252)
    assert metrics(values, risk_free_rate)["sortino"] == pytest.approx(expected)


@pytest.mark.parametrize("risk_free_rate", [-1, -1.1])
def test_api_risk_free_rate_rejects_invalid_compounding_domain(risk_free_rate):
    # This field is shared by normal, research and optimization application requests.
    from pydantic import TypeAdapter
    field = BacktestApplicationRequest.model_fields["risk_free_rate"]
    with pytest.raises(ValidationError):
        TypeAdapter(field.rebuild_annotation()).validate_python(risk_free_rate)


def test_high_positive_return_is_not_a_price():
    assert metrics([1.2, 0.1, 0.2])["totalReturn"] == pytest.approx(1.904)

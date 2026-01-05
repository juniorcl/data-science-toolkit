import pytest
import numpy as np
import pandas as pd
from dstoolkit.feature.monitoring import ks_test_drift


def test_ks_no_drift_same_distribution():
    rng = np.random.default_rng(42)

    expected = rng.normal(loc=0, scale=1, size=1000)
    actual = rng.normal(loc=0, scale=1, size=1000)

    result = ks_test_drift(expected, actual)

    assert result["drift_detected"] is False
    assert result["p_value"] > 0.05

def test_ks_detects_drift_different_distribution():
    rng = np.random.default_rng(42)

    expected = rng.normal(loc=0, scale=1, size=1000)
    actual = rng.normal(loc=2, scale=1, size=1000)

    result = ks_test_drift(expected, actual)

    assert result["drift_detected"] is True
    assert result["p_value"] < 0.05

def test_ks_accepts_array_like_input():
    expected = [1, 2, 3, 4, 5] * 20
    actual = np.array([1, 2, 3, 4, 6] * 20)

    result = ks_test_drift(expected, actual)

    assert isinstance(result, dict)
    assert "ks_statistic" in result

def test_ks_ignores_nan_values():
    expected = pd.Series([1, 2, 3, np.nan, 4, 5] * 20)
    actual = pd.Series([1, 2, 2, 3, np.nan, 5] * 20)

    result = ks_test_drift(expected, actual)

    assert isinstance(result["p_value"], float)
    assert 0.0 <= result["p_value"] <= 1.0

def test_ks_raises_error_if_expected_empty():
    expected = pd.Series([np.nan, np.nan])
    actual = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        ks_test_drift(expected, actual)

def test_ks_raises_error_if_actual_empty():
    expected = pd.Series([1, 2, 3])
    actual = pd.Series([np.nan, np.nan])

    with pytest.raises(ValueError):
        ks_test_drift(expected, actual)

def test_ks_raises_type_error_for_non_numeric_expected():
    expected = pd.Series(["A", "B", "C"])
    actual = pd.Series([1, 2, 3])

    with pytest.raises(TypeError):
        ks_test_drift(expected, actual)

def test_ks_raises_type_error_for_non_numeric_actual():
    expected = pd.Series([1, 2, 3])
    actual = pd.Series(["A", "B", "C"])

    with pytest.raises(TypeError):
        ks_test_drift(expected, actual)

def test_ks_return_structure():
    expected = np.random.normal(size=100)
    actual = np.random.normal(size=100)

    result = ks_test_drift(expected, actual)

    assert set(result.keys()) == {
        "ks_statistic",
        "p_value",
        "drift_detected"
    }

    assert isinstance(result["ks_statistic"], float)
    assert isinstance(result["p_value"], float)
    assert isinstance(result["drift_detected"], bool)

def test_ks_alpha_threshold_effect():
    rng = np.random.default_rng(123)

    expected = rng.normal(0, 1, 500)
    actual = rng.normal(0.3, 1, 500)

    result_strict = ks_test_drift(expected, actual, alpha=0.01)
    result_loose = ks_test_drift(expected, actual, alpha=0.10)

    assert result_loose["drift_detected"] >= result_strict["drift_detected"]
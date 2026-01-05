import pandas as pd
from dstoolkit.feature.monitoring import chi_squared_monitoring


def test_chi_squared_no_drift():
    expected = pd.Series(["A", "B", "A", "B", "C"] * 20)
    actual = pd.Series(["A", "B", "A", "B", "C"] * 20)

    result = chi_squared_monitoring(expected, actual)

    assert result["drift_detected"] is False
    assert result["p_value"] > 0.05

def test_chi_squared_detects_drift():
    expected = pd.Series(["A"] * 80 + ["B"] * 20)
    actual = pd.Series(["A"] * 20 + ["B"] * 80)

    result = chi_squared_monitoring(expected, actual)

    assert result["drift_detected"] is True
    assert result["p_value"] < 0.05

def test_chi_squared_accepts_list_input():
    expected = ["A", "B", "A", "C", "B"] * 10
    actual = ["A", "A", "A", "B", "C"] * 10

    result = chi_squared_monitoring(expected, actual)

    assert isinstance(result, dict)
    assert "chi2" in result

def test_chi_squared_ignores_nan():
    expected = pd.Series(["A", "B", None, "A", "C"] * 20)
    actual = pd.Series(["A", "B", "B", None, "C"] * 20)

    result = chi_squared_monitoring(expected, actual)

    assert isinstance(result["p_value"], float)

def test_chi_squared_invalid_due_to_low_expected_frequency():
    expected = pd.Series(["A"] * 3 + ["B"] * 2)
    actual = pd.Series(["A"] * 4 + ["B"] * 1)

    result = chi_squared_monitoring(
        expected,
        actual,
        min_expected_freq=5
    )

    assert result["drift_detected"] is False

def test_chi_squared_return_structure():
    expected = pd.Series(["A", "B", "C"] * 10)
    actual = pd.Series(["A", "B", "C"] * 10)

    result = chi_squared_monitoring(expected, actual)

    assert set(result.keys()) == {"chi2", "p_value", "drift_detected"}
    assert isinstance(result["chi2"], float)
    assert isinstance(result["drift_detected"], bool)
import pytest
import numpy as np
import pandas as pd

from dstoolkit.feature.monitoring import jensen_shannon_divergence


def test_js_numeric_same_distribution():
    rng = np.random.default_rng(42)

    expected = rng.normal(0, 1, 1000)
    actual = rng.normal(0, 1, 1000)

    result = jensen_shannon_divergence(expected, actual, bins=10)

    assert result["js_divergence"] < 0.05
    assert result["js_distance"] < 0.25

def test_js_numeric_different_distribution():
    rng = np.random.default_rng(42)

    expected = rng.normal(0, 1, 1000)
    actual = rng.normal(3, 1, 1000)

    result = jensen_shannon_divergence(expected, actual, bins=10)

    assert result["js_divergence"] > 0.10
    assert result["js_distance"] > 0.30

def test_js_numeric_uniform_strategy():
    rng = np.random.default_rng(123)

    expected = rng.normal(0, 1, 1000)
    actual = rng.normal(1, 1, 1000)

    result = jensen_shannon_divergence(
        expected,
        actual,
        bins=15,
        strategy="uniform"
    )

    assert isinstance(result["js_divergence"], float)
    assert 0.0 <= result["js_divergence"] <= 1.0

def test_js_categorical_same_distribution():
    expected = pd.Series(["A", "B", "C"] * 100)
    actual = pd.Series(["A", "B", "C"] * 100)

    result = jensen_shannon_divergence(expected, actual, bins=5)

    assert result["js_divergence"] < 0.01

def test_js_categorical_different_distribution():
    expected = pd.Series(["A"] * 80 + ["B"] * 20)
    actual = pd.Series(["A"] * 20 + ["B"] * 80)

    result = jensen_shannon_divergence(expected, actual, bins=5)

    assert result["js_divergence"] > 0.10

def test_js_accepts_array_and_ignores_nan():
    expected = [1, 2, 3, np.nan, 4, 5] * 50
    actual = np.array([1, 2, np.nan, 3, 6, 7] * 50)

    result = jensen_shannon_divergence(expected, actual, bins=5)

    assert isinstance(result["js_distance"], float)

def test_js_raises_error_if_expected_empty():
    expected = pd.Series([np.nan, np.nan])
    actual = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        jensen_shannon_divergence(expected, actual, bins=5)

def test_js_raises_error_if_actual_empty():
    expected = pd.Series([1, 2, 3])
    actual = pd.Series([np.nan, np.nan])

    with pytest.raises(ValueError):
        jensen_shannon_divergence(expected, actual, bins=5)

def test_js_raises_error_for_invalid_bins_numeric():
    expected = np.random.normal(size=100)
    actual = np.random.normal(size=100)

    with pytest.raises(ValueError):
        jensen_shannon_divergence(expected, actual, bins=1)

def test_js_raises_error_for_invalid_strategy():
    expected = np.random.normal(size=100)
    actual = np.random.normal(size=100)

    with pytest.raises(ValueError):
        jensen_shannon_divergence(
            expected,
            actual,
            bins=10,
            strategy="invalid"
        )

def test_js_return_structure():
    expected = np.random.normal(size=200)
    actual = np.random.normal(size=200)

    result = jensen_shannon_divergence(expected, actual, bins=10)

    assert set(result.keys()) == {"js_divergence", "js_distance"}
    assert isinstance(result["js_divergence"], float)
    assert isinstance(result["js_distance"], float)

def test_js_symmetry_property():
    rng = np.random.default_rng(7)

    a = rng.normal(0, 1, 500)
    b = rng.normal(1, 1, 500)

    result_ab = jensen_shannon_divergence(a, b, bins=10)
    result_ba = jensen_shannon_divergence(b, a, bins=10)

    assert np.isclose(
        result_ab["js_divergence"],
        result_ba["js_divergence"],
        atol=1e-6
    )
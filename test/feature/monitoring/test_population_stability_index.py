import pytest
import numpy as np
import pandas as pd

from dstoolkit.feature.monitoring import psi


def test_psi_numeric_same_distribution_low_value():
    rng = np.random.default_rng(42)

    expected = rng.normal(0, 1, 1000)
    actual = rng.normal(0, 1, 1000)

    result = psi(expected, actual, bins=10)

    assert result < 0.10


def test_psi_numeric_different_distribution_high_value():
    rng = np.random.default_rng(42)

    expected = rng.normal(0, 1, 1000)
    actual = rng.normal(3, 1, 1000)

    result = psi(expected, actual, bins=10)

    assert result > 0.25


def test_psi_uniform_strategy_works():
    rng = np.random.default_rng(123)

    expected = rng.normal(0, 1, 1000)
    actual = rng.normal(1, 1, 1000)

    result = psi(expected, actual, bins=15, strategy="uniform")

    assert isinstance(result, float)
    assert result >= 0.0


def test_psi_accepts_array_like_and_ignores_nan():
    expected = [1, 2, 3, np.nan, 4, 5] * 50
    actual = np.array([1, 2, np.nan, 3, 6, 7] * 50)

    result = psi(expected, actual, bins=5)

    assert isinstance(result, float)


def test_psi_raises_error_if_expected_empty():
    expected = pd.Series([np.nan, np.nan])
    actual = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        psi(expected, actual, bins=5)


def test_psi_raises_error_if_actual_empty():
    expected = pd.Series([1, 2, 3])
    actual = pd.Series([np.nan, np.nan])

    with pytest.raises(ValueError):
        psi(expected, actual, bins=5)


def test_psi_raises_error_for_invalid_bins():
    expected = np.random.normal(size=100)
    actual = np.random.normal(size=100)

    with pytest.raises(ValueError):
        psi(expected, actual, bins=1)


def test_psi_raises_error_for_invalid_strategy():
    expected = np.random.normal(size=100)
    actual = np.random.normal(size=100)

    with pytest.raises(ValueError):
        psi(expected, actual, bins=10, strategy="invalid")


def test_psi_returns_zero_for_identical_constant_distributions():
    expected = np.ones(1000)
    actual = np.ones(1000)

    result = psi(expected, actual, bins=5)

    assert np.isclose(result, 0.0, atol=1e-6)


def test_psi_monotonicity_larger_shift_increases_psi():
    rng = np.random.default_rng(99)

    expected = rng.normal(0, 1, 2000)
    actual_small_shift = rng.normal(0.5, 1, 2000)
    actual_large_shift = rng.normal(3.0, 1, 2000)

    psi_small = psi(expected, actual_small_shift, bins=10)
    psi_large = psi(expected, actual_large_shift, bins=10)

    assert psi_large > psi_small
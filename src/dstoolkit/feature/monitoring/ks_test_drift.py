import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


def ks_test_drift(expected, actual, alpha=0.05):
    """
    Perform Kolmogorov-Smirnov test to detect distribution drift.

    Parameters
    ----------
    expected : pd.Series or array-like
        Reference distribution (e.g., training data).
    actual : pd.Series or array-like
        Current distribution (e.g., production data).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        Dictionary with:
        - ks_statistic: KS statistic value
        - p_value: p-value of the test
        - drift_detected: whether drift is detected at given alpha
    """

    if not isinstance(expected, pd.Series):
        expected = pd.Series(expected)

    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual)

    expected = expected.dropna()
    actual = actual.dropna()

    if expected.empty or actual.empty:
        raise ValueError("Expected and actual distributions must not be empty.")

    if not np.issubdtype(expected.dtype, np.number) or not np.issubdtype(actual.dtype, np.number):
        raise TypeError("KS test is only valid for numerical variables.")

    ks_stat, p_value = ks_2samp(
        expected,
        actual,
        alternative="two-sided",
        method="auto"
    )

    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < alpha)
    }
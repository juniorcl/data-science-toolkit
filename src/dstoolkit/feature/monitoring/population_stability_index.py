import numpy as np
import pandas as pd


def psi(expected, actual, bins=10, strategy="quantile", eps=1e-6):
    """
    Calculate the Population Stability Index (PSI) between two distributions.

    Parameters
    ----------
    expected : pd.Series or array-like
        Reference distribution (e.g., training data).
    actual : pd.Series or array-like
        Current distribution (e.g., production data).
    bins : int, default=10
        Number of bins to split the data.
    strategy : {"quantile", "uniform"}, default="quantile"
        - "quantile": bins based on expected quantiles (recommended).
        - "uniform": equal-width bins.
    eps : float, default=1e-6
        Small value to avoid division by zero or log(0).

    Notes
    -----
    PSI < 0.10 → estability
    0.10–0.25  → atention
    > 0.25     → drift

    Returns
    -------
    dict
        Dictionary with:
        - psi: total PSI value
        - psi_max_bin: max PSI contribution among bins
        - psi_mean_bin: mean PSI contribution
    """
    if not isinstance(expected, pd.Series):
        expected = pd.Series(expected)

    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual)

    expected = expected.dropna()
    actual = actual.dropna()

    if expected.empty or actual.empty:
        raise ValueError("Expected and actual distributions must not be empty.")

    if bins < 2:
        raise ValueError("Number of bins must be >= 2.")

    # Define bin edges
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, bins + 1)
        breakpoints = np.unique(expected.quantile(quantiles).values)
    elif strategy == "uniform":
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        breakpoints = np.linspace(min_val, max_val, bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'.")

    # Bin counts
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions
    expected_perc = expected_counts / expected_counts.sum()
    actual_perc = actual_counts / actual_counts.sum()

    # PSI per bin
    psi_bins = (actual_perc - expected_perc) * np.log((actual_perc + eps) / (expected_perc + eps))

    return float(np.sum(psi_bins))
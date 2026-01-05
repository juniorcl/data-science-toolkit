import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def jensen_shannon_divergence(expected, actual, bins, strategy="quantile", eps=1e-12):
    """
    Compute Jensen-Shannon Divergence (JSD) between two distributions
    using shared binning to ensure symmetry.

    Parameters
    ----------
    expected : pd.Series or array-like
        Reference distribution (e.g., training data).
    actual : pd.Series or array-like
        Current distribution (e.g., production data).
    bins : int
        Number of bins (used for numerical variables).
    strategy : {"quantile", "uniform"}, default="quantile"
        Binning strategy for numerical variables.
    eps : float, default=1e-12
        Small value to avoid zero probabilities.

    Returns
    -------
    dict
        Dictionary with:
        - js_divergence: Jensen-Shannon divergence (0 to 1)
        - js_distance: Jensen-Shannon distance (sqrt of divergence)
    """

    # Convert to Series
    if not isinstance(expected, pd.Series):
        expected = pd.Series(expected)

    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual)

    # Drop NaNs
    expected = expected.dropna()
    actual = actual.dropna()

    if expected.empty or actual.empty:
        raise ValueError("Expected and actual distributions must not be empty.")

    # Detect categorical vs numerical
    is_categorical = (
        expected.dtype == "object"
        or actual.dtype == "object"
        or expected.nunique() < bins
    )

    # =========================
    # CATEGORICAL
    # =========================
    if is_categorical:
        categories = sorted(set(expected.unique()) | set(actual.unique()))

        p = expected.value_counts(normalize=True).reindex(categories, fill_value=0).values
        q = actual.value_counts(normalize=True).reindex(categories, fill_value=0).values

    # =========================
    # NUMERICAL (shared bins)
    # =========================
    else:
        if bins < 2:
            raise ValueError("Number of bins must be >= 2.")

        combined = pd.concat([expected, actual])

        if strategy == "quantile":
            quantiles = np.linspace(0, 1, bins + 1)
            breakpoints = np.unique(combined.quantile(quantiles).values)

        elif strategy == "uniform":
            min_val = combined.min()
            max_val = combined.max()
            breakpoints = np.linspace(min_val, max_val, bins + 1)

        else:
            raise ValueError("strategy must be 'quantile' or 'uniform'.")

        p = np.histogram(expected, bins=breakpoints)[0].astype(float)
        q = np.histogram(actual, bins=breakpoints)[0].astype(float)

        p = p / p.sum()
        q = q / q.sum()

    # Avoid zeros
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    js_distance = jensenshannon(p, q)
    js_divergence = js_distance ** 2

    return {
        "js_divergence": float(js_divergence),
        "js_distance": float(js_distance),
    }
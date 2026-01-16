import numpy as np
from sklearn.metrics import roc_curve


def ks_score(y_true, y_scores):
    """
    Calculate the Kolmogorov-Smirnov (KS) score between positive and negative classes.

    This score is useful for evaluating the performance of binary classifiers.

    Parameters
    ----------
    y_true : pd.Series or arrray-like
        True binary labels (0 or 1).
    y_prob : pd.Series or array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        The KS score, which ranges from 0 to 1.
    """

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    
    if y_scores.ndim != 1:
        raise ValueError("y_scores must be a 1D array.")
    
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("y_true and y_scores must have the same length.")

    unique_values = np.unique(y_true)
    if set(unique_values) - {0, 1}:
        raise ValueError("y_true must be binary (0/1).")

    if len(unique_values) < 2:
        raise ValueError("y_true must contain both positive (1) and negative (0) samples.")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ks_statistic = max(tpr - fpr)
    return ks_statistic
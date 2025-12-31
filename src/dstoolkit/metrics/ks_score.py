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
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ks_statistic = max(tpr - fpr)
    return ks_statistic
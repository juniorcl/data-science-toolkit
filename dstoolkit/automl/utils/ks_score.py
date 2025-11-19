from scipy.stats import ks_2samp

def ks_score(y_true, y_prob):
    """
    Calculate the Kolmogorov-Smirnov (KS) score between positive and negative classes.

    This score is useful for evaluating the performance of binary classifiers.

    Parameters
    ----------
    y_true : pd.Series
        True binary labels (0 or 1).
    y_prob : pd.Series
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        The KS score, which ranges from 0 to 1.
    """
    pos_prob = y_prob[y_true == 1]
    neg_prob = y_prob[y_true == 0]
    if len(pos_prob) == 0 or len(neg_prob) == 0:
        return 0.0
    result = ks_2samp(pos_prob, neg_prob)
    return result.statistic
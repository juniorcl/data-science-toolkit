import numpy as np
from sklearn.metrics import average_precision_score


def average_precision_lift_score(y_true, y_score, baseline=None):
    """
    Compute the relative lift of Average Precision (AP) over a random baseline.

    The random baseline for Precision–Recall is defined as the prevalence of
    the positive class (i.e., the fraction of positive samples). The lift
    expresses how much the model improves over this baseline in relative terms.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels. Positive class must be encoded as 1 and negative
        class as 0.

    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive
        class or confidence values.

    baseline : float, optional (default=None)
        Baseline Average Precision value. If None, the baseline is computed
        as the prevalence of the positive class in ``y_true``.

    Returns
    -------
    lift : float
        Relative lift of Average Precision over the baseline, expressed as
        a fraction. For example, a value of 1.0 means a 100% improvement
        over the baseline, while 0.0 means no improvement.

    Notes
    -----
    - A lift value greater than 0 indicates performance above the random
      baseline.
    - Negative values indicate performance worse than random.
    - When the baseline (prevalence) is very small, lift values can become
      large and should be interpreted with care.

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 0, 1, 1])
    >>> y_score = np.array([0.1, 0.4, 0.8, 0.2, 0.9, 0.7])
    >>> average_precision_lift(y_true, y_score)
    1.25
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_score.ndim != 1:
        raise ValueError("y_score must be a 1D array.")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    ap = average_precision_score(y_true, y_score)

    if baseline is None:
        baseline = np.mean(y_true)

    if baseline <= 0:
        raise ValueError("Baseline must be greater than 0.")

    return (ap - baseline) / baseline
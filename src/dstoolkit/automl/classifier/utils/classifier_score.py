from dstoolkit.metrics.scores import average_precision_lift_scorer, ks_scorer


def get_classifier_score(scoring):
    """
    Retrieve the appropriate scorer for classifier evaluation.

    This function maps a scoring metric name to its corresponding scorer.

    Parameters
    ----------
    scoring : str
        The name of the scoring metric to retrieve.

    Returns
    -------
    str or callable
        The scorer corresponding to the specified metric.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """
    scorers = {
        "ks": ks_scorer,
        "roc_auc": "roc_auc",
        "brier": "neg_brier_score",
        "log_loss": "neg_log_loss",
        "avg_precision_lift": average_precision_lift_scorer,
    }

    if scoring not in scorers:
        raise ValueError(f"Metric '{scoring}' is not supported.")

    return scorers[scoring]

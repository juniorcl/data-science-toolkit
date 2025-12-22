from .ks_score import ks_score
from .ks_scorer import ks_scorer
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


def get_classifier_function_score(scoring):
    """
    Retrieve the appropriate scoring function for classifier evaluation.

    This function maps a scoring metric name to its corresponding function.

    Parameters
    ----------
    scoring : str
        The name of the scoring metric to retrieve.

    Returns
    -------
    Callable
        The scoring function corresponding to the specified metric.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """
    functions = {
        'ks': ks_score,
        'roc_auc': roc_auc_score,
        'brier': lambda y_true, y_pred: -1 * brier_score_loss(y_true, y_pred),
        'log_loss': lambda y_true, y_pred: -1 * log_loss(y_true, y_pred)
    }

    if scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    
    return functions[scoring]
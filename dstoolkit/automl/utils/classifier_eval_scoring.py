from .ks_score import ks_score
from .ks_scorer import ks_scorer
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


def get_classifier_eval_scoring(scoring, return_func=True):
    """
    Retrieve the appropriate scoring function or scorer for classifier evaluation.

    This function maps a scoring metric name to its corresponding function or scorer.

    Parameters
    ----------
    scoring : str
        The name of the scoring metric to retrieve.
    return_func : bool, optional
        Whether to return the scoring function (True) or the scorer (False), by default True.

    Returns
    -------
    Union[Callable, str]
        The scoring function or scorer corresponding to the specified metric.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """
    scorers = {
        'ks': ks_scorer,
        'roc_auc': 'roc_auc',
        'brier': 'neg_brier_score',
        'log_loss': 'neg_log_loss',
    }

    functions = {
        'ks': ks_score,
        'roc_auc': roc_auc_score,
        'brier': lambda y_true, y_pred: -1 * brier_score_loss(y_true, y_pred),
        'log_loss': lambda y_true, y_pred: -1 * log_loss(y_true, y_pred)
    }

    if scoring not in scorers or scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    
    return functions[scoring] if return_func else scorers[scoring]
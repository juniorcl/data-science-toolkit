from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)


def get_regressor_eval_scoring(scoring, return_func=True):
    """
    Retrieve the scoring string or function for a given regression metric.

    This function maps a regression metric name to its corresponding scoring string
    or function, allowing for flexible evaluation of regression models.

    Parameters
    ----------
    scoring : str
        The name of the regression metric. Supported metrics are:
        - 'r2'
        - 'explained_variance'
        - 'mean_absolute_error'
        - 'median_absolute_error'
        - 'root_mean_squared_error'
        - 'mean_absolute_percentage_error'
    return_func : bool, optional
        If True, return the metric function; if False, return the scoring string. 
        Default is True.

    Returns
    -------
    str or function
        Depending on `return_func`, either the scoring string or the metric function.

    Raises
    ------
    ValueError
        If `scoring` is not a supported metric.
    """
    scorers = {
        'r2': 'r2',
        'explained_variance': 'explained_variance',
        'mean_absolute_error': 'neg_mean_absolute_error',
        'median_absolute_error': 'neg_median_absolute_error',
        'root_mean_squared_error': 'neg_root_mean_squared_error',
        'mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error'
    }

    functions = {
        'r2': r2_score,
        'explained_variance': explained_variance_score,
        'mean_absolute_error': lambda y_true, y_pred: -1*mean_absolute_error(y_true, y_pred),
        'median_absolute_error': lambda y_true, y_pred: -1*median_absolute_error(y_true, y_pred),
        'root_mean_squared_error': lambda y_true, y_pred: -1*root_mean_squared_error(y_true, y_pred),
        'mean_absolute_percentage_error': lambda y_true, y_pred: -1*mean_absolute_percentage_error(y_true, y_pred)
    }

    if scoring not in scorers or scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    
    return functions[scoring] if return_func else scorers[scoring]
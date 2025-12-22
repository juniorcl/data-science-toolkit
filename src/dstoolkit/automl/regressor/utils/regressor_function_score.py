from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)


def get_regressor_function_score(scoring):
    """
    Retrieve the appropriate scoring function for regressor evaluation.

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
        'r2': r2_score,
        'explained_variance': explained_variance_score,
        'mean_absolute_error': lambda y_true, y_pred: -1*mean_absolute_error(y_true, y_pred),
        'median_absolute_error': lambda y_true, y_pred: -1*median_absolute_error(y_true, y_pred),
        'root_mean_squared_error': lambda y_true, y_pred: -1*root_mean_squared_error(y_true, y_pred),
        'mean_absolute_percentage_error': lambda y_true, y_pred: -1*mean_absolute_percentage_error(y_true, y_pred)
    }

    if scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    
    return functions[scoring]
def get_regressor_score(scoring):
    """
    Retrieve the appropriate scoring function for regressor evaluation.

    This function maps a scoring metric name to its corresponding function.

    Parameters
    ----------
    scoring : str
        The name of the scoring metric to retrieve.

    Returns
    -------
    string
        The scoring string corresponding to the specified metric.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """
    scorers = {
        'r2': 'r2',
        'explained_variance': 'explained_variance',
        'mean_absolute_error': 'neg_mean_absolute_error',
        'median_absolute_error': 'neg_median_absolute_error',
        'root_mean_squared_error': 'neg_root_mean_squared_error',
        'mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error'
    }

    if scoring not in scorers:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    
    return scorers[scoring]
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def get_cluster_eval_scoring(scoring, return_func=True):
    """
    Retrieve clustering evaluation metric by name.

    This function maps a scoring metric name to its corresponding function or scorer.

    Parameters
    ----------
    scoring : str
        The name of the clustering evaluation metric. Supported metrics are:
        'silhouette', 'calinski', and 'davies'.
    return_func : bool, optional
        If True, return the function object; if False, return the string name of the scorer.
        Default is True.
    
    Returns
    -------
    Union[Callable, str]
        The clustering evaluation metric function or scorer name.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """
    scorers = {
        'silhouette': 'silhouette_score',
        'calinski': 'calinski_harabasz_score',
        'davies': 'davies_bouldin_score'
    }

    functions = {
        'silhouette': silhouette_score,
        'calinski': calinski_harabasz_score,
        'davies': davies_bouldin_score
    }

    if scoring not in scorers or scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    
    return functions[scoring] if return_func else scorers[scoring]
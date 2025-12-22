from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)


def get_cluster_function_score(scoring):
    """
    Retrieve clustering evaluation metric by name.

    This function maps a scoring metric name to its corresponding function or scorer.

    Parameters
    ----------
    scoring : str
        The name of the clustering evaluation metric. Supported metrics are:
        'silhouette', 'calinski', and 'davies'.
    
    Returns
    -------
    str
        The clustering evaluation metric function or scorer name.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """
    scorers = {
        'silhouette': silhouette_score,
        'calinski': calinski_harabasz_score,
        'davies': davies_bouldin_score
    }

    if scoring not in scorers:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    
    return scorers[scoring]
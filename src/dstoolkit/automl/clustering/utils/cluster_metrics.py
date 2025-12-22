from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)


def get_cluster_metrics(X, labels):
    """
    Calculate clustering metrics for the given data and labels.

    Return a dictionary containing the computed clustering metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Explanation of the first parameter.
    labels : list or np.ndarray
        Labels assigned to each data point.

    Returns
    -------
    dict
        A dictionary with clustering metric names as keys and their corresponding scores as values.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """
    return {
        'Silhouette': silhouette_score(X, labels),
        'Calinski Harabasz': calinski_harabasz_score(X, labels),
        'Davies-Bouldin': davies_bouldin_score(X, labels)
    }
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def get_cluster_metrics(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    
    return {
        'Silhouetete': silhouette_score(X, y),
        'Davies Bouldin': davies_bouldin_score(X, y),
        'Calinski Harabasz': calinski_harabasz_score(X, y)
    }


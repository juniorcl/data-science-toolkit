from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

def get_cluster_metrics(X, labels):
    return {
        'Silhouette': silhouette_score(X, labels),
        'Calinski Harabasz': calinski_harabasz_score(X, labels),
        'Davies-Bouldin': davies_bouldin_score(X, labels)
    }
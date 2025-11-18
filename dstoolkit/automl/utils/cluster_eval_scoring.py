from sklearn.metrics import (
    silhouette_score,  
    calinski_harabasz_score,
    davies_bouldin_score
)

def get_cluster_eval_scoring(scoring, return_func=True):
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
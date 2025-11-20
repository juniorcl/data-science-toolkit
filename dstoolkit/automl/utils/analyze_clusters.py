from .plot_cluster_pca import plot_cluster_pca
from .plot_cluster_umap import plot_cluster_umap
from .plot_cluster_sizes import plot_cluster_sizes
from .plot_silhouette_analysis import plot_silhouette_analysis
from .plot_kdeplots_by_cluster import plot_kdeplots_by_cluster


def analyze_clusters(X_orig, X, labels):
    """
    Analyze and visualize clustering results using various plots.

    This function orchestrates the various plotting functions to provide a comprehensive
    analysis of the clustering results.

    Parameters
    ----------
    X_orig : pd.DataFrame
        Original dataset before any preprocessing.
    X : pd.DataFrame
        Preprocessed dataset used for clustering.
    labels : array-like
        Cluster labels assigned to each data point.

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the input data is not valid for clustering.
    """
    plot_cluster_sizes(labels)
    plot_silhouette_analysis(X, labels)
    plot_kdeplots_by_cluster(X_orig, labels)
    plot_cluster_pca(X, labels)
    plot_cluster_umap(X, labels)
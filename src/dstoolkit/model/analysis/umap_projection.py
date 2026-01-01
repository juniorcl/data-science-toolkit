import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP


def plot_umap_projection(X, labels, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    """
    Plots a 2D UMAP projection of clustered data points.

    This function reduces the dimensionality of the input data using UMAP
    and visualizes the clusters in a 2D scatter plot.

    Parameters
    ----------
    X : pd.DataFrame
        The input data to be reduced and plotted.
    labels : array-like
        Cluster labels for each data point in X.
    n_neighbors : int, optional
        The size of local neighborhood (in terms of number of neighboring sample points) 
        used for manifold approximation. Default is 15.
    min_dist : float, optional
        The effective minimum distance between embedded points. Default is 0.1.
    metric : str, optional
        The metric to use for distance computation. Default is 'euclidean'.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the input data is invalid or if the UMAP fitting fails.
    """
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, palette="tab10", s=50)
    plt.title("UMAP Projection of Clusters")
    plt.xlabel("Dimension 1 (UMAP)")
    plt.ylabel("Dimension 2 (UMAP)")
    plt.legend(title="Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
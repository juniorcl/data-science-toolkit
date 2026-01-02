import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP


def plot_umap_projection(X, labels, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42, figsize=(7, 6)):
    """
    Plot a 2D UMAP projection of clustered data.

    Parameters
    ----------
    X : pd.DataFrame or array-like of shape (n_samples, n_features)
        Input data to be reduced.

    labels : array-like of shape (n_samples,)
        Cluster labels.

    n_neighbors : int, default=15
        Size of the local neighborhood used for manifold approximation.

    min_dist : float, default=0.1
        Minimum distance between embedded points.

    metric : str, default="euclidean"
        Distance metric.

    random_state : int, default=42
        Random seed for reproducibility.

    figsize : tuple, default=(7, 6)
        Size of the matplotlib figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    X_umap = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=labels,
        palette="tab10",
        s=50,
        ax=ax,
    )

    ax.set_title("UMAP Projection of Clusters")
    ax.set_xlabel("Dimension 1 (UMAP)")
    ax.set_ylabel("Dimension 2 (UMAP)")
    ax.legend(title="Cluster")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig, ax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_silhouette_analysis(X, labels, metric="euclidean", ax=None):
    """
    Plot the silhouette analysis for a clustering result.

    This function computes and visualizes the silhouette scores for each sample,
    allowing an assessment of cluster cohesion and separation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data used for clustering.

    labels : array-like of shape (n_samples,)
        Cluster labels for each sample.

    metric : str, default="euclidean"
        Distance metric used to compute silhouette scores.

    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    silhouette_avg = silhouette_score(X, labels, metric=metric)
    sample_silhouette_values = silhouette_samples(X, labels, metric=metric)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    y_lower = 10
    colors = sns.color_palette("tab10", n_clusters)

    for i, c in enumerate(unique_labels):
        cluster_silhouette_vals = sample_silhouette_values[labels == c]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i % len(colors)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(
            -0.05,
            y_lower + 0.5 * size_cluster_i,
            str(c),
        )

        y_lower = y_upper + 10

    ax.axvline(
        x=silhouette_avg,
        linestyle="--",
        label=f"Average = {silhouette_avg:.3f}",
    )

    ax.set_title("Silhouette Analysis")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    return fig, ax
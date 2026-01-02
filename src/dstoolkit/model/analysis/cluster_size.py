import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cluster_sizes(labels, ax=None):
    """
    Plot the sizes of clusters based on the provided labels.

    Each bar represents the number of samples assigned to a cluster.

    Parameters
    ----------
    labels : array-like
        Cluster labels for each sample.

    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.

    Raises
    ------
    ValueError
        If labels are empty or invalid.
    """
    if labels is None or len(labels) == 0:
        raise ValueError("labels must be a non-empty array-like.")

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    sns.barplot(
        x=cluster_counts.index,
        y=cluster_counts.values,
        ax=ax,
        palette="Blues_d",
    )

    ax.set_title("Cluster Sizes")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Observations")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    return fig, ax
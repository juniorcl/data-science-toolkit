import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_kdeplots_by_cluster(X, labels, n_cols=3, figsize=(15, 10), fill=False, alpha=0.8):
    """
    Plot KDE distributions of numerical features grouped by cluster labels.

    Parameters
    ----------
    X : pd.DataFrame or array-like of shape (n_samples, n_features)
        Input feature matrix.

    labels : array-like of shape (n_samples,)
        Cluster labels.

    n_cols : int, default=3
        Number of columns in the subplot grid.

    figsize : tuple, default=(15, 10)
        Size of the matplotlib figure.

    fill : bool, default=False
        Whether to fill the KDE area.

    alpha : float, default=0.8
        Transparency level for KDE curves.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    axes : np.ndarray of matplotlib.axes.Axes
        Array of axes corresponding to subplots.

    Raises
    ------
    ValueError
        If X is empty or if labels length does not match X.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if X.empty:
        raise ValueError("X must not be empty.")

    if len(labels) != len(X):
        raise ValueError("labels must have the same length as X.")

    X_plot = X.copy()
    X_plot["cluster"] = labels

    X_num = X_plot.select_dtypes(include=[np.number]).drop(
        columns=["cluster"], errors="ignore"
    )

    n_features = X_num.shape[1]
    if n_features == 0:
        raise ValueError("X must contain at least one numerical feature.")

    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, X_num.columns):
        sns.kdeplot(
            data=X_plot,
            x=col,
            hue="cluster",
            palette="tab10",
            fill=fill,
            alpha=alpha,
            common_norm=False,
            ax=ax,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(alpha=0.2)

    # Remove unused axes
    for ax in axes[len(X_num.columns):]:
        ax.remove()

    fig.suptitle(
        "Distribution (KDE) of Numerical Variables by Cluster",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()

    return fig, axes[:n_features]
import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_kdeplots_by_cluster(X, labels, n_cols=3, figsize=(15, 10), fill=False, alpha=0.8):
    """
    Plots KDE (Kernel Density Estimate) plots for each feature in the dataset, grouped by cluster labels.

    This function is useful for visualizing the distribution of features within each cluster,
    helping to identify patterns and differences between clusters.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the features to plot.
    labels : pd.Series or np.ndarray
        Cluster labels for each sample in the dataset.
    n_cols : int, optional
        Number of columns for the subplot grid (default is 3).
    figsize : tuple, optional
        Size of the figure (default is (15, 10)).
    fill : bool, optional
        Whether to fill the area under the KDE curve (default is False).
    alpha : float, optional
        Transparency level for the filled area (default is 0.8).

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the input DataFrame is empty or if the labels do not match the number of samples.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_plot = X.copy()
    X_plot["cluster"] = labels

    X_num = X_plot.select_dtypes(include=[np.number]).drop(columns=["cluster"], errors="ignore")
    n_features = X_num.shape[1]
    n_rows = int(np.ceil(n_features / n_cols))

    plt.figure(figsize=figsize)
    for i, col in enumerate(X_num.columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(
            data=X_plot, x=col, hue="cluster", palette="tab10",
            fill=fill, alpha=alpha, common_norm=False
        )
        plt.title(col, fontsize=10)
        plt.xlabel("")
        plt.ylabel("")
        plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.suptitle("Distribution (KDE) of Numerical Variables by Cluster", fontsize=14, y=1.02)
    plt.show()
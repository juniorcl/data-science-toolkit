import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cluster_feature_means(X, labels, top_n=10):
    """
    Plot a heatmap of the average feature values for each cluster.

    This function is useful for understanding the characteristics of each cluster
    by visualizing the mean feature values. It helps in identifying the most
    important features that differentiate the clusters.

    Parameters
    ----------
    X : pd.DataFrame
        The input features for clustering.
    labels : np.ndarray
        The cluster labels for each sample.
    top_n : int
        The number of top features to display.

    Returns
    -------
    None
        This function does not return a value.
        It plots a heatmap of average feature values per cluster.
    """
    X_df = pd.DataFrame(X).copy()
    X_df['cluster'] = labels
    cluster_means = X_df.groupby('cluster').mean()

    plt.figure(figsize=(10, min(top_n, X_df.shape[1]) * 0.4 + 2))
    sns.heatmap(cluster_means.iloc[:, :top_n], cmap="viridis", annot=False)
    plt.title("Average Features per Cluster")
    plt.tight_layout()
    plt.show()
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_silhouette_analysis(X, labels, metric='euclidean'):
    """
    Plot the silhouette analysis for the given clustering results.

    This function computes and visualizes the silhouette scores for each sample
    in the dataset, providing insights into the quality of the clustering.

    Parameters
    ----------
    X : pd.DataFrame
        The input data to cluster.
    labels : array-like, shape (n_samples,)
        The cluster labels for each sample.
    metric : str, optional
        The distance metric to use for computing the silhouette scores.
        Default is 'euclidean'.

    Returns
    -------
    None
        This function does not return a value, but it displays a plot
        of the silhouette analysis.
    
    Raises
    ------
    None
    """
    silhouette_avg = silhouette_score(X, labels, metric=metric)
    sample_silhouette_values = silhouette_samples(X, labels, metric=metric)

    n_clusters = len(np.unique(labels))
    y_lower = 10

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("tab10", n_clusters)

    for i, c in enumerate(np.unique(labels)):
        cluster_silhouette_vals = sample_silhouette_values[labels == c]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i % len(colors)]
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(c))
        y_lower = y_upper + 10

    plt.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Média = {silhouette_avg:.3f}")
    plt.title("Gráfico de Silhueta (Silhouette Plot)")
    plt.xlabel("Coeficiente de Silhueta")
    plt.ylabel("Cluster")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
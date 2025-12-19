import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cluster_sizes(labels):
    """
    Plots the sizes of clusters based on the provided labels.

    Each bar in the plot represents the number of samples in a particular cluster.

    Parameters
    ----------
    labels : list or array-like
        Cluster labels for each sample.

    Returns
    -------
    None
        This function does not return a value.
        It displays a bar plot of cluster sizes.

    Raises
    ------
    ValueError
        If the input labels are invalid.
    """
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Blues_d")
    plt.title("Clusters Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Observations")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.show()
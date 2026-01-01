import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca_projection(X, labels):
    """
    Plots the PCA projection of clustered data.

    It plots the first two principal components of the data points, 
    coloring them according to their cluster labels.
    
    Parameters
    ----------
    X : pd.DataFrame
        The input data to be projected.
    labels : array-like
        The cluster labels for each data point.

    Returns
    -------
    None
        This function does not return a value.
        It displays a scatter plot of the PCA projection.

    Raises
    ------
    ValueError
        If the input data X has less than 2 features.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=50)
    plt.title("PCA Projection of Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Cluster")
    plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca_projection(X, labels, figsize=(7, 6), random_state=42):
    """
    Plot a 2D PCA projection of clustered data.

    Parameters
    ----------
    X : pd.DataFrame or array-like of shape (n_samples, n_features)
        Input data to be projected.

    labels : array-like of shape (n_samples,)
        Cluster labels.

    figsize : tuple, default=(7, 6)
        Size of the matplotlib figure.

    random_state : int, default=42
        Random seed for PCA reproducibility.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If X has less than 2 features.
    """
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 features for PCA projection.")

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=labels,
        palette="tab10",
        s=50,
        ax=ax,
    )

    ax.set_title("PCA Projection of Clusters")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Cluster")

    return fig, ax
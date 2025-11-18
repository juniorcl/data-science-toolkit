import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP

def plot_cluster_umap(X, labels, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, palette="tab10", s=50)
    plt.title("UMAP Projection of Clusters")
    plt.xlabel("Dimension 1 (UMAP)")
    plt.ylabel("Dimension 2 (UMAP)")
    plt.legend(title="Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
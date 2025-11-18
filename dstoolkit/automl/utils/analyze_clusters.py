from .plot_cluster_sizes import plot_cluster_sizes
from .plot_silhouette_analysis import plot_silhouette_analysis
from .plot_kdeplots_by_cluster import plot_kdeplots_by_cluster
from .plot_cluster_pca import plot_cluster_pca
from .plot_cluster_umap import plot_cluster_umap

def analyze_clusters(X_orig, X, labels):
    plot_cluster_sizes(labels)
    plot_silhouette_analysis(X, labels)
    plot_kdeplots_by_cluster(X_orig, labels)
    plot_cluster_pca(X, labels)
    plot_cluster_umap(X, labels)
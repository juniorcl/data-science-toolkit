import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_cluster_pca(X, labels):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", s=50)
    plt.title("PCA Projection of Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Cluster")
    plt.show()
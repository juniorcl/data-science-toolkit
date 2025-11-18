import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_feature_means(X, labels, top_n=10):
    X_df = pd.DataFrame(X).copy()
    X_df['cluster'] = labels
    cluster_means = X_df.groupby('cluster').mean()

    plt.figure(figsize=(10, min(top_n, X_df.shape[1]) * 0.4 + 2))
    sns.heatmap(cluster_means.iloc[:, :top_n], cmap="viridis", annot=False)
    plt.title("Average Features per Cluster")
    plt.tight_layout()
    plt.show()
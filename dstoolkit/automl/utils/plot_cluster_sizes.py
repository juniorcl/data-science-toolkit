import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_sizes(labels):
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Blues_d")
    plt.title("Clusters Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Observations")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.show()

import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_kdeplots_by_cluster(X, labels, n_cols=3, figsize=(15, 10), fill=False, alpha=0.8):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_plot = X.copy()
    X_plot["cluster"] = labels

    X_num = X_plot.select_dtypes(include=[np.number]).drop(columns=["cluster"], errors="ignore")
    n_features = X_num.shape[1]
    n_rows = int(np.ceil(n_features / n_cols))

    plt.figure(figsize=figsize)
    for i, col in enumerate(X_num.columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(
            data=X_plot, x=col, hue="cluster", palette="tab10",
            fill=fill, alpha=alpha, common_norm=False
        )
        plt.title(col, fontsize=10)
        plt.xlabel("")
        plt.ylabel("")
        plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.suptitle("Distribution (KDE) of Numerical Variables by Cluster", fontsize=14, y=1.02)
    plt.show()
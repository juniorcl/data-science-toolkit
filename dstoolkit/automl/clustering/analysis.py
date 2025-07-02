import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

from scipy.spatial.distance import cdist

from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def get_umap_params(trial):

    return {
        'n_components': trial.suggest_int('n_components', 2, 10),  
        'n_neighbors': trial.suggest_int('n_neighbors', 5, 100),  
        'min_dist': trial.suggest_float('min_dist', 0.01, 0.99)
    }

def get_pca_params(trial, n_features, random_state=42):

    return {
        'n_components': trial.suggest_int('n_components', 2, n_features),
        'random_state': trial.suggest_categorical('random_state', [random_state])
    }

def plot_clusters(X, y, model_name):
    
    if X.shape[1] > 5:
        print("X has more than 5 features, selecting the first 5 for visualization.")
        X = X.iloc[:, :5]

    df = pd.concat([X, y], axis=1)

    plt.figure(figsize=(10, 6))
    sns.pairplot(data=df, hue=model_name)
    plt.show()

def plot_cluster_distribution(y, model_name):
    
    counts = pd.Series(y).value_counts().sort_index()
    
    plt.figure(figsize=(10, 4))
    bars = plt.bar(counts.index.astype(str), counts.values)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{int(height)}', ha='center', va='bottom'
        )
    
    plt.title(f"Distribuição de Clusters - {model_name}\nTotal clusters: {len(counts)}")
    plt.xlabel("Cluster ID")
    plt.ylabel("Número de Pontos")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_cluster_density(X, y, bandwidth=0.5):
    
    plt.figure(figsize=(10, 6))
    
    for cluster in np.unique(y):
        kde = KernelDensity(bandwidth=bandwidth).fit(X[y == cluster])
        scores = kde.score_samples(X[y == cluster])
        plt.hist(scores, bins=30, alpha=0.5, label=f'Cluster {cluster}')
    
    plt.xlabel('Log-Densidade')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Densidade por Cluster')
    plt.legend()
    plt.show()

def plot_silhouette_analysis(X, y, model_name):
    
    silhouette_vals = silhouette_samples(X, y)
    cluster_labels = np.unique(y)
    n_clusters = len(cluster_labels)
    
    plt.figure(figsize=(10, 6))
    y_lower = 10
    
    for i in cluster_labels:
        ith_cluster_silhouette = silhouette_vals[y == i]
        ith_cluster_silhouette.sort()
        
        size_cluster_i = ith_cluster_silhouette.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, ith_cluster_silhouette,
            facecolor=color, edgecolor=color, alpha=0.7
        )
        
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    silhouette_avg = np.mean(silhouette_vals)
    plt.title(f"Silhouette Plot - {model_name}\nAvg Score: {silhouette_avg:.2f}")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster Label")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.show()

def plot_cluster_centroid_distances(X, y):
    
    centroids = pd.DataFrame(X).groupby(y).mean()
    distances = cdist(centroids, centroids)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(distances, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Distance between Cluster Centers")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.show()

def plot_feature_importance_by_cluster(X, y, top_n=10):
    
    importances = []

    for cluster in np.unique(y):
        mean_diff = X[y == cluster].mean() - X.mean()
        importance = mean_diff.abs().sort_values(ascending=False).head(top_n)
        importances.append(importance)

    df_importance = pd.concat(importances, axis=1)
    df_importance.columns = [f"Cluster {i}" for i in np.unique(y)]

    df_importance.plot(kind='barh', figsize=(10, 6))
    plt.title("Top Features by Cluster")
    plt.xlabel("Mean Absolute Difference")
    plt.grid(True, axis='x')
    plt.show()

def get_results(X, y, model_name):

    plot_clusters(X, y[model_name], model_name)
    plot_silhouette_analysis(X, y[model_name], model_name)
    plot_feature_importance_by_cluster(X, y[model_name])
    plot_cluster_centroid_distances(X, y[model_name])
    plot_cluster_distribution(y[model_name], model_name)
    plot_cluster_density(X, y[model_name])

def get_metrics(X, y):
    
    return {
        'Silhouetete': silhouette_score(X, y),
        'Davies Bouldin': davies_bouldin_score(X, y),
        'Calinski Harabasz': calinski_harabasz_score(X, y)
    }
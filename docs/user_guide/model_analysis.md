# Model Analysis

The `dstoolkit.model.analysis` package provides diagnostic and visualization tools for inspecting trained models.

---

## Classification & Regression

```python
from dstoolkit.model.analysis import (
    plot_learning_curve,
    plot_error_by_quantile,
    plot_waste_distribution,
)
```

### plot_learning_curve(model, X, y, scoring)

Plots training and validation scores as a function of training set size.

```python
plot_learning_curve(model, X_train, y_train, scoring="roc_auc")
```

### plot_error_by_quantile(y_true, y_score)

Boxplot of absolute error grouped by target quantile. Useful for identifying where a regression model performs poorly.

```python
plot_error_by_quantile(y_test, y_pred)
```

### plot_waste_distribution(y_true, y_score)

Histogram of residuals (predicted - actual) for regression models.

```python
plot_waste_distribution(y_test, y_pred)
```

---

## Clustering

```python
from dstoolkit.model.analysis import (
    plot_cluster_sizes,
    plot_silhouette_analysis,
    plot_pca_projection,
    plot_umap_projection,
    plot_kdeplots_by_cluster,
    plot_numerical_variables_distribution_by_cluster,
    plot_tree_ovr,
    plot_tree_ovo,
)
```

### plot_cluster_sizes(labels)

Bar plot of the number of samples per cluster.

### plot_silhouette_analysis(X, labels)

Silhouette coefficient plot showing cluster cohesion and separation.

### plot_pca_projection(X, labels)

2D scatter plot of data projected onto the first two principal components, colored by cluster.

### plot_umap_projection(X, labels)

2D scatter plot of UMAP embedding, colored by cluster. Requires `umap-learn`.

### plot_kdeplots_by_cluster(X, labels)

Grid of KDE plots, one per feature, with distributions colored by cluster.

### plot_numerical_variables_distribution_by_cluster(X, labels)

KDE plot for each numerical variable grouped by cluster.

### plot_tree_ovr(X, labels)

One-vs-Rest decision tree surrogates — trains a simple decision tree per cluster to explain cluster membership rules.

```python
plot_tree_ovr(X_orig, automl.labels)
```

### plot_tree_ovo(X, labels)

One-vs-One decision tree surrogates — trains decision trees for each pair of clusters.

---

## API Reference

- [model API](../api_reference/model.md)

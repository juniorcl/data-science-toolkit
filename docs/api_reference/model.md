# dstoolkit.model

## Subpackages

```
dstoolkit.model.analysis         — Model diagnostics and visualization
dstoolkit.model.interpretability — Model explanation tools
```

---

## dstoolkit.model.analysis

### plot_learning_curve(model, X, y, scoring)

Plot training and validation scores vs. training set size.

- **model:** estimator instance.
- **X:** array-like, features.
- **y:** array-like, target.
- **scoring:** str or callable, sklearn-compatible scorer.

### plot_error_by_quantile(y_true, y_pred)

Boxplot of absolute error grouped by target quantile. (Regression)

### plot_waste_distribution(y_true, y_pred)

Histogram of residuals (predicted – actual). (Regression)

### plot_cluster_sizes(labels)

Bar plot of sample counts per cluster.

- **labels:** array-like, cluster assignments.

### plot_silhouette_analysis(X, labels)

Silhouette coefficient plot showing cluster cohesion.

### plot_pca_projection(X, labels)

2D PCA scatter plot colored by cluster label.

### plot_umap_projection(X, labels)

2D UMAP scatter plot colored by cluster label. Requires `umap-learn`.

### plot_kdeplots_by_cluster(X, labels)

Grid of KDE plots per feature, colored by cluster.

### plot_numerical_variables_distribution_by_cluster(X, labels)

KDE plot per numerical variable grouped by cluster.

### plot_tree_ovr(X, labels)

One-vs-Rest decision tree surrogates per cluster.

### plot_tree_ovo(X, labels)

One-vs-One decision tree surrogates for each cluster pair.

---

## dstoolkit.model.interpretability

### plot_feature_importance(model, top_n=20)

Horizontal bar chart of native feature importance from the model.

- **model:** trained model with `feature_importances_` attribute.
- **top_n:** int, number of top features to display.

### plot_permutation_importance(model, X, y, scoring)

Boxplot of permutation-based feature importance (model-agnostic).

- **model:** trained estimator.
- **X:** array-like, features.
- **y:** array-like, target.
- **scoring:** str or callable, sklearn-compatible scorer.

### plot_shap_tree_summary(model, X)

SHAP summary beeswarm plot using `TreeExplainer`.

- **model:** tree-based model (LightGBM, CatBoost, HistGradientBoosting).
- **X:** array-like, features for explanation.

### plot_shap_linear_summary(model, X)

SHAP summary beeswarm plot using `LinearExplainer`.

- **model:** linear model.
- **X:** array-like, features for explanation.

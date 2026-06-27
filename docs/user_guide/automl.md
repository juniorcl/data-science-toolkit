# AutoML

The `dstoolkit.automl` package provides automated ML pipelines for classification, regression, and clustering. Each pipeline wraps a popular model class with **Optuna-based hyperparameter tuning**, **holdout or cross-validation evaluation**, and **integrated diagnostics**.

---

## Classification

All classifiers share a common API with methods `train()`, `get_metrics()`, and `analyze()`.

### Available Classes

| Class | Backend | Validation |
|---|---|---|
| `AutoMLLightGBM` | LightGBM `LGBMClassifier` | Holdout |
| `AutoMLLightGBMCV` | LightGBM `LGBMClassifier` | Cross-validation |
| `AutoMLCatBoost` | CatBoost `CatBoostClassifier` | Holdout |
| `AutoMLCatBoostCV` | CatBoost `CatBoostClassifier` | Cross-validation |
| `AutoMLHistGradientBoosting` | sklearn `HistGradientBoostingClassifier` | Holdout |
| `AutoMLHistGradientBoostingCV` | sklearn `HistGradientBoostingClassifier` | Cross-validation |

### Usage

```python
from dstoolkit.automl.classifier import AutoMLLightGBM

automl = AutoMLLightGBM(
    scoring="roc_auc",  # metric to optimize
    tune=True,          # enable Optuna tuning
    n_trials=50,
    random_state=42
)

automl.train(
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    target="target"
)

# Metrics
df = automl.get_metrics(return_df=True)

# Best params
print(automl.best_params)

# Diagnostic plots (ROC, KS, PR, calibration, learning curve, SHAP, etc.)
automl.analyze()
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scoring` | str | `'roc_auc'` | Metric to optimize. See [scoring utils](#scoring-utils). |
| `tune` | bool | `False` | Enable Optuna hyperparameter search. |
| `n_trials` | int | `50` | Number of Optuna trials (only when `tune=True`). |
| `random_state` | int | `42` | Random seed for reproducibility. |

---

## Regression

Same API as classifiers. All classes are in `dstoolkit.automl.regressor`.

### Available Classes

| Class | Backend | Validation |
|---|---|---|
| `AutoMLLightGBM` | LightGBM `LGBMRegressor` | Holdout |
| `AutoMLLightGBMCV` | LightGBM `LGBMRegressor` | Cross-validation |
| `AutoMLCatBoost` | CatBoost `CatBoostRegressor` | Holdout |
| `AutoMLCatBoostCV` | CatBoost `CatBoostRegressor` | Cross-validation |
| `AutoMLHistGradientBoosting` | sklearn `HistGradientBoostingRegressor` | Holdout |
| `AutoMLHistGradientBoostingCV` | sklearn `HistGradientBoostingRegressor` | Cross-validation |

### Usage

```python
from dstoolkit.automl.regressor import AutoMLCatBoost

automl = AutoMLCatBoost(scoring="r2", tune=True, n_trials=30)
automl.train(X_train, y_train, X_valid, y_valid, X_test, y_test)
print(automl.get_metrics())
automl.analyze()
```

---

## Clustering

Clustering automators use `train(X)` (no target labels). Available in `dstoolkit.automl.clustering`.

### Available Classes

| Class | Algorithm | Metric |
|---|---|---|
| `AutoMLKMeans` | KMeans | silhouette, davies_bouldin_score, calinski_harabasz_score |
| `AutoMLGaussianMixture` | GaussianMixture | silhouette, davies_bouldin_score, calinski_harabasz_score |

### Usage

```python
from dstoolkit.automl.clustering import AutoMLKMeans

automl = AutoMLKMeans(scoring="silhouette", n_trials=30)
automl.train(X)

print(automl.get_metrics())
automl.analyze(X_orig=X)  # diagnostics with original features
```

### analyze() Output

The `analyze(X_orig)` method for clustering produces:

- Cluster size bar plot
- Silhouette analysis
- KDE plots per feature per cluster
- Decision-tree surrogate (OvR)
- PCA projection (2D)
- UMAP projection (2D)

---

## PCA Optimizer

`dstoolkit.automl.steps.PCAOptimizer` optimizes PCA hyperparameters via Optuna.

```python
from dstoolkit.automl.steps import PCAOptimizer

pca = PCAOptimizer(scoring="explained_variance", n_trials=30)
pca.fit(X)
print(pca.get_metrics())
print("Best n_components:", pca.best_params["n_components"])
```

Supported scoring metrics:

| Metric | Description |
|---|---|
| `'explained_variance'` | Cumulative explained variance ratio |
| `'reconstruction_error'` | Negative mean squared error of reconstruction |
| `'trustworthiness'` | Trustworthiness of the embedding (sklearn metric) |

---

## Scoring Utils

Each automl subpackage provides utility functions for scoring:

- `get_classifier_score(scoring)` / `get_classifier_function_score(scoring)` — classifier scorers
- `get_regressor_score(scoring)` / `get_regressor_function_score(scoring)` — regressor scorers
- `get_cluster_score(scoring)` / `get_cluster_function_score(scoring)` — cluster scorers

These are used internally but can be imported directly:

```python
from dstoolkit.automl.classifier.utils import get_classifier_score

scorer = get_classifier_score("roc_auc")  # returns a sklearn-compatible scorer
```

---

## API Reference

- [automl API](../api_reference/automl.md)

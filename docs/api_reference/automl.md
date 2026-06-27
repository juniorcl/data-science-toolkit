# dstoolkit.automl

## Subpackages

```
dstoolkit.automl.classifier       — AutoML classification pipelines
dstoolkit.automl.classifier.cv    — AutoML classification with cross-validation
dstoolkit.automl.regressor        — AutoML regression pipelines
dstoolkit.automl.regressor.cv     — AutoML regression with cross-validation
dstoolkit.automl.clustering       — AutoML clustering pipelines
dstoolkit.automl.steps            — Pipeline step optimizers (PCA)
```

---

## dstoolkit.automl.classifier

### AutoMLLightGBM

```python
class AutoMLLightGBM(
    scoring: str = "roc_auc",
    tune: bool = False,
    n_trials: int = 50,
    random_state: int = 42
)
```

Automated LightGBM classifier with Optuna hyperparameter tuning (holdout).

- `train(X_train, y_train, X_valid, y_valid, X_test, y_test, target="target")` — Fit model.
- `get_metrics(return_df=True)` — Return evaluation metrics.
- `analyze()` — Generate ROC, KS, PR, calibration, learning curve, SHAP plots.

### AutoMLCatBoost

```python
class AutoMLCatBoost(
    scoring: str = "roc_auc",
    tune: bool = False,
    n_trials: int = 50,
    random_state: int = 42
)
```

Automated CatBoost classifier with holdout validation.

### AutoMLHistGradientBoosting

```python
class AutoMLHistGradientBoosting(
    scoring: str = "roc_auc",
    tune: bool = False,
    n_trials: int = 50,
    random_state: int = 42
)
```

Automated sklearn HistGradientBoosting classifier with holdout validation.

### AutoMLLightGBMCV

```python
class AutoMLLightGBMCV(
    scoring: str = "roc_auc",
    tune: bool = False,
    n_trials: int = 50,
    random_state: int = 42
)
```

LightGBM classifier with cross-validation tuning.

### AutoMLCatBoostCV

```python
class AutoMLCatBoostCV(
    scoring: str = "roc_auc",
    tune: bool = False,
    n_trials: int = 50,
    random_state: int = 42
)
```

CatBoost classifier with cross-validation tuning.

### AutoMLHistGradientBoostingCV

```python
class AutoMLHistGradientBoostingCV(
    scoring: str = "roc_auc",
    tune: bool = False,
    n_trials: int = 50,
    random_state: int = 42
)
```

HistGradientBoosting classifier with cross-validation tuning.

---

## dstoolkit.automl.regressor

Same class names as classifier but for regression. Identical API — `train()`, `get_metrics()`, `analyze()`.

| Class | Backend |
|---|---|
| `AutoMLLightGBM` | `LGBMRegressor` |
| `AutoMLCatBoost` | `CatBoostRegressor` |
| `AutoMLHistGradientBoosting` | `HistGradientBoostingRegressor` |
| `AutoMLLightGBMCV` | `LGBMRegressor` (CV) |
| `AutoMLCatBoostCV` | `CatBoostRegressor` (CV) |
| `AutoMLHistGradientBoostingCV` | `HistGradientBoostingRegressor` (CV) |

---

## dstoolkit.automl.clustering

### AutoMLKMeans

```python
class AutoMLKMeans(
    scoring: str = "silhouette",
    n_trials: int = 50,
    random_state: int = 42
)
```

Automated KMeans clustering with Optuna.

- `train(X)` — Fit model.
- `get_metrics(return_df=True)` — Return clustering metrics.
- `analyze(X_orig)` — Generate diagnostic plots.

### AutoMLGaussianMixture

```python
class AutoMLGaussianMixture(
    scoring: str = "silhouette",
    n_trials: int = 50,
    random_state: int = 42
)
```

Automated Gaussian Mixture clustering with Optuna.

---

## dstoolkit.automl.steps

### PCAOptimizer

```python
class PCAOptimizer(
    scoring: str = "explained_variance",
    n_trials: int = 50,
    random_state: int = 42
)
```

PCA with Optuna hyperparameter optimization.

- `fit(X)` — Fit optimized PCA.
- `get_metrics(return_df=True)` — Return explained variance, reconstruction error, trustworthiness.

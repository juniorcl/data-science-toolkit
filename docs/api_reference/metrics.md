# dstoolkit.metrics

## Subpackages

```
dstoolkit.metrics.scores     — Custom evaluation metric functions
dstoolkit.metrics.plots      — Evaluation plot functions
```

---

## dstoolkit.metrics.scores

### get_classifier_metrics(y_true, y_pred, y_score) -> dict

Compute a comprehensive set of classifier metrics.

**Returns:** dict with keys `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`, `ks`, `brier_score`, `log_loss`, `average_precision_lift`.

### get_regressor_metrics(y_true, y_pred) -> dict

Compute regression metrics.

**Returns:** dict with keys `r2`, `mae`, `made`, `mape`, `rmse`, `explained_variance`.

### get_cluster_metrics(X, labels) -> dict

Compute clustering metrics.

**Returns:** dict with keys `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`.

### ks_score(y_true, y_score) -> float

Kolmogorov–Smirnov statistic from the ROC curve. Range [0, 1].

### ks_scorer

An sklearn `make_scorer` wrapper for `ks_score`. Use with `cross_val_score`, `GridSearchCV`, etc.

```python
from dstoolkit.metrics.scores import ks_scorer
scores = cross_val_score(model, X, y, scoring=ks_scorer)
```

### average_precision_lift_score(y_true, y_score) -> float

Average Precision divided by the baseline (random classifier) precision.

### average_precision_lift_scorer

An sklearn `make_scorer` wrapper for `average_precision_lift_score`.

---

## dstoolkit.metrics.plots

### plot_roc_curve(y_true, y_score)

Plot ROC curve with AUC annotation.

- **y_true:** array-like, true binary labels.
- **y_score:** array-like, predicted probabilities.

### plot_ks_curve(y_true, y_score)

Plot KS cumulative distribution curves with KS statistic.

### plot_precision_recall_curve(y_true, y_score)

Plot Precision-Recall curve with Average Precision annotation.

### plot_calibration_curve(y_true, y_score, strategy='uniform')

Plot calibration curve (reliability diagram) with Brier score.

- **strategy:** binning strategy — `'uniform'` or `'quantile'`.

# Metrics & Plots

The `dstoolkit.metrics` package provides custom evaluation metrics and publication-ready diagnostic plots.

---

## Score Functions

`dstoolkit.metrics.scores` provides metric calculators that return dictionaries of evaluation results.

### Classification

```python
from dstoolkit.metrics.scores import get_classifier_metrics

metrics = get_classifier_metrics(y_true, y_pred, y_score)
# Returns dict with: accuracy, balanced_accuracy, precision, recall,
#                    f1_score, roc_auc, ks, brier_score, log_loss,
#                    average_precision_lift
```

### Regression

```python
from dstoolkit.metrics.scores import get_regressor_metrics

metrics = get_regressor_metrics(y_true, y_pred)
# Returns dict with: r2, mae, made, mape, rmse, explained_variance
```

### Clustering

```python
from dstoolkit.metrics.scores import get_cluster_metrics

metrics = get_cluster_metrics(X, labels)
# Returns dict with: silhouette_score, calinski_harabasz_score, davies_bouldin_score
```

### Custom Scorers

| Function | Description |
|---|---|
| `ks_score(y_true, y_score)` | Kolmogorov–Smirnov statistic from the ROC curve. |
| `ks_scorer` | sklearn-compatible `make_scorer` wrapper for `ks_score`. |
| `average_precision_lift_score(y_true, y_score)` | Average Precision relative to a random baseline. |
| `average_precision_lift_scorer` | sklearn-compatible `make_scorer` wrapper. |

```python
from dstoolkit.metrics.scores import ks_score, ks_scorer
from sklearn.model_selection import cross_val_score

# Function
ks = ks_score(y_true, y_score)

# Scorer (use with sklearn's cross_val_score, GridSearchCV, etc.)
scores = cross_val_score(model, X, y, scoring=ks_scorer)
```

---

## Plots

`dstoolkit.metrics.plots` provides matplotlib-based evaluation plots.

```python
from dstoolkit.metrics.plots import (
    plot_roc_curve,
    plot_ks_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
)
```

### plot_roc_curve(y_true, y_score)

ROC curve with AUC annotation.

### plot_ks_curve(y_true, y_score)

KS cumulative distribution curves with KS statistic annotation.

### plot_precision_recall_curve(y_true, y_score)

Precision-Recall curve with Average Precision annotation.

### plot_calibration_curve(y_true, y_score, strategy='uniform')

Calibration curve (reliability diagram) with Brier score annotation.

---

## API Reference

- [metrics API](../api_reference/metrics.md)

# Interpretability

The `dstoolkit.model.interpretability` package provides model explanation tools using **SHAP** and built-in feature importance.

```python
from dstoolkit.model.interpretability import (
    plot_feature_importance,
    plot_permutation_importance,
    plot_shap_tree_summary,
    plot_shap_linear_summary,
)
```

---

## Feature Importance

### plot_feature_importance(model, top_n=20)

Horizontal bar chart of native feature importance from the model.

```python
plot_feature_importance(automl.model, top_n=15)
```

Supports tree-based models (LightGBM, CatBoost, HistGradientBoosting) that expose `feature_importances_`.

### plot_permutation_importance(model, X, y, scoring)

Boxplot of permutation-based feature importance using `sklearn.inspection.permutation_importance`. Model-agnostic.

```python
plot_permutation_importance(automl.model, X_test, y_test, scoring="roc_auc")
```

---

## SHAP

### plot_shap_tree_summary(model, X)

SHAP summary beeswarm plot using `TreeExplainer`. Works with LightGBM, CatBoost, and HistGradientBoosting.

```python
plot_shap_tree_summary(automl.model, X_test)
```

### plot_shap_linear_summary(model, X)

SHAP summary beeswarm plot using `LinearExplainer`. Works with linear models.

```python
plot_shap_linear_summary(linear_model, X_test)
```

---

## API Reference

- [model API](../api_reference/model.md)

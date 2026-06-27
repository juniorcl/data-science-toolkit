# DSToolkit

**Version 0.9.5** | A modular Python library for applied data science.

DSToolkit accelerates real-world ML workflows with **AutoML pipelines**, **model evaluation utilities**, **interpretability tools**, and **feature engineering helpers** — all following a scikit-learn–inspired API.

---

## Features

| Module | Description |
|---|---|
| `dstoolkit.automl` | Automated training pipelines for classification, regression, and clustering with LightGBM, CatBoost, HistGradientBoosting, KMeans, and Gaussian Mixture Models. Hyperparameter tuning via Optuna. Holdout and cross-validation strategies. |
| `dstoolkit.metrics` | Custom evaluation metrics (KS, Average Precision Lift, Brier, etc.) and publication-ready plots (ROC, PR, KS, calibration curves). |
| `dstoolkit.model.analysis` | Model diagnostics: learning curves, residual analysis, cluster size distributions, silhouette analysis, PCA/UMAP projections, and decision-tree surrogates (OvR, OvO). |
| `dstoolkit.model.interpretability` | SHAP-based explanations, native feature importance, and permutation importance plots. |
| `dstoolkit.feature.engineering` | sklearn-compatible transformers for categorical encoding and custom transformations. |
| `dstoolkit.feature.selection` | Feature subset selection compatible with sklearn Pipelines. |
| `dstoolkit.feature.monitoring` | Drift detection: Population Stability Index (PSI), Kolmogorov–Smirnov test, Chi-squared test, and Jensen–Shannon divergence. |

---

## Example at a Glance

```python
from dstoolkit.automl.classifier import AutoMLLightGBM

automl = AutoMLLightGBM(scoring="roc_auc", tune=True, n_trials=50)
automl.train(X_train, y_train, X_valid, y_valid, X_test, y_test)

# Evaluation metrics
print(automl.get_metrics())

# Diagnostic plots
automl.analyze()
```

---

## Contents

- **[Installation](install.md)** — How to install DSToolkit.
- **[Getting Started](getting_started.md)** — A quick tour of the library.
- **[User Guide](user_guide/index.md)** — In-depth guides for each module.
- **[API Reference](api_reference/index.md)** — Full class and function reference.
- **[Examples](examples.md)** — Jupyter notebooks demonstrating use cases.
- **[Contributing](contributing.md)** — How to contribute to the project.
- **[Changelog](changelog.md)** — Version history.

---

## License

MIT — see the [LICENSE](https://github.com/clebiomojunior/data-science-toolkit/blob/main/LICENSE) file.

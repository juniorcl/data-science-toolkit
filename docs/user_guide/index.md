# User Guide

```text
src/dstoolkit/
├── automl/                # AutoML pipelines
├── metrics/                # Custom metrics and plots
├── model/
│   ├── analysis/           # Model diagnostics and visualization
│   └── interpretability/   # Explainability tools
└── feature/
    ├── engineering/        # Feature engineering transformers
    ├── selection/          # Feature subset selection
    └── monitoring/         # Distribution drift detection
```

---

## Contents

| Guide | Description |
|---|---|
| **[AutoML](automl.md)** | Classification, regression, and clustering pipelines with Optuna hyperparameter tuning. LightGBM, CatBoost, HistGradientBoosting, KMeans, Gaussian Mixture, and PCA optimizer. |
| **[Metrics & Plots](metrics_and_plots.md)** | Custom metric functions (KS, Average Precision Lift, Brier, etc.) and evaluation plots (ROC, PR, KS, calibration curves). |
| **[Model Analysis](model_analysis.md)** | Learning curves, residual analysis, cluster diagnostics (silhouette, PCA/UMAP projections, KDE plots), and decision-tree surrogates. |
| **[Interpretability](interpretability.md)** | SHAP summary plots, feature importance, and permutation importance. |
| **[Feature Engineering](feature_engineering.md)** | Categorical encoders, custom function transformers, feature selectors, and drift monitoring. |

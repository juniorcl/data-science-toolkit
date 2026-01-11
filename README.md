# Data Science Toolkit

DSToolkit is a modular Python library designed to accelerate applied data science workflows.  
It provides **AutoML pipelines**, **model evaluation utilities**, **interpretability tools**, and **feature engineering helpers** for:

- Classification
- Regression
- Clustering

The library follows a **scikit-learn–inspired API**, focusing on:
- Reproducibility
- Clean abstractions
- Practical diagnostics for real-world ML problems

## Key Features

### AutoML
- Automated training pipelines for:
  - Classification
  - Regression
  - Clustering
- Built-in support for:
  - LightGBM
  - CatBoost
  - HistGradientBoosting
  - KMeans
  - Gaussian Mixture Models
- Hyperparameter optimization
- Holdout and cross-validation strategies

### Model Analysis & Metrics
- Classification metrics:
  - ROC, PR, KS, calibration curves
  - Custom scorers and lift-based metrics
- Regression diagnostics:
  - Residual analysis
  - Error by quantile
  - True vs predicted
- Clustering evaluation:
  - Silhouette analysis
  - Cluster size distribution
  - Feature statistics per cluster

### Interpretability
- SHAP-based explanations
- Feature importance
- Decision tree surrogates (OvR and OvO)

### Feature Engineering
- Encoders and wrappers compatible with sklearn pipelines
- Custom transformation utilities

## Project Structure

```text
src/dstoolkit/
├── automl/                # AutoML pipelines
├── feature/               # Feature engineering utilities
│   ├── engineering/       # Classes for feature engineering
│   └── monitoring/        # Function for feature monitoring
│   └── selection/         # Class for feature selection
├── metrics/               # Custom metrics and plots
│   └── plots/             # Plots for evaluation
│   └── scores/            # Function for metrics evaluation
├── model/
│   ├── analysis/          # Model diagnostics and visualization
│   └── interpretability/  # Explainability tools
```

Example notebooks demonstrating usage can be found in the `notebooks/` directory.

## Instalation

```bash
pip install -r requirements.txt
```

Or, for development:


```bash
pip install -e .
```

## Quick Example

```python
from dstoolkit.automl.classifier import AutoMLLightGBMClassifier

automl = AutoMLLightGBMClassifier(
    scoring="roc_auc",
    n_trials=50
)

automl.fit(X_train, y_train)
automl.evaluate(X_test, y_test)
```

## Documentation

Detailed documentation is available in the `docs/` directory:
- AutoML APIs
- Metrics and scoring
- Model analysis and interpretability
- Feature engineering utilities

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

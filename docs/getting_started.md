# Getting Started

This guide walks through a complete AutoML classification workflow.

---

## 1. Load and Split Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

---

## 2. Train an AutoML Classifier

```python
from dstoolkit.automl.classifier import AutoMLLightGBM

automl = AutoMLLightGBM(
    scoring="roc_auc",
    tune=True,
    n_trials=50,
    random_state=42
)

automl.train(X_train, y_train, X_valid, y_valid, X_test, y_test, target="target")
```

---

## 3. Inspect Metrics

```python
metrics_df = automl.get_metrics(return_df=True)
print(metrics_df)
```

This returns a DataFrame with evaluation metrics (accuracy, AUC, F1, etc.) for Train, Valid, and Test sets.

```python
print("Best hyperparameters:", automl.best_params)
```

---

## 4. Generate Diagnostic Plots

```python
automl.analyze()
```

This produces:

- ROC curve
- KS curve
- Precision-Recall curve
- Calibration curve
- Learning curve
- Feature importance
- Permutation importance
- SHAP summary

---

## 5. Regression and Clustering

The same workflow applies to regressors and clustering:

```python
from dstoolkit.automl.regressor import AutoMLCatBoost
from dstoolkit.automl.clustering import AutoMLKMeans

# Regression
reg = AutoMLCatBoost(scoring="r2", tune=True, n_trials=30)
reg.train(X_train, y_train, X_valid, y_valid, X_test, y_test)
print(reg.get_metrics())

# Clustering
cluster = AutoMLKMeans(scoring="silhouette", n_trials=30)
cluster.train(X)
print(cluster.get_metrics())
cluster.analyze(X_orig=X)
```

---

## Next Steps

- **[User Guide: AutoML](user_guide/automl.md)** — Deep dive into all AutoML options.
- **[User Guide: Metrics & Plots](user_guide/metrics_and_plots.md)** — Custom metrics and evaluation plots.
- **[Examples](examples.md)** — Full example notebooks.

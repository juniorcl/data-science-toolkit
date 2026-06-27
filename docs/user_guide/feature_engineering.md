# Feature Engineering

The `dstoolkit.feature` package provides transformers, selectors, and monitoring functions for data preprocessing.

---

## Engineering

### CategoryEncoderWrapper

A sklearn-compatible wrapper for categorical encoders.

```python
from dstoolkit.feature.engineering import CategoryEncoderWrapper
import category_encoders as ce

encoder = CategoryEncoderWrapper(
    encoder=ce.TargetEncoder(),
    columns=["category_col1", "category_col2"]
)

X_encoded = encoder.fit_transform(X, y)
```

### FunctionTransformerWrapper

A `FunctionTransformer` wrapper that preserves or generates feature names.

```python
from dstoolkit.feature.engineering import FunctionTransformerWrapper
import numpy as np

def sqrt_transform(X):
    return np.sqrt(X)

transformer = FunctionTransformerWrapper(
    func=sqrt_transform,
    feature_names_out="sqrt_{}"
)

X_transformed = transformer.fit_transform(X)
```

---

## Selection

### FeatureSelector

Select a subset of features from a DataFrame or NumPy array. Compatible with sklearn `Pipeline`.

```python
from dstoolkit.feature.selection.feature_selector import FeatureSelector

# By feature names
selector = FeatureSelector(features=["age", "income", "score"])
X_selected = selector.fit_transform(X)

# By boolean mask
selector = FeatureSelector(mask=[True, False, True, True])
X_selected = selector.fit_transform(X)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `features` | Iterable[str or int] | `None` | Column names (str) or indices (int) to select. |
| `mask` | Iterable[bool] | `None` | Boolean mask for feature selection. |
| `check_missing` | bool | `True` | Validate that requested features exist in input. |

**Methods:**

| Method | Description |
|---|---|
| `fit(X, y=None)` | Learn the selected features from the data. |
| `transform(X)` | Return a subset of columns. |
| `inverse_transform(X)` | Return original structure with NaNs for unselected features. |
| `get_feature_names_out(input_features=None)` | Names of the selected features. |
| `get_support(indices=False)` | Boolean mask or index array of selected features. |

---

## Monitoring

Feature monitoring functions detect distribution drift between reference and production datasets.

### Population Stability Index (PSI)

```python
from dstoolkit.feature.monitoring import psi

psi_value = psi(expected, actual)
```

Measures the stability of a feature distribution across two time periods.

### Kolmogorov–Smirnov Drift Test

```python
from dstoolkit.feature.monitoring import ks_test_drift

stat, p_value = ks_test_drift(expected, actual)
```

Two-sample KS test for continuous feature drift detection.

### Chi-Squared Monitoring

```python
from dstoolkit.feature.monitoring import chi_squared_monitoring

stat, p_value = chi_squared_monitoring(expected, actual)
```

Chi-squared test for categorical feature drift detection.

### Jensen–Shannon Divergence

```python
from dstoolkit.feature.monitoring import jensen_shannon_divergence

jsd = jensen_shannon_divergence(expected, actual)
```

Symmetrical divergence measure for comparing two probability distributions.

---

## API Reference

- [feature API](../api_reference/feature.md)

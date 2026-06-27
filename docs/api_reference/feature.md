# dstoolkit.feature

## Subpackages

```
dstoolkit.feature.engineering  — Feature engineering transformers
dstoolkit.feature.selection    — Feature subset selection
dstoolkit.feature.monitoring   — Distribution drift detection
```

---

## dstoolkit.feature.engineering

### CategoryEncoderWrapper

```python
class CategoryEncoderWrapper(
    encoder: object,
    columns: list
)
```

A sklearn-compatible wrapper for categorical encoders (e.g., from `category_encoders`).

- `fit(X, y=None)` — Fit the encoder on selected columns.
- `transform(X)` — Transform selected columns.
- `fit_transform(X, y=None)` — Fit then transform.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `encoder` | object | An encoder instance with `fit_transform` and `transform` methods. |
| `columns` | list | Column names to encode. |

### FunctionTransformerWrapper

```python
class FunctionTransformerWrapper(
    func: callable,
    feature_names_out: str = None
)
```

A FunctionTransformer wrapper that preserves feature names.

- `fit(X, y=None)` — Store feature names.
- `transform(X)` — Apply function and rename output columns.
- `fit_transform(X, y=None)` — Fit then transform.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `func` | callable | Function to apply to the data. |
| `feature_names_out` | str | Format string for output feature names (e.g., `"sqrt_{}"`). |

---

## dstoolkit.feature.selection

### FeatureSelector

```python
class FeatureSelector(
    features: Iterable[str | int] = None,
    mask: Iterable[bool] = None,
    check_missing: bool = True
)
```

Select a subset of features from a DataFrame or NumPy array. Compatible with sklearn `Pipeline`.

- `fit(X, y=None)` — Validate and store selected features.
- `transform(X)` — Return selected subset.
- `inverse_transform(X)` — Restore original shape with NaNs.
- `get_feature_names_out(input_features=None)` — Names of selected features.
- `get_support(indices=False)` — Bool mask or index array.

---

## dstoolkit.feature.monitoring

### psi(expected, actual) -> float

Population Stability Index.

```python
psi(expected, actual)
```

- **expected:** array-like, reference distribution.
- **actual:** array-like, current distribution.

### ks_test_drift(expected, actual) -> (float, float)

Two-sample Kolmogorov–Smirnov drift test.

```python
stat, p_value = ks_test_drift(expected, actual)
```

- **Returns:** KS statistic and p-value.

### chi_squared_monitoring(expected, actual) -> (float, float)

Chi-squared drift test for categorical features.

```python
stat, p_value = chi_squared_monitoring(expected, actual)
```

- **Returns:** chi-squared statistic and p-value.

### jensen_shannon_divergence(expected, actual) -> float

Jensen–Shannon divergence between two distributions.

```python
jsd = jensen_shannon_divergence(expected, actual)
```

- **Returns:** JSD value (0 = identical, log(2) = maximally different for base-2).

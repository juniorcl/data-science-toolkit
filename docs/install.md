# Installation

## Requirements

- Python >= 3.10
- pip

DSToolkit depends on `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `catboost`, `optuna`, `shap`, `seaborn`, `umap-learn`, and `numba`.

---

## Standard Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/clebiomojunior/data-science-toolkit.git
cd data-science-toolkit
pip install -r requirements.txt
```

---

## Development Install

Install in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

This installs `pytest` and `pytest-cov` for running tests.

---

## Verify the Installation

```python
import dstoolkit
print(dstoolkit.__version__)
```

You should see `0.9.5` (or the installed version).

---

## Optional Dependencies

DSToolkit uses `numba` for performance in certain metrics. If you encounter installation issues with `numba` on your platform, you can still use the library — only the affected functions will be slower.

---

## Next Steps

- [Getting Started](getting_started.md) — Run your first AutoML pipeline.
- [User Guide](user_guide/index.md) — Explore the library in depth.

#%%
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from dstoolkit.automl.classifier import AutoMLClassifier

#%%
def test_automl_runs():
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        random_state=42
    )

    automl = AutoMLClassifier(
        models=[
            LogisticRegression(max_iter=100),
            DecisionTreeClassifier(max_depth=3),
        ],
        metric="roc_auc"
    )

    automl.fit(X, y)

    assert automl.best_model_ is not None
    assert isinstance(automl.best_score_, float)

# %%
def test_automl_classifier_metric():
    X, y = load_iris(return_X_y=True, as_frame=True)

    model = AutoMLLightgbm(max_trials=3)
    model.fit(X, y)

    acc = accuracy_score(y, model.predict(X))
    assert acc > 0.7

#%%
def test_automl_api_contract():
    model = AutoMLLightgbm()

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

#%%
def test_model_serialization(tmp_path):
    model = AutoMLLightgbm()
    joblib.dump(model, tmp_path / "model.pkl")
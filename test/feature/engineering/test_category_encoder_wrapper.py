import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dstoolkit.feature.engineering import CategoryEncoderWrapper


class DummyCategoryEncoder:
    def __init__(self, cols=None):
        self.cols = cols
        self.categories_ = {}

    def fit(self, X, y=None):
        for col in self.cols:
            self.categories_[col] = sorted(X[col].unique())
        return self

    def transform(self, X):
        outputs = []

        for col in self.cols:
            cats = self.categories_[col]
            encoded = np.zeros((X.shape[0], len(cats)))

            for i, cat in enumerate(cats):
                encoded[:, i] = (X[col] == cat).astype(int)

            outputs.append(encoded)

        return np.hstack(outputs)

    def get_feature_names_out(self):
        names = []
        for col, cats in self.categories_.items():
            for cat in cats:
                names.append(f"DummyCategoryEncoder__{col}_{cat}")
        return names

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "color": ["red", "blue", "red", "green"],
        "size": ["S", "M", "S", "L"]
    })

@pytest.fixture
def encoder_wrapper():
    return CategoryEncoderWrapper(
        encoder_cls=DummyCategoryEncoder,
        cols=["color", "size"]
    )

def test_fit_transform_basic(sample_data, encoder_wrapper):
    encoder_wrapper.fit(sample_data)
    X_transformed = encoder_wrapper.transform(sample_data)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == sample_data.shape[0]
    assert X_transformed.shape[1] > 0

def test_fit_transform_method(sample_data, encoder_wrapper):
    X_transformed = encoder_wrapper.fit_transform(sample_data)

    assert X_transformed.shape[0] == sample_data.shape[0]

def test_transform_consistent_shape(sample_data, encoder_wrapper):
    encoder_wrapper.fit(sample_data)

    X1 = encoder_wrapper.transform(sample_data)
    X2 = encoder_wrapper.transform(sample_data)

    assert X1.shape == X2.shape

def test_get_feature_names_out(sample_data, encoder_wrapper):
    encoder_wrapper.fit(sample_data)

    feature_names = encoder_wrapper.get_feature_names_out()

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0

    for name in feature_names:
        assert name.startswith("DummyCategoryEncoder__")

def test_works_inside_pipeline(sample_data):
    pipeline = Pipeline(
        steps=[
            (
                "encoder",
                CategoryEncoderWrapper(
                    encoder_cls=DummyCategoryEncoder,
                    cols=["color", "size"]
                )
            ),
            ("scaler", StandardScaler())
        ]
    )

    X_transformed = pipeline.fit_transform(sample_data)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == sample_data.shape[0]

def test_transform_before_fit_raises(sample_data):
    wrapper = CategoryEncoderWrapper(
        encoder_cls=DummyCategoryEncoder,
        cols=["color"]
    )

    with pytest.raises(AttributeError):
        wrapper.transform(sample_data)
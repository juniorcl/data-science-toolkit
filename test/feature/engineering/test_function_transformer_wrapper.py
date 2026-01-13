import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from dstoolkit.feature.engineering import FunctionTransformerWrapper


@pytest.fixture
def X_numpy():
    return np.array([[1, 2], [3, 4]])

@pytest.fixture
def X_pandas():
    return pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])

def test_initialization():
    wrapper = FunctionTransformerWrapper(
        func=np.square,
        inverse_func=np.sqrt,
        feature_suffix="squared"
    )

    assert wrapper.func is np.square
    assert wrapper.inverse_func is np.sqrt
    assert wrapper.feature_suffix == "squared"
    assert isinstance(wrapper.transformer, FunctionTransformer)

def test_fit_returns_self(X_numpy):
    wrapper = FunctionTransformerWrapper(func=np.square)
    result = wrapper.fit(X_numpy)

    assert result is wrapper

def test_transform_applies_function(X_numpy):
    wrapper = FunctionTransformerWrapper(func=np.square)
    wrapper.fit(X_numpy)

    Xt = wrapper.transform(X_numpy)

    expected = np.array([[1, 4], [9, 16]])
    np.testing.assert_array_equal(Xt, expected)

def test_transform_identity_when_func_is_none(X_numpy):
    wrapper = FunctionTransformerWrapper()
    wrapper.fit(X_numpy)

    Xt = wrapper.transform(X_numpy)

    np.testing.assert_array_equal(Xt, X_numpy)

def test_inverse_function_is_passed_correctly(X_numpy):
    wrapper = FunctionTransformerWrapper(
        func=np.square,
        inverse_func=np.sqrt
    )
    wrapper.fit(X_numpy)

    Xt = wrapper.transform(X_numpy)
    X_inv = wrapper.transformer.inverse_transform(Xt)

    np.testing.assert_allclose(X_inv, X_numpy)

def test_get_feature_names_with_suffix():
    wrapper = FunctionTransformerWrapper(feature_suffix="log")

    input_features = ["f1", "f2"]
    output_features = wrapper.get_feature_names_out(input_features)

    assert output_features == ["f1_log", "f2_log"]

def test_get_feature_names_without_suffix():
    wrapper = FunctionTransformerWrapper()

    input_features = ["f1", "f2"]
    output_features = wrapper.get_feature_names_out(input_features)

    assert output_features == ["f1", "f2"]

def test_get_feature_names_none_input():
    wrapper = FunctionTransformerWrapper(feature_suffix="x")

    assert wrapper.get_feature_names_out(None) is None

def test_transform_with_pandas_dataframe(X_pandas):
    wrapper = FunctionTransformerWrapper(func=np.square)
    wrapper.fit(X_pandas)

    Xt = wrapper.transform(X_pandas)

    expected = X_pandas.values ** 2
    np.testing.assert_array_equal(Xt, expected)

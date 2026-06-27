import pytest
import numpy as np
import pandas as pd
from dstoolkit.feature.selection.feature_selector import FeatureSelector


@pytest.fixture
def X_numpy():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def X_pandas():
    return pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])


class TestInit:
    def test_default_parameters(self):
        sel = FeatureSelector(features=["a", "b"])
        assert sel.features == ["a", "b"]
        assert sel.mask is None
        assert sel.check_missing is True

    def test_features_none_and_mask_none_raises(self):
        with pytest.raises(ValueError, match="Either 'features' or 'mask' must be provided"):
            FeatureSelector().fit(np.array([[1, 2]]))

    def test_both_features_and_mask_raises(self):
        with pytest.raises(ValueError, match="Only one of 'features' or 'mask' should be provided"):
            FeatureSelector(features=[0, 1], mask=[True, False]).fit(np.array([[1, 2]]))


class TestFit:
    def test_fit_returns_self(self, X_pandas):
        sel = FeatureSelector(features=["a", "b"])
        result = sel.fit(X_pandas)
        assert result is sel

    def test_fit_pandas_sets_feature_names(self, X_pandas):
        sel = FeatureSelector(features=["a"])
        sel.fit(X_pandas)
        np.testing.assert_array_equal(sel.feature_names_in_, np.array(["a", "b", "c"], dtype=object))
        assert sel.n_features_in_ == 3

    def test_fit_numpy_generates_synthetic_names(self, X_numpy):
        sel = FeatureSelector(features=[0])
        sel.fit(X_numpy)
        np.testing.assert_array_equal(sel.feature_names_in_, np.array(["x0", "x1", "x2"], dtype=object))
        assert sel.n_features_in_ == 3

    def test_select_by_names_pandas(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        assert sel.selected_features_ == ["a", "c"]

    def test_select_by_indices_pandas(self, X_pandas):
        sel = FeatureSelector(features=[0, 2])
        sel.fit(X_pandas)
        assert sel.selected_features_ == [0, 2]

    def test_select_by_indices_numpy(self, X_numpy):
        sel = FeatureSelector(features=[0, 2])
        sel.fit(X_numpy)
        assert sel.selected_features_ == [0, 2]

    def test_select_by_synthetic_names_numpy(self, X_numpy):
        sel = FeatureSelector(features=["x0", "x2"])
        sel.fit(X_numpy)
        assert sel.selected_features_ == [0, 2]

    def test_string_features_numpy_no_synthetic_match_raises(self):
        sel = FeatureSelector(features=["foo", "bar"])
        with pytest.raises(ValueError, match="String feature names cannot be mapped"):
            sel.fit(np.array([[1, 2, 3]]))

    def test_select_by_mask_pandas(self, X_pandas):
        sel = FeatureSelector(mask=[True, False, True])
        sel.fit(X_pandas)
        assert sel.selected_features_ == ["a", "c"]

    def test_select_by_mask_numpy(self, X_numpy):
        sel = FeatureSelector(mask=[True, False, True])
        sel.fit(X_numpy)
        assert sel.selected_features_ == [0, 2]

    def test_mask_length_mismatch_raises(self, X_pandas):
        sel = FeatureSelector(mask=[True, False])
        with pytest.raises(ValueError, match="Mask length"):
            sel.fit(X_pandas)

    def test_check_missing_detects_absent_columns(self, X_pandas):
        sel = FeatureSelector(features=["a", "z"], check_missing=True)
        with pytest.raises(ValueError, match="do not exist"):
            sel.fit(X_pandas)

    def test_check_missing_false_ignores_absent_columns(self, X_pandas):
        sel = FeatureSelector(features=["a", "z"], check_missing=False)
        sel.fit(X_pandas)
        assert "z" in sel.selected_features_

    def test_mask_with_bool_numpy_array(self, X_pandas):
        sel = FeatureSelector(mask=np.array([True, False, True]))
        sel.fit(X_pandas)
        assert sel.selected_features_ == ["a", "c"]


class TestTransform:
    def test_transform_pandas_by_name(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        result = sel.transform(X_pandas)
        expected = pd.DataFrame([[1, 3], [4, 6]], columns=["a", "c"])
        pd.testing.assert_frame_equal(result, expected)

    def test_transform_pandas_by_index(self, X_pandas):
        sel = FeatureSelector(features=[0, 2])
        sel.fit(X_pandas)
        result = sel.transform(X_pandas)
        expected = pd.DataFrame([[1, 3], [4, 6]], columns=["a", "c"])
        pd.testing.assert_frame_equal(result, expected)

    def test_transform_numpy(self, X_numpy):
        sel = FeatureSelector(features=[0, 2])
        sel.fit(X_numpy)
        result = sel.transform(X_numpy)
        np.testing.assert_array_equal(result, np.array([[1, 3], [4, 6]]))

    def test_transform_not_fitted_raises(self, X_numpy):
        sel = FeatureSelector(features=[0])
        with pytest.raises(Exception):
            sel.transform(X_numpy)

    def test_transform_numpy_from_pandas_fit_with_strings(self):
        X_pd = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pd)
        result = sel.transform(np.array([[10, 20, 30], [40, 50, 60]]))
        np.testing.assert_array_equal(result, np.array([[10, 30], [40, 60]]))


class TestInverseTransform:
    def test_inverse_transform_pandas_by_name(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        Xt = sel.transform(X_pandas)
        result = sel.inverse_transform(Xt)
        assert result["a"].tolist() == [1, 4]
        assert result["c"].tolist() == [3, 6]
        assert result["b"].isna().all()

    def test_inverse_transform_pandas_by_index(self, X_pandas):
        sel = FeatureSelector(features=[0, 2])
        sel.fit(X_pandas)
        Xt = sel.transform(X_pandas)
        result = sel.inverse_transform(Xt)
        assert result["a"].tolist() == [1, 4]
        assert result["c"].tolist() == [3, 6]
        assert result["b"].isna().all()

    def test_inverse_transform_numpy(self, X_numpy):
        sel = FeatureSelector(features=[0, 2])
        sel.fit(X_numpy)
        Xt = sel.transform(X_numpy)
        result = sel.inverse_transform(Xt)
        expected = np.array([[1.0, np.nan, 3.0], [4.0, np.nan, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_inverse_transform_not_fitted_raises(self):
        sel = FeatureSelector(features=[0])
        with pytest.raises(Exception):
            sel.inverse_transform(np.array([[1, 2]]))


class TestGetFeatureNamesOut:
    def test_with_string_names(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        result = sel.get_feature_names_out()
        np.testing.assert_array_equal(result, np.array(["a", "c"], dtype=object))

    def test_with_numeric_indices(self, X_numpy):
        sel = FeatureSelector(features=[0, 2])
        sel.fit(X_numpy)
        result = sel.get_feature_names_out()
        np.testing.assert_array_equal(result, np.array(["x0", "x2"], dtype=object))

    def test_not_fitted_raises(self):
        sel = FeatureSelector(features=["a"])
        with pytest.raises(Exception):
            sel.get_feature_names_out()


class TestGetSupport:
    def test_get_support_mask_with_strings(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        result = sel.get_support()
        np.testing.assert_array_equal(result, np.array([True, False, True]))

    def test_get_support_mask_with_indices(self, X_numpy):
        sel = FeatureSelector(features=[0, 2])
        sel.fit(X_numpy)
        result = sel.get_support()
        np.testing.assert_array_equal(result, np.array([True, False, True]))

    def test_get_support_indices(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        result = sel.get_support(indices=True)
        np.testing.assert_array_equal(result, np.array([0, 2]))

    def test_not_fitted_raises(self):
        sel = FeatureSelector(features=["a"])
        with pytest.raises(Exception):
            sel.get_support()


class TestProperties:
    def test_features_property(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        assert sel.features_ == ["a", "c"]

    def test_len(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        assert len(sel) == 2

    def test_repr_fitted(self, X_pandas):
        sel = FeatureSelector(features=["a", "c"])
        sel.fit(X_pandas)
        assert repr(sel) == "FeatureSelector(n_features=2)"

    def test_repr_not_fitted(self):
        sel = FeatureSelector(features=["a", "c"])
        assert repr(sel) == "FeatureSelector(n_features=Not Fitted)"

    def test_sklearn_pipeline_compatible(self, X_numpy):
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([
            ("selector", FeatureSelector(features=[0, 2])),
        ])
        pipe.fit(X_numpy)
        result = pipe.transform(X_numpy)
        np.testing.assert_array_equal(result, np.array([[1, 3], [4, 6]]))

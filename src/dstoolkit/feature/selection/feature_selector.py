from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select a subset of features from a DataFrame or NumPy array.

    Parameters
    ----------
    features : Iterable[str] or Iterable[int], default=None
        List of column names (for pandas) or column indices (for numpy/pandas) 
        to select.

    mask : Iterable[bool], default=None
        Boolean mask of length n_features_in_ indicating which features to select.

    check_missing : bool, default=True
        If True, validates whether the requested features exist in the input data.
        Only applicable when `features` are provided as strings.

    Attributes
    ----------
    selected_features_ : list
        List of selected feature names (strings) or indices (ints).

    n_features_in_ : int
        Number of features seen during `fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names (e.g., a pandas DataFrame) or generated as string indices for numpy.
    """

    def __init__(
        self, 
        features: Iterable[str | int] | None = None, 
        mask: Iterable[bool] | None = None, 
        check_missing: bool = True
    ):
        self.features = features
        self.mask = mask
        self.check_missing = check_missing

    def fit(self, X, y=None):
        """Fit the feature selector to the input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. This parameter exists only for compatibility with 
            sklearn's Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        if self.features is None and self.mask is None:
            raise ValueError("Either 'features' or 'mask' must be provided.")
        if self.features is not None and self.mask is not None:
            raise ValueError("Only one of 'features' or 'mask' should be provided.")

        
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = X.shape[1]
            is_pandas = True

        else:
            X_arr = np.asarray(X)
            self.n_features_in_ = X_arr.shape[1]
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object) # Generate synthetic names for numpy alignment
            is_pandas = False

        # Case 1: Mask-based selection
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if len(mask) != self.n_features_in_:
                raise ValueError(
                    f"Mask length ({len(mask)}) does not match "
                    f"number of features ({self.n_features_in_})."
                )
            
            # Save selection based on input type
            if is_pandas:
                self.selected_features_ = list(self.feature_names_in_[mask])
            else:
                self.selected_features_ = list(np.where(mask)[0])

        # Case 2: Explicit features list (names or indices)
        else:
            features_list = list(self.features)
            
            # If strings are passed but input is numpy, try to map from synthetic names or indices
            if not is_pandas and all(isinstance(f, str) for f in features_list):
                # If they passed synthetic names like ['x0', 'x2']
                if all(f in self.feature_names_in_ for f in features_list):
                    self.selected_features_ = [list(self.feature_names_in_).index(f) for f in features_list]
                else:
                    raise ValueError("String feature names cannot be mapped to a NumPy array unless they match 'x0', 'x1', etc.")
            else:
                self.selected_features_ = features_list

            # Optional check for missing columns (only makes sense for string names in pandas)
            if self.check_missing and is_pandas and all(isinstance(f, str) for f in self.selected_features_):
                missing = sorted(set(self.selected_features_) - set(self.feature_names_in_))
                if missing:
                    raise ValueError(f"The following features do not exist in X: {missing}")

        return self

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_sliced : {array-like, sparse matrix} of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        check_is_fitted(self, attributes=["selected_features_", "n_features_in_"])

        if isinstance(X, pd.DataFrame):
            # If fit was on pandas or indices are used, pandas .loc/.iloc handles it safely
            if all(isinstance(f, (int, np.integer)) for f in self.selected_features_):
                return X.iloc[:, self.selected_features_]
            return X.loc[:, self.selected_features_]
        
        else:
            X_arr = np.asarray(X)
            # If selected_features_ contains string names (from pandas fit) but X is numpy
            if Janus_indices := [isinstance(f, str) for f in self.selected_features_]:
                if any(Janus_indices):
                    # Map string names back to positions using stored feature_names_in_
                    indices = [list(self.feature_names_in_).index(f) for f in self.selected_features_]
                    return X_arr[:, indices]
            
            return X_arr[:, self.selected_features_]

    def inverse_transform(self, X):
        """Reverse the transformation, filling unselected features with NaNs.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_selected_features)
            The converted input samples.

        Returns
        -------
        X_original : {array-like, sparse matrix} of shape (n_samples, n_features_in_)
            The original structure filled with NaNs where features were excluded.
        """
        check_is_fitted(self, attributes=["selected_features_", "feature_names_in_"])

        if isinstance(X, pd.DataFrame):
            out = pd.DataFrame(index=X.index, columns=self.feature_names_in_)
            if all(isinstance(f, (int, np.integer)) for f in self.selected_features_):
                cols = self.feature_names_in_[self.selected_features_]
                out[cols] = X.values
            else:
                out[self.selected_features_] = X
            return out
        
        else:
            X_arr = np.asarray(X)
            out = np.full((X_arr.shape[0], self.n_features_in_), np.nan, dtype=float)
            
            # Map features to numeric indices for array assignment
            if all(isinstance(f, str) for f in self.selected_features_):
                indices = [list(self.feature_names_in_).index(f) for f in self.selected_features_]
            else:
                indices = self.selected_features_
                
            out[:, indices] = X_arr
            return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformation."""
        check_is_fitted(self, attributes=["selected_features_"])
        if all(isinstance(f, (int, np.integer)) for f in self.selected_features_):
            return np.asarray(self.feature_names_in_[self.selected_features_], dtype=object)
        return np.asarray(self.selected_features_, dtype=object)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a mask or index array of the features selected."""
        check_is_fitted(self, attributes=["selected_features_", "feature_names_in_"])
        
        if all(isinstance(f, (int, np.integer)) for f in self.selected_features_):
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[self.selected_features_] = True
        else:
            mask = np.isin(self.feature_names_in_, self.selected_features_)

        if indices:
            return np.where(mask)[0]
        return mask

    @property
    def features_(self) -> list:
        """Backward compatibility helper for selected features."""
        check_is_fitted(self, attributes=["selected_features_"])
        return self.selected_features_

    def __len__(self) -> int:
        check_is_fitted(self, attributes=["selected_features_"])
        return len(self.selected_features_)

    def __repr__(self) -> str:
        try:
            length = len(self)
        except Exception:
            length = "Not Fitted"
        return f"FeatureSelector(n_features={length})"
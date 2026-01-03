from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer


class FunctionTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for sklearn's FunctionTransformer that adds feature name suffixing.

    Parameters
    ----------
    func : callable, optional
        The function to apply to the data during transformation.
    inverse_func : callable, optional
        The function to apply to the data during inverse transformation.
    feature_suffix : str, optional
        A suffix to append to the feature names after transformation.
    **kwargs : additional keyword arguments
        Additional arguments to pass to the FunctionTransformer.

    Attributes
    ----------
    transformer : FunctionTransformer
        The underlying FunctionTransformer instance.
    feature_suffix : str
        The suffix to append to feature names.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data.
    transform(X)
        Transform the data using the specified function.
    get_feature_names_out(input_features=None)
        Get the output feature names with the specified suffix.

    Examples
    --------
    >>> obj = FunctionTransformerWrapper(func=lambda x: x ** 2, feature_suffix='squared')
    >>> obj.fit(X)
    >>> X_transformed = obj.transform(X)
    >>> feature_names = obj.get_feature_names_out(input_features=['feature1', 'feature2'])
    """
    def __init__(self, func=None, inverse_func=None, feature_suffix=None, **kwargs):
        self.func = func
        self.inverse_func = inverse_func
        self.feature_suffix = feature_suffix
        self.kwargs = kwargs
        self.transformer = FunctionTransformer(func=func, inverse_func=inverse_func, **kwargs)

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        suffix = f"_{self.feature_suffix}" if self.feature_suffix else ""
        return [f"{col}{suffix}" for col in input_features]
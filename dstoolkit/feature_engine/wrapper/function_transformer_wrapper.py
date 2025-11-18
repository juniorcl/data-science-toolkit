from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

class FunctionTransformerWrapper(BaseEstimator, TransformerMixin):
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
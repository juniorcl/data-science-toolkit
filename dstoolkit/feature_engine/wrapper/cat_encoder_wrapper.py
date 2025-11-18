from sklearn.base import BaseEstimator, TransformerMixin

class CatEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_cls, cols):
        self.encoder_cls = encoder_cls
        self.cols = cols
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = self.encoder_cls(cols=self.cols)
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

    def get_feature_names_out(self, input_features=None):
        if hasattr(self.encoder, 'get_feature_names_out'):
            return self.encoder.get_feature_names_out()
        return [f"{self.encoder_cls.__name__}__{c}" for c in self.cols]
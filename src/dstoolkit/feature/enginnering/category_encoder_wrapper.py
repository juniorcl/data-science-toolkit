from sklearn.base import BaseEstimator, TransformerMixin


class CategoryEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for category encoders to make them compatible with scikit-learn's
    Transformer API.

    Parameters
    ----------
    encoder_cls : class
        The category encoder class to be wrapped. It should follow the
        scikit-learn transformer interface.
    cols : list
        List of column names to be encoded. If None, all columns will be encoded.

    Attributes
    ----------
    encoder : object
        An instance of the wrapped encoder class.

    Methods
    -------
    fit(X, y=None)
        Fit the encoder to the data.
    transform(X)
        Transform the data using the fitted encoder.
    get_feature_names_out(input_features=None)
        Get output feature names for the transformed data.

    Examples
    --------
    >>> obj = CategoryEncoderWrapper(encoder_cls=SomeCategoryEncoder, cols=['col1', 'col2'])
    >>> obj.fit(X_train, y_train)
    CategoryEncoderWrapper(...)
    >>> X_transformed = obj.transform(X_test)
    >>> feature_names = obj.get_feature_names_out()
    >>> print(feature_names)
    ['SomeCategoryEncoder__col1_A', 'SomeCategoryEncoder__col1_B', 'SomeCategoryEncoder__col2_X', ...]
    """
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
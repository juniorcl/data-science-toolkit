import optuna
import numpy as np
import pandas as pd

from sklearn.manifold import trustworthiness
from sklearn.decomposition import PCA


class PCAOptimizer:
    """
    PCAOptimizer is a class that automates the process of PCA model selection
    and hyperparameter tuning using Optuna. It supports various evaluation metrics.

    Parameters
    ----------
    scoring : str, optional (default='explained_variance')
        The evaluation metric to optimize during hyperparameter tuning.
        Supported metrics include 'explained_variance', 'reconstruction_error', 'trustworthiness'.
    n_trials : int, optional (default=50)
        The number of trials for the hyperparameter optimization process.
    random_state : int, optional (default=42)
        The random seed for reproducibility.

    Attributes
    ----------
    random_state : int
        The random seed for reproducibility.
    model_class : class
        The PCA model class obtained from model_name.
    scorer : str
        The evaluation metric used for scoring.
    func_metric : function
        The function corresponding to the evaluation metric.

    Methods
    -------
    train(X)
        Fits the PCA model to the data X after hyperparameter tuning.
    get_metrics(return_df=True)
        Returns the PCA evaluation metrics.

    Examples
    --------
    >>> obj = AutoMLPCA(scoring='explained_variance', n_trials=20)
    >>> obj.train(X)
    AutoMLPCA(...)
    >>> obj.get_metrics()
               Train
    explained_variance    0.65
    """
    def __init__(self, scoring='explained_variance', n_trials=50, random_state=42):
        self.scoring = scoring
        self.n_trials = n_trials
        self.random_state = random_state

    def _get_reconstruction_error(self, model, X, X_embedding):
        X_recon = model.inverse_transform(X_embedding)
        return -np.mean((X - X_recon)**2)

    def _get_explained_variance(self, model):
        return np.sum(model.explained_variance_ratio_)

    def _get_trustworthiness(self, X, X_embedding, n_neighbors=10):
        return trustworthiness(X, X_embedding, n_neighbors=n_neighbors)

    def _get_best_params(self, X):
        def objective(trial):
            params = {
                "n_components": trial.suggest_int("n_components", 2, X.shape[1] - 1),
                "whiten": trial.suggest_categorical("whiten", [True, False]),
                "svd_solver": trial.suggest_categorical("svd_solver", ["auto", "full", "arpack", "randomized"]),
                "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True),
                "iterated_power": trial.suggest_int("iterated_power", 1, 10),
                "random_state": trial.suggest_categorical("random_state", [self.random_state])
            }

            model = PCA(**params)
            X_embedding = model.fit_transform(X)

            if self.scoring == 'explained_variance':
                return self._get_explained_variance(model)
            elif self.scoring == 'reconstruction_error':
                return self._get_reconstruction_error(model, X, X_embedding)
            elif self.scoring == 'trustworthiness':
                return self._get_trustworthiness(X, X_embedding)
            else:
                raise ValueError("Invalid metric. Use: 'trustworthiness', 'reconstruction_error', 'explained_variance'.")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_trial.params

    def _fit(self, X):
        self.best_params = self._get_best_params(X)
        self.model = PCA(**self.best_params)
        self.X_embedding = self.model.fit_transform(X)

        self.results = {
            'Train': {
                'Explained Variance': self._get_explained_variance(self.model),
                'Reconstruction Error': self._get_reconstruction_error(self.model, X, self.X_embedding),
                'Trustworthiness': self._get_trustworthiness(X, self.X_embedding)
            }
        }
        return self

    def fit(self, X):
        self.X = X
        return self._fit(self.X)

    def get_metrics(self, return_df=True):
        if return_df:
            return pd.DataFrame(self.results).T
        return self.results
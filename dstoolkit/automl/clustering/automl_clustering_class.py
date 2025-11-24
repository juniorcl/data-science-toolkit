import optuna

import pandas as pd

from .params_space import get_params_space
from .model_instance import get_model_instance

from ..utils import get_cluster_eval_scoring, get_cluster_metrics, analyze_clusters


class AutoMLClustering:
    """
    AutoMLClustering is a class that automates the process of clustering model selection
    and hyperparameter tuning using Optuna. It supports various clustering algorithms
    and evaluation metrics.

    Parameters
    ----------
    model_name : str
        The name of the clustering model to be used (e.g., 'KMeans', 'DBSCAN').
    scoring : str, optional (default='silhouette')
        The evaluation metric to optimize during hyperparameter tuning.
        Supported metrics include 'silhouette', 'davies_bouldin_score', etc.
    n_trials : int, optional (default=50)
        The number of trials for the hyperparameter optimization process.
    random_state : int, optional (default=42)
        The random seed for reproducibility.

    Attributes
    ----------
    model_name : str
        The name of the clustering model.
    random_state : int
        The random seed for reproducibility.
    model_class : class
        The clustering model class obtained from model_name.
    scorer : str
        The evaluation metric used for scoring.
    func_metric : function
        The function corresponding to the evaluation metric.

    Methods
    -------
    train(X)
        Fits the clustering model to the data X after hyperparameter tuning.
    get_metrics(return_df=True)
        Returns the clustering evaluation metrics.
    analyze(X_orig)
        Analyzes and visualizes the clustering results.

    Examples
    --------
    >>> obj = AutoMLClustering(model_name='KMeans', scoring='silhouette', n_trials=20)
    >>> obj.train(X)
    AutoMLClustering(...)
    >>> obj.get_metrics()
               Train
    silhouette_score    0.65
    """
    def __init__(self, model_name, scoring='silhouette', n_trials=50, random_state=42):
        self.n_trials = n_trials
        self.model_name = model_name
        self.random_state = random_state
        self.model_class = get_model_instance(model_name)
        self.scorer = get_cluster_eval_scoring(scoring, return_func=False)
        self.func_metric = get_cluster_eval_scoring(scoring, return_func=True)

    def _get_best_params(self, X):
        def objective(trial):
            params = get_params_space(self.model_name, trial, self.random_state)
            model = self.model_class(**params)
            labels = model.fit_predict(X)
            return self.func_metric(X, labels)

        direction = 'minimize' if self.scorer == 'davies_bouldin_score' else 'maximize'
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_trial.params
    
    def _fit(self, X):
        self.best_params = self._get_best_params(X)
        self.model = self.model_class(**self.best_params)
        self.labels = self.model.fit_predict(X)
        self.results = {'Train': get_cluster_metrics(X, self.labels)}
        return self

    def train(self, X):
        self.X = X
        return self._fit(self.X)

    def get_metrics(self, return_df=True):
        if return_df:
            return pd.DataFrame(self.results).T
        return self.results

    def analyze(self, X_orig):
        analyze_clusters(X_orig, self.X, self.labels)
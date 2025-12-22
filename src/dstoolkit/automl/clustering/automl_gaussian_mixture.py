import optuna
import pandas as pd

from . import utils
from sklearn.mixture import GaussianMixture


class AutoMLGaussianMixture:
    """
    AutoMLGaussianMixture is a class that automates the process of Gaussian Mixture clustering model selection
    and hyperparameter tuning using Optuna. It supports various clustering algorithms
    and evaluation metrics.

    Parameters
    ----------
    scoring : str, optional (default='silhouette')
        The evaluation metric to optimize during hyperparameter tuning.
        Supported metrics include 'silhouette', 'davies_bouldin_score', etc.
    n_trials : int, optional (default=50)
        The number of trials for the hyperparameter optimization process.
    random_state : int, optional (default=42)
        The random seed for reproducibility.

    Attributes
    ----------
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
    >>> obj = AutoMLKMeans(scoring='silhouette', n_trials=20)
    >>> obj.train(X)
    AutoMLKMeans(...)
    >>> obj.get_metrics()
               Train
    silhouette_score    0.65
    """
    def __init__(self, scoring='silhouette', n_trials=50, random_state=42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.scorer = utils.get_cluster_score(scoring)
        self.func_metric = utils.get_cluster_function_score(scoring)

    def _get_best_params(self, X):
        def objective(trial):
            params = utils.get_gaussian_mixture_params_space(trial, self.random_state)
            model = GaussianMixture(**params)
            labels = model.fit_predict(X)
            return self.func_metric(X, labels)

        direction = 'minimize' if self.scorer == 'davies_bouldin_score' else 'maximize'
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_trial.params
    
    def _fit(self, X):
        self.best_params = self._get_best_params(X)
        self.model = GaussianMixture(**self.best_params)
        self.labels = self.model.fit_predict(X)
        self.results = {'Train': utils.get_cluster_metrics(X, self.labels)}
        return self

    def fit(self, X):
        self.X = X
        return self._fit(self.X)

    def get_metrics(self, return_df=True):
        if return_df:
            return pd.DataFrame(self.results).T
        return self.results

    def analyze(self, X_orig):
        utils.plot_cluster_sizes(self.labels)
        utils.plot_silhouette_analysis(self.X, self.labels)
        utils.plot_kdeplots_by_cluster(X_orig, self.labels)
        utils.plot_tree_ovr(X_orig, self.labels)
        utils.plot_cluster_pca(self.X, self.labels)
        utils.plot_cluster_umap(self.X, self.labels)
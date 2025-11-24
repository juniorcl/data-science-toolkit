import optuna
import inspect

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate

from .params_space import get_params_space
from .model_instance import get_model_instance

from ..utils import get_classifier_eval_scoring, get_classifier_metrics, analyze_classifier, ks_scorer


class AutoMLClassifierCV:
    """
    Automated Machine Learning Classifier with Hyperparameter Tuning. 
    This class provides an interface to train and evaluate classification models
    with optional hyperparameter tuning using Optuna.

    Parameters
    ----------
    model_name : str
        The name of the classification model to use.
    scoring : str, optional
        The evaluation metric to use for model selection. Default is 'roc_auc'.
    tune : bool, optional
        Whether to perform hyperparameter tuning. Default is False.
    n_trials : int, optional
        The number of Optuna trials to run for hyperparameter tuning. Default is 50.
    random_state : int, optional
        The random seed for reproducibility. Default is 42.
    cv : int or cross-validation generator, optional
        The cross-validation strategy to use for hyperparameter tuning and evaluation. Default is 3.
                
    Attributes
    ----------
    model : object
        The trained classification model.
    best_params : dict
        The best hyperparameters found during tuning.
    results : dict
        A dictionary containing evaluation metrics for training, validation, and test sets.
    scorer : callable
        The scoring function used for evaluation.
    func_metric : callable
        The metric function used for optimization during tuning.
    cv : int or cross-validation generator
        The cross-validation strategy used.
    X_train : pd.DataFrame
        The training feature set.
    y_train : pd.DataFrame
        The training target set.
    X_test : pd.DataFrame
        The test feature set.
    y_test : pd.DataFrame
        The test target set.

    Methods
    -------
    train(X_train, y_train, X_test, y_test, target='target')
        Trains the model on the provided datasets.
    get_metrics(return_df=True)
        Returns the evaluation metrics as a DataFrame or dictionary.
    analyze()
        Analyzes the trained model and generates performance plots.

    Examples
    --------
    >>> obj = AutoMLClassifier(model_name='RandomForest', scoring='roc_auc', tune=True, n_trials=30)
    >>> obj.train(X_train, y_train, X_valid, y_valid, X_test, y_test, target='target')
    >>> metrics_df = obj.get_metrics(return_df=True)
    >>> print(metrics_df)
                accuracy    roc_auc  f1_score
    Train       0.95       0.98      0.94
    Test        0.93       0.96      0.92
    >>> obj.analyze()
    >>> # Example of accessing best parameters
    >>> print(obj.best_params)
    {'n_estimators': 100, 'max_depth': 10, ...}
    >>> # Example of using the trained model for predictions
    >>> predictions = obj.model.predict(X_test)
    """
    def __init__(self, model_name, scoring='roc_auc', cv=3, tune=False, n_trials=50, random_state=42):
        self.cv = cv
        self.tune = tune
        self.n_trials = n_trials
        self.model_name = model_name
        self.random_state = random_state
        self.model_class = get_model_instance(model_name)
        self.scorer = get_classifier_eval_scoring(scoring, return_func=False)
        self.func_metric = get_classifier_eval_scoring(scoring, return_func=True)

    def _cross_validate(self, model):
        cv_results = cross_validate(
            estimator=model,
            X=self.X_train,
            y=self.y_train[self.target],
            cv=self.cv,
            scoring={
                'balanced_accuracy': 'balanced_accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc',
                'ks': ks_scorer,
                'brier': 'neg_brier_score',
                'log_loss': 'neg_log_loss'
            }
        )
        return {
            'Balanced Accuracy': cv_results['test_balanced_accuracy'].mean(),
            'Precision': cv_results['test_precision'].mean(),
            'Recall': cv_results['test_recall'].mean(),
            'F1': cv_results['test_f1'].mean(),
            'AUC': cv_results['test_roc_auc'].mean(),
            'KS': cv_results['test_ks'].mean(),
            'Brier': abs(cv_results['test_brier'].mean()),
            'LogLoss': abs(cv_results['test_log_loss'].mean())
        }

    def _get_best_params(self):
        def objective(trial):
            params = get_params_space(self.model_name, trial, self.random_state)
            model = self.model_class(**params)
            cv_results = cross_validate(
                estimator=model,
                X=self.X_train,
                y=self.y_train[self.target],
                cv=self.cv,
                scoring=self.scorer
            )
            return np.mean(cv_results['test_score'])

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def _fit(self):
        model_sig = inspect.signature(self.model_class).parameters
        init_params = {}

        if 'random_state' in model_sig:
            init_params['random_state'] = self.random_state
        if 'verbose' in model_sig:
            init_params['verbose'] = 0
        if 'verbosity' in model_sig:
            init_params['verbosity'] = -1
        if 'n_jobs' in model_sig:
            init_params['n_jobs'] = -1

        self.best_params = self._get_best_params() if self.tune else init_params
        self.model = self.model_class(**self.best_params)

        self.results = {'Train CV': self._cross_validate(self.model)}

        self.model.fit(self.X_train, self.y_train[self.target])
        self.y_test['pred'] = self.model.predict(self.X_test)
        self.y_test['prob'] = self.model.predict_proba(self.X_test)[:, 1]

        self.results['Test'] = get_classifier_metrics(self.y_test, target=self.target, pred_col='pred', prob_col='prob')
        return self.model, self.results

    def train(self, X_train, y_train, X_test, y_test, target='target'):
        self.target = target
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.model, self.results = self._fit()
        return self

    def get_metrics(self, return_df=True):
        if return_df:
            return pd.DataFrame(self.results).T
        return self.results

    def analyze(self):
        analyze_classifier(self.model, self.X_train, self.y_train, self.y_test, self.target, self.scorer)
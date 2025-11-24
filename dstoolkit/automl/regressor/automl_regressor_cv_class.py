import optuna
import inspect

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate

from .params_space import get_params_space
from .model_instance import get_model_instance

from ..utils import get_regressor_eval_scoring, get_regressor_metrics, analyze_regressor


class AutoMLRegressorCV:
    """
    AutoMLRegressorCV is a class that automates the process of training and tuning regression models with cross-validation.
    It supports hyperparameter optimization using Optuna and provides evaluation metrics for the trained models.

    Parameters
    ----------
    model_name : str
        The name of the regression model to use.
    target : str
        The name of the target variable in the training data.
    scoring : str
        The evaluation metric to use for model selection.
    tune : bool, optional
        Whether to perform hyperparameter tuning. Default is False.
    n_trials : int, optional
        The number of trials to run for hyperparameter tuning. Default is 50.
    random_state : int, optional
        The random seed for reproducibility. Default is 42.
    cv : int, optional
        The number of cross-validation folds. Default is 3.

    Attributes
    ----------
    model_class : type
        The class of the regression model to use.
    scorer : callable
        The function to use for evaluating model performance.
    func_metric : callable
        The function to use for computing the evaluation metric.

    Methods
    -------
    train(X_train, y_train, X_test, y_test, target='target')
        Trains the model on the provided training data and evaluates it on test data.
    get_metrics(return_df=True)
        Returns the evaluation metrics for the trained model.
    analyze()
        Analyzes the trained model and provides visualizations.

    Examples
    --------
    >>> obj = AutoMLRegressor(model_name='RandomForest', target='price', scoring='neg_mean_squared_error', tune=True, n_trials=100, random_state=123)
    >>> obj.train(X_train, y_train, X_valid, y_valid, X_test, y_test, target='price')
    >>> metrics_df = obj.get_metrics(return_df=True)
    >>> print(metrics_df)
                r2    neg_mean_squared_error
    Train  0.85               -2500.0
    Valid  0.80               -3000.0
    Test   0.78               -3200.0
    """
    def __init__(self, model_name, target='target', scoring='r2', cv=3, tune=False, n_trials=50, random_state=42):
        self.cv = cv
        self.tune = tune
        self.target = target
        self.n_trials = n_trials
        self.model_name = model_name
        self.random_state = random_state
        self.model_class = get_model_instance(model_name)
        self.scorer = get_regressor_eval_scoring(scoring, return_func=False)
        self.func_metric = get_regressor_eval_scoring(scoring, return_func=True)

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
    
    def _cross_validate(self, model):
        cv_results = cross_validate(
            estimator=model, 
            X=self.X_train, 
            y=self.y_train[self.target],
            cv=self.cv,
            scoring=(
                'r2', 
                'neg_mean_absolute_error', 
                'neg_median_absolute_error', 
                'neg_mean_absolute_percentage_error', 
                'neg_root_mean_squared_error', 
                'explained_variance'
            )
        )
        return {
            'R2': cv_results['test_r2'].mean(),
            'MAE': np.abs(cv_results['test_neg_mean_absolute_error'].mean()),
            'MadAE': np.abs(cv_results['test_neg_median_absolute_error'].mean()),
            'MAPE': np.abs(cv_results['test_neg_mean_absolute_percentage_error'].mean()),
            'RMSE': np.abs(cv_results['test_neg_root_mean_squared_error'].mean()),
            'Explained Variance': cv_results['test_explained_variance'].mean()
        }
    
    def _fit(self):
        model_sig = inspect.signature(self.model_class).parameters
        init_params = {}
        if 'random_state' in model_sig:
            init_params['random_state'] = self.random_state
        if 'verbose' in model_sig:
            init_params['verbose'] = 0
        elif 'verbosity' in model_sig:
            init_params['verbosity'] = -1

        params = self._get_best_params() if self.tune else init_params
        model = self.model_class(**params)

        results = {'Train CV': self._cross_validate(model)}
        model.fit(self.X_train, self.y_train[self.target])
        self.y_test['pred'] = model.predict(self.X_test)
        results['Test'] = get_regressor_metrics(self.y_test, target=self.target, pred_col='pred')
        return model, results

    def train(self, X_train, y_train, X_test, y_test, target='target'):
        self.target = target
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.model, self.results = self._fit()

    def get_metrics(self, return_df=True):
        return pd.DataFrame(self.results).T if return_df else self.results
    
    def analyze(self):
        analyze_regressor(self.model, self.X_train, self.y_train, self.y_test, self.target, self.scorer)
import optuna
import pandas as pd

from . import utils
from lightgbm import LGBMRegressor


class AutoMLLightGBM:
    """
    AutoMLRegressor is a class that automates the process of training and tuning regression models. 
    It supports hyperparameter optimization using Optuna and provides evaluation metrics for the trained models.

    Parameters
    ----------
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
    train(X_train, y_train, X_valid, y_valid, X_test, y_test, target='target')
        Trains the model on the provided training data and evaluates it on validation and test data.
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
    def __init__(self, target='target', scoring='r2', tune=False, n_trials=50, random_state=42):
        self.tune = tune
        self.target = target
        self.n_trials = n_trials
        self.random_state = random_state
        self.scorer = utils.get_regressor_score(scoring)
        self.func_metric = utils.get_regressor_function_score(scoring)

    def _get_best_params(self):
        def objective(trial):
            params = utils.get_lightgbm_params_space(trial, self.random_state)
            model = LGBMRegressor(**params)
            model.fit(self.X_train, self.y_train[self.target])
            preds = model.predict(self.X_valid)
            return self.func_metric(self.y_valid[self.target], preds)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params
    
    def _fit(self):
        params = self._get_best_params() if self.tune else {"random_state": self.random_state, "verbosity": -1}
        model = LGBMRegressor(**params)

        model.fit(self.X_train, self.y_train[self.target], eval_set=[(self.X_valid, self.y_valid[self.target])])

        self.y_train['pred'] = model.predict(self.X_train)
        self.y_valid['pred'] = model.predict(self.X_valid)
        self.y_test['pred'] = model.predict(self.X_test)

        results = {
            'Train': utils.get_regressor_metrics(self.y_train, target=self.target, pred_col='pred'),
            'Valid': utils.get_regressor_metrics(self.y_valid, target=self.target, pred_col='pred'),
            'Test': utils.get_regressor_metrics(self.y_test, target=self.target, pred_col='pred')
        }
        return model, results

    def train(self, X_train, y_train, X_valid, y_valid, X_test, y_test, target='target'):
        self.target = target
        self.X_train, self.X_valid, self.X_test = X_train, X_valid, X_test
        self.y_train, self.y_valid, self.y_test = y_train, y_valid, y_test
        self.model, self.results = self._fit()

    def get_metrics(self, return_df=True):
        if return_df:
            return pd.DataFrame(self.results).T
        return self.results

    def analyze(self):
        utils.plot_residuals(self.y_test, 'pred', self.target)
        utils.plot_pred_vs_true(self.y_test, 'pred', self.target)
        utils.plot_error_by_quantile(self.y_test, 'pred', self.target)
        utils.plot_learning_curve(self.model, self.X_train, self.y_train[self.target], scoring=self.scorer)
        utils.plot_feature_importance(self.model)
        utils.plot_permutation_importance(self.model, self.X_train, self.y_train[self.target], scoring=self.scorer)
        utils.plot_shap_summary(self.model, self.X_train)
import optuna
import pandas as pd

from . import utils
from catboost import CatBoostClassifier


class AutoMLCatBoost:
    """
    Automated Machine Learning Classifier with Hyperparameter Tuning. 
    This class provides an interface to train and evaluate classification models
    with optional hyperparameter tuning using Optuna.

    Parameters
    ----------
    scoring : str, optional
        The evaluation metric to use for model selection. Default is 'roc_auc'.
    tune : bool, optional
        Whether to perform hyperparameter tuning. Default is False.
    n_trials : int, optional
        The number of Optuna trials to run for hyperparameter tuning. Default is 50.
    random_state : int, optional
        The random seed for reproducibility. Default is 42.
                
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
    X_train : pd.DataFrame
        The training feature set.
    y_train : pd.DataFrame
        The training target set.
    X_valid : pd.DataFrame
        The validation feature set.
    y_valid : pd.DataFrame
        The validation target set.
    X_test : pd.DataFrame
        The test feature set.
    y_test : pd.DataFrame
        The test target set.

    Methods
    -------
    train(X_train, y_train, X_valid, y_valid, X_test, y_test, target='target')
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
    Valid       0.92       0.95      0.91
    Test        0.93       0.96      0.92
    >>> obj.analyze()
    >>> # Example of accessing best parameters
    >>> print(obj.best_params)
    {'n_estimators': 100, 'max_depth': 10, ...}
    >>> # Example of using the trained model for predictions
    >>> predictions = obj.model.predict(X_test)
    """
    def __init__(self, scoring='roc_auc', tune=False, n_trials=50, random_state=42):
        self.tune = tune
        self.n_trials = n_trials
        self.random_state = random_state
        self.scorer = utils.get_classifier_score(scoring)
        self.func_metric = utils.get_classifier_function_score(scoring)

    def _get_best_params(self):
        def objective(trial):
            params = utils.get_catboost_params_space(trial, self.random_state)
            model = CatBoostClassifier(**params)
            model.fit(self.X_train, self.y_train[self.target])
            probs = model.predict_proba(self.X_valid)[:, 1]
            return self.func_metric(self.y_valid[self.target], probs)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def _fit(self):
        self.best_params = self._get_best_params() if self.tune else {"random_state": self.random_state, "verbose": 0}
        self.model = CatBoostClassifier(**self.best_params)

        self.model.fit(
            X=self.X_train, y=self.y_train[self.target], 
            eval_set=[(self.X_valid, self.y_valid[self.target])]
        )

        for X, y in [(self.X_train, self.y_train), (self.X_valid, self.y_valid), (self.X_test, self.y_test)]:
            y['pred'] = self.model.predict(X)
            y['prob'] = self.model.predict_proba(X)[:, 1]

        self.results = {
            'Train': utils.get_classifier_metrics(self.y_train, target=self.target, pred_col='pred', prob_col='prob'),
            'Valid': utils.get_classifier_metrics(self.y_valid, target=self.target, pred_col='pred', prob_col='prob'),
            'Test': utils.get_classifier_metrics(self.y_test, target=self.target, pred_col='pred', prob_col='prob')
        }
        return self.model, self.results

    def train(self, X_train, y_train, X_valid, y_valid, X_test, y_test, target='target'):
        self.target = target
        self.X_train, self.X_valid, self.X_test = X_train, X_valid, X_test
        self.y_train, self.y_valid, self.y_test = y_train, y_valid, y_test
        self.model, self.results = self._fit()
        return self

    def get_metrics(self, return_df=True):
        if return_df:
            return pd.DataFrame(self.results).T
        return self.results

    def analyze(self):
        utils.plot_roc_curve(self.y_test, self.target, 'prob')
        utils.plot_ks_curve(self.y_test, self.target)
        utils.plot_precision_recall_curve(self.y_test, self.target, 'prob')
        utils.plot_calibration_curve(self.y_test, self.target, strategy='uniform')
        utils.plot_learning_curve(self.model, self.X_train, self.y_train[self.target], scoring=self.scorer)
        utils.plot_feature_importance(self.model)
        utils.plot_permutation_importance(self.model, self.X_train, self.y_train[self.target], scoring=self.scorer)
        utils.plot_shap_summary(self.model, self.X_train)
import optuna
import inspect

import pandas as pd

from .params_space import get_params_space
from .model_instance import get_model_instance

from ..utils import get_classifier_eval_scoring, get_classifier_metrics, analyze_classifier

class AutoMLClassifier:
    def __init__(self, model_name, scoring='roc_auc', tune=False, n_trials=50, random_state=42):
        self.tune = tune
        self.n_trials = n_trials
        self.model_name = model_name
        self.random_state = random_state
        self.model_class = get_model_instance(model_name)
        self.scorer = get_classifier_eval_scoring(scoring, return_func=False)
        self.func_metric = get_classifier_eval_scoring(scoring, return_func=True)

    def _get_best_params(self):
        def objective(trial):
            params = get_params_space(self.model_name, trial, self.random_state)
            model = self.model_class(**params)
            model.fit(self.X_train, self.y_train[self.target])
            probs = model.predict_proba(self.X_valid)[:, 1]
            return self.func_metric(self.y_valid[self.target], probs)

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

        fit_sig = inspect.signature(self.model.fit).parameters
        fit_args = {'X': self.X_train, 'y': self.y_train[self.target]}

        if 'eval_set' in fit_sig and self.X_valid is not None:
            fit_args['eval_set'] = [(self.X_valid, self.y_valid[self.target])]
        if 'early_stopping_rounds' in fit_sig:
            fit_args['early_stopping_rounds'] = 50

        self.model.fit(**fit_args)

        for X, y in [(self.X_train, self.y_train), (self.X_valid, self.y_valid), (self.X_test, self.y_test)]:
            y['pred'] = self.model.predict(X)
            y['prob'] = self.model.predict_proba(X)[:, 1]

        self.results = {
            'Train': get_classifier_metrics(self.y_train, target=self.target, pred_col='pred', prob_col='prob'),
            'Valid': get_classifier_metrics(self.y_valid, target=self.target, pred_col='pred', prob_col='prob'),
            'Test': get_classifier_metrics(self.y_test, target=self.target, pred_col='pred', prob_col='prob')
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
        analyze_classifier(self.model, self.X_train, self.y_train, self.y_test, self.target, self.scorer)
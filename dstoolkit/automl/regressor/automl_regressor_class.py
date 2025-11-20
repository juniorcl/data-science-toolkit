import optuna
import inspect

import pandas as pd

from .params_space import get_params_space
from .model_instance import get_model_instance

from ..utils import get_regressor_eval_scoring, get_regressor_metrics, analyze_regressor

class AutoMLRegressor:
    def __init__(self, model_name, target='target', scoring='r2', tune=False, n_trials=50, random_state=42):
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
            model.fit(self.X_train, self.y_train[self.target])
            preds = model.predict(self.X_valid)
            return self.func_metric(self.y_valid[self.target], preds)
        
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
        elif 'verbosity' in model_sig:
            init_params['verbosity'] = -1

        params = self._get_best_params() if self.tune else init_params
        model = self.model_class(**params)

        fit_sig = inspect.signature(model.fit).parameters
        fit_args = {'X': self.X_train, 'y': self.y_train[self.target]}
        if 'eval_set' in fit_sig and self.X_valid is not None:
            fit_args['eval_set'] = [(self.X_valid, self.y_valid[self.target])]

        model.fit(**fit_args)

        self.y_train['pred'] = model.predict(self.X_train)
        self.y_valid['pred'] = model.predict(self.X_valid)
        self.y_test['pred'] = model.predict(self.X_test)

        results = {
            'Train': get_regressor_metrics(self.y_train, target=self.target, pred_col='pred'),
            'Valid': get_regressor_metrics(self.y_valid, target=self.target, pred_col='pred'),
            'Test': get_regressor_metrics(self.y_test, target=self.target, pred_col='pred')
        }
        return model, results

    def train(self, X_train, y_train, X_valid, y_valid, X_test, y_test, target='target'):
        self.target = target
        self.X_train, self.X_valid, self.X_test = X_train, X_valid, X_test
        self.y_train, self.y_valid, self.y_test = y_train, y_valid, y_test
        self.model, self.results = self._fit()

    def get_metrics(self, return_df=True):
        return pd.DataFrame(self.results).T if return_df else self.results
    
    def analyze(self):
        analyze_regressor(self.model, self.X_train, self.y_train, self.y_test, self.target, self.scorer)
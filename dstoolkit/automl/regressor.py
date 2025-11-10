import optuna
import inspect
import numpy as np
import pandas as pd
import utils_regressor as utils
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_validate, BaseCrossValidator
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


def get_params_space(model_name, trial, random_state=42):
    match model_name:
        case 'LightGBM':
            return {
                'objective': trial.suggest_categorical('objective', ['regression']),
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
                'metric': trial.suggest_categorical('metric', ['rmse', 'mae', 'mape', 'mse']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'random_state': trial.suggest_categorical('random_state', [random_state]),
                'verbose': trial.suggest_categorical('verbose', [-1])
            }
        case 'CatBoost':
            return {
                'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE']),
                'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'rsm': trial.suggest_float('rsm', 0.5, 1.0),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                'boosting_type': trial.suggest_categorical('boosting_type', ['Plain']),
                'random_seed': trial.suggest_categorical('random_seed', [random_state]),
                'verbose': trial.suggest_categorical('verbose', [0])
            }
        case 'HistGradientBoosting':
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 255),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
                'early_stopping': trial.suggest_categorical('early_stopping', [False]),
                'scoring': trial.suggest_categorical('scoring', ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']),
                'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.4),
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20),
                'random_state': trial.suggest_categorical('random_state', [random_state])
            }
        case 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': trial.suggest_categorical('random_state', [random_state]),
                'n_jobs': trial.suggest_categorical('n_jobs', [-1])
            }
        case 'GradientBoosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber']),
                'random_state': trial.suggest_categorical('random_state', [random_state])
            }
        case _:
            raise ValueError(f"Model '{model_name}' is not supported.")

def get_model_instance(model_name):
    match model_name:
        case 'LightGBM':
            return LGBMRegressor
        case 'CatBoost':
            return CatBoostRegressor
        case 'HistGradientBoosting':
            return HistGradientBoostingRegressor
        case 'RandomForest':
            return RandomForestRegressor
        case 'GradientBoosting':
            return GradientBoostingRegressor
        case _:
            raise ValueError(f"Model '{model_name}' is not suported.")

class AutoMLRegressor:
    def __init__(
        self, X_train, y_train, X_valid, y_valid, X_test, y_test, model_name, target='target', 
        scoring='r2', cv=None, features=None, tune=False, params_space=None, n_trials=50, random_state=42):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        self.tune = tune
        self.target = target
        self.n_trials = n_trials
        self.model_name = model_name
        self.model_instance = get_model_instance(model_name)
        self.features = features if features else self.X_train.columns.tolist()
        self.scorer = utils.get_eval_scoring(scoring, return_func=False)
        self.func_metric = utils.get_eval_scoring(scoring, return_func=True)
        self.random_state = random_state

    def _get_best_params(self):
        def objective(trial):
            params = get_params_space(self.model_name, trial, self.random_state)
            model = self.model_instance(**params)
            
            if isinstance(self.cv, (int, BaseCrossValidator)):
                cv_results = cross_validate(
                    estimator=model, 
                    X=self.X_train[self.features],
                    y=self.y_train[self.target],
                    cv=self.cv,
                    scoring=self.scorer
                )
                return np.mean(cv_results['test_score'])
            
            model.fit(self.X_train[self.features], self.y_train[self.target])
            preds = model.predict(self.X_valid[self.features])
            return self.func_metric(self.y_valid[self.target], preds)
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params
    
    def _cross_validate(self, model):
        cv_results = cross_validate(
            estimator=model, 
            X=self.X_train[self.features], 
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
        model_sig = inspect.signature(self.model_instance).parameters
        init_params = {}
        if 'random_state' in model_sig:
            init_params['random_state'] = self.random_state
        if 'verbose' in model_sig:
            init_params['verbose'] = 0
        elif 'verbosity' in model_sig:
            init_params['verbosity'] = -1

        params = self._get_best_params() if self.tune else init_params
        model = self.model_instance(**params)

        if isinstance(self.cv, (int, BaseCrossValidator)):
            results = {'Train CV': self._cross_validate(model)}
            model.fit(self.X_train[self.features], self.y_train[self.target])
            self.y_test['pred'] = model.predict(self.X_test[self.features])
            results['Test'] = utils.get_metrics(self.y_test, 'pred', self.target)
            return model, results

        fit_sig = inspect.signature(model.fit).parameters
        fit_args = {'X': self.X_train[self.features], 'y': self.y_train[self.target]}
        if 'eval_set' in fit_sig and self.X_valid is not None:
            fit_args['eval_set'] = [(self.X_valid[self.features], self.y_valid[self.target])]
        if 'early_stopping_rounds' in fit_sig:
            fit_args['early_stopping_rounds'] = 50

        model.fit(**fit_args)

        self.y_train['pred'] = model.predict(self.X_train[self.features])
        self.y_valid['pred'] = model.predict(self.X_valid[self.features])
        self.y_test['pred'] = model.predict(self.X_test[self.features])

        results = {
            'Train': utils.get_metrics(self.y_train, 'pred', self.target),
            'Valid': utils.get_metrics(self.y_valid, 'pred', self.target),
            'Test': utils.get_metrics(self.y_test, 'pred', self.target)
        }
        return model, results

    def train(self):
        self.model, self.results = self._fit()

    def get_metrics(self, return_df=True):
        return pd.DataFrame(self.results).T if return_df else self.results
    
    def get_result_analysis(self):
        display(self.get_metrics())
        utils.analyze_model(
            self.model, self.features, 
            self.X_train, self.y_train, self.y_test, self.target, self.scorer
        )
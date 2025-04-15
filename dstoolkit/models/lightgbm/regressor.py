import shap
import optuna

import numpy   as np
import pandas  as pd
import seaborn as sns

import matplotlib.pyplot as plt

from typing   import List, Dict
from sklearn  import metrics
from lightgbm import LGBMRegressor

from sklearn.base            import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_validate, train_test_split, KFold, StratifiedKFold

from ..metrics import (
    ks_scorer, 
    summarize_metric_results, 
    analyze_model, 
    get_regressor_metrics, 
    get_classifier_metrics,
    get_eval_scoring
)

from ..analysis import (
    plot_permutation_importance, 
    plot_feature_importance, 
    plot_shap_summary, 
    plot_residuals, 
    plot_pred_vs_true,
    plot_error_by_quantile,
    plot_learning_curve,
    analyze_model
)


def get_default_model(random_state=42, n_jobs=-1) -> LGBMRegressor:
    
    return LGBMRegressor(random_state=random_state, n_jobs=n_jobs, verbose=-1)

def get_objective_params(trial, random_state=42, n_jobs=-1) -> dict:

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
        'verbose': trial.suggest_categorical('verbose', [-1]),
        'n_jobs': trial.suggest_categorical('n_jobs', [n_jobs])
    }


class AutoMLLGBMRegressor:
    
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, 
        best_features: list[str], target: str, scoring = str, n_trials: int = 50, random_state: int = 42, n_jobs: int = -1):

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.best_features = best_features
        self.target = target
        self.n_trials = n_trials
        self.scorer = get_eval_scoring(scoring, return_func=False)
        self.func_metric = get_eval_scoring(scoring, return_func=True)
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _train_model(self, model_name: str, features: list[str], model: LGBMRegressor) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        model.fit(
            self.X_train[features], self.y_train[self.target],
            eval_set=[(self.X_valid[features], self.y_valid[self.target])]
        )

        self.y_train[f'{model_name}_pred'] = model.predict(self.X_train[features])
        self.y_valid[f'{model_name}_pred'] = model.predict(self.X_valid[features])
        self.y_test[f'{model_name}_pred'] = model.predict(self.X_test[features])

        results = {
            'Train': get_regressor_metrics(self.y_train, f'{model_name}_pred', self.target),
            'Valid': get_regressor_metrics(self.y_valid, f'{model_name}_pred', self.target),
            'Test': get_regressor_metrics(self.y_test, f'{model_name}_pred', self.target)
        }
        
        return model, results

    def _train_base_model(self) -> dict[str, dict[str, float]]:
        
        model = get_default_model(random_state=self.random_state, n_jobs=self.n_jobs)
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self) -> dict[str, dict[str, float]]:
        
        model = get_default_model(random_state=self.random_state, n_jobs=self.n_jobs)
        
        return self._train_model('best_feature_model', self.best_features, model)

    def _get_best_params(self) -> dict:
        
        def objective(trial):
            
            params = get_objective_params(trial, self.random_state, self.n_jobs)
            
            model = LGBMRegressor(**params)
            model.fit(
                self.X_train[self.best_features], self.y_train[self.target],
                eval_set=[(self.X_valid[self.best_features], self.y_valid[self.target])],
            )
            
            preds = model.predict(self.X_valid[self.best_features])
            
            return self.func_metric(self.y_valid[self.target], preds)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params

    def _train_best_params_model(self) -> dict[str, dict[str, float]]:
        
        best_params = self._get_best_params()
        self.best_params_model = LGBMRegressor(**best_params)
        
        return self._train_model('best_params_model', self.best_features, self.best_params_model)

    def train(self) -> None:
        
        self.base_model, self.base_model_results = self._train_base_model()
        self.best_feature_model, self.best_feature_model_results = self._train_best_feature_model()
        self.best_params_model, self.best_params_model_results = self._train_best_params_model()

    def get_metrics(self) -> pd.DataFrame:

        model_results = {
            "Base Model": self.base_model_results,
            "Best Feature Model": self.best_feature_model_results,
            "Best Params Model": self.best_params_model_results,
        }

        summary_frames = [summarize_metric_results(results).assign(Model=name) for name, results in model_results.items()]

        return pd.concat(summary_frames, ignore_index=True)
    
    def get_result_analysis(self) -> None:
    
        analyze_model("base_model", self.base_model, self.base_model_results, self.X_train, self.y_train, self.y_test, self.target, self.scorer)
        analyze_model("best_feature_model", self.best_feature_model, self.best_feature_model_results, self.X_train, self.y_train, self.y_test, self.target, self.scorer)
        analyze_model("best_params_model", self.best_params_model, self.best_params_model_results, self.X_train, self.y_train, self.y_test, self.target, self.scorer)


class AutoMLLGBMRegressorCV:
    
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, 
        best_features: list[str], target: str, scoring: str, n_trials: int = 50, cv: int = 5, random_state: int = 42, n_jobs: int = -1):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_features = best_features
        self.target = target
        self.n_trials = n_trials
        self.cv = cv
        self.scorer = get_eval_scoring(scoring, return_func=False)
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _cross_validate(self, model: LGBMRegressor, features: list[str]) -> None:

        cv_results = cross_validate(
            estimator=model, X=self.X_train[features], y=self.y_train[self.target], cv=self.cv, n_jobs=self.n_jobs,
            scoring=('r2', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error', 'explained_variance')
        )

        return {
            'R2': cv_results['test_r2'].mean(),
            'MAE': np.abs(cv_results['test_neg_mean_absolute_error'].mean()),
            'MadAE': np.abs(cv_results['test_neg_median_absolute_error'].mean()),
            'MAPE': np.abs(cv_results['test_neg_mean_absolute_percentage_error'].mean()),
            'RMSE': np.abs(cv_results['test_neg_root_mean_squared_error'].mean()),
            'Explained Variance': cv_results['test_explained_variance'].mean()
        }

    def _train_model(self, model_name: str, features: list[str], model: LGBMRegressor) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        model.fit(self.X_train[features], self.y_train[self.target])
        
        self.y_train[f'{model_name}_pred'] = model.predict(self.X_train[features])
        self.y_test[f'{model_name}_pred'] = model.predict(self.X_test[features])

        results = {
            'Train CV': self._cross_validate(model, features),
            'Test': get_regressor_metrics(self.y_test, f'{model_name}_pred', self.target)
        }
        
        return model, results

    def _get_best_params(self) -> dict:
        
        def objective(trial):
            
            params = get_objective_params(trial, self.random_state, self.n_jobs)

            cv_results = cross_validate(
                estimator=LGBMRegressor(**params), cv=self.cv, n_jobs=self.n_jobs, scoring=self.scorer,
                X=self.X_train[self.best_features], y=self.y_train[self.target])

            return cv_results['test_score'].mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params

    def _train_base_model(self) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        model = get_default_model(random_state=self.random_state, n_jobs=self.n_jobs)
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:

        model = get_default_model(random_state=self.random_state, n_jobs=self.n_jobs)

        return self._train_model('best_feature_model', self.best_features, model)

    def _train_best_params_model(self) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        best_params = self._get_best_params()
        model = LGBMRegressor(**best_params)
        
        return self._train_model('best_params_model', self.best_features, model)

    def train(self) -> None:
        
        self.base_model, self.base_model_results = self._train_base_model()
        self.best_feature_model, self.best_feature_model_results = self._train_best_feature_model()
        self.best_params_model, self.best_params_model_results = self._train_best_params_model()
    
    def get_metrics(self) -> pd.DataFrame:

        model_results = {
            "Base Model": self.base_model_results,
            "Best Feature Model": self.best_feature_model_results,
            "Best Params Model": self.best_params_model_results,
        }

        summary_frames = [summarize_metric_results(results).assign(Model=name) for name, results in model_results.items()]

        return pd.concat(summary_frames, ignore_index=True)

    def get_result_analysis(self) -> None:
    
        analyze_model("base_model", self.base_model, self.base_model_results, self.X_train, self.y_train, self.y_test, self.target, self.scorer)
        analyze_model("best_feature_model", self.best_feature_model, self.best_feature_model_results, self.X_train, self.y_train, self.y_test, self.target, self.scorer)
        analyze_model("best_params_model", self.best_params_model, self.best_params_model_results, self.X_train, self.y_train, self.y_test, self.target, self.scorer)

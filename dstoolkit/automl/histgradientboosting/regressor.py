import optuna

import numpy   as np
import pandas  as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import cross_validate

from ..metrics import (
    analyze_model, 
    get_regressor_metrics,
    get_eval_scoring
)

from ..analysis import analyze_model, summarize_metric_results


def get_default_model(random_state=42) -> HistGradientBoostingRegressor:
    
    return HistGradientBoostingRegressor(random_state=random_state, verbose=0)

def get_objective_params(trial, random_state=42) -> dict:

    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
        'scoring': trial.suggest_categorical('scoring', ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']),
        'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.4),
        'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20),
        'random_state': trial.suggest_categorical('random_state', [random_state]),
        'verbose': trial.suggest_categorical('verbose', [0]),
        'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error'])
    }


class AutoMLHistGradientBoostingRegressor:
    
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, 
        best_features: list[str], target: str, scoring = str, n_trials: int = 50, random_state: int = 42):

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

    def _train_model(self, model_name: str, features: list[str], model: HistGradientBoostingRegressor) -> tuple[HistGradientBoostingRegressor, dict[str, dict[str, float]]]:
        
        model.fit(self.X_train[features], self.y_train[self.target])

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
        
        model = get_default_model(random_state=self.random_state)
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self) -> dict[str, dict[str, float]]:
        
        model = get_default_model(random_state=self.random_state)
        
        return self._train_model('best_feature_model', self.best_features, model)

    def _get_best_params(self) -> dict:
        
        def objective(trial):
            
            params = get_objective_params(trial, self.random_state)
            
            model = HistGradientBoostingRegressor(**params)
            model.fit(self.X_train[self.best_features], self.y_train[self.target])
            
            preds = model.predict(self.X_valid[self.best_features])
            
            return self.func_metric(self.y_valid[self.target], preds)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params

    def _train_best_params_model(self) -> dict[str, dict[str, float]]:
        
        best_params = self._get_best_params()
        self.best_params_model = HistGradientBoostingRegressor(**best_params)
        
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
    
        analyze_model(
            "base_model", 
            self.base_model, 
            self.base_model_results, 
            self.X_train.columns.tolist(), 
            self.X_train, 
            self.y_train, 
            self.y_test, 
            self.target, 
            self.scorer
        )
        analyze_model(
            "best_feature_model", 
            self.best_feature_model, 
            self.best_feature_model_results, 
            self.best_features, 
            self.X_train, 
            self.y_train, 
            self.y_test, 
            self.target, 
            self.scorer
        )
        analyze_model(
            "best_params_model", 
            self.best_params_model, 
            self.best_params_model_results,
            self.best_features, 
            self.X_train, 
            self.y_train, 
            self.y_test, 
            self.target, 
            self.scorer
        )


class AutoMLHistGradientBoostingRegressorCV:
    
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, 
        best_features: list[str], target: str, scoring: str, n_trials: int = 50, cv: int = 5, random_state: int = 42):

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

    def _cross_validate(self, model: HistGradientBoostingRegressor, features: list[str]) -> None:

        cv_results = cross_validate(
            estimator=model, 
            X=self.X_train[features], 
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

    def _train_model(self, model_name: str, features: list[str], model: HistGradientBoostingRegressor) -> tuple[HistGradientBoostingRegressor, dict[str, dict[str, float]]]:
        
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
            
            params = get_objective_params(trial, self.random_state)

            cv_results = cross_validate(
                estimator=HistGradientBoostingRegressor(**params), cv=self.cv, scoring=self.scorer,
                X=self.X_train[self.best_features], y=self.y_train[self.target])

            return cv_results['test_score'].mean()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params

    def _train_base_model(self) -> tuple[HistGradientBoostingRegressor, dict[str, dict[str, float]]]:
        
        model = get_default_model(random_state=self.random_state)
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self) -> tuple[HistGradientBoostingRegressor, dict[str, dict[str, float]]]:

        model = get_default_model(random_state=self.random_state)

        return self._train_model('best_feature_model', self.best_features, model)

    def _train_best_params_model(self) -> tuple[HistGradientBoostingRegressor, dict[str, dict[str, float]]]:
        
        best_params = self._get_best_params()
        model = HistGradientBoostingRegressor(**best_params)
        
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
    
        analyze_model(
            "base_model", 
            self.base_model, 
            self.base_model_results, 
            self.X_train.columns.tolist(), 
            self.X_train, 
            self.y_train, 
            self.y_test, 
            self.target, 
            self.scorer
        )
        analyze_model(
            "best_feature_model", 
            self.best_feature_model, 
            self.best_feature_model_results, 
            self.best_features, 
            self.X_train, 
            self.y_train, 
            self.y_test, 
            self.target, 
            self.scorer
        )
        analyze_model(
            "best_params_model", 
            self.best_params_model, 
            self.best_params_model_results,
            self.best_features, 
            self.X_train, 
            self.y_train, 
            self.y_test, 
            self.target, 
            self.scorer
        )
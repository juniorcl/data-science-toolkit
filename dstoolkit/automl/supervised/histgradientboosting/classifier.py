import optuna

import numpy   as np
import pandas  as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import cross_validate

from ..metrics import (
    ks_scorer, 
    analyze_model,
    get_classifier_metrics,
    get_eval_scoring
)

from ..analysis import (
    analyze_model, 
    summarize_metric_results,
    prob_to_score,
    calc_rating_limits,
    apply_ratings
)


def get_default_model(random_state) -> HistGradientBoostingClassifier:
    
    return HistGradientBoostingClassifier(random_state=random_state, verbose=0)

def get_params_objective(trial, random_state=42) -> dict:

    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=100),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        'early_stopping': trial.suggest_categorical('early_stopping', [False]),
        'scoring': trial.suggest_categorical('scoring', ['roc_auc', 'neg_brier_score']),
        'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.4),
        'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20),
        'random_state': trial.suggest_categorical('random_state', [random_state])
    }


class AutoMLHistGradientBoostingClassifier:

    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, 
        best_features: list[str], scoring: str, target: str, n_trials: int = 50, random_state: int = 42):

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

    def _train_model(self, model_name: str, features: list[str], model: HistGradientBoostingClassifier) -> tuple[HistGradientBoostingClassifier, dict[str, dict[str, float]]]:
        
        model.fit(self.X_train[features], self.y_train[self.target])

        self.y_train[f'{model_name}_prob'] = model.predict_proba(self.X_train[features])[:, 1]
        self.y_train[f'{model_name}_score'] = prob_to_score(self.y_train[f'{model_name}_prob'], inverse=True)
        self.train_rating_limits = calc_rating_limits(self.y_train[f'{model_name}_score'])
        self.y_train[f'{model_name}_rating'] = apply_ratings(self.y_train[f'{model_name}_score'], self.train_rating_limits)

        self.y_valid[f'{model_name}_prob'] = model.predict_proba(self.X_valid[features])[:, 1]
        self.y_valid[f'{model_name}_score'] = prob_to_score(self.y_valid[f'{model_name}_prob'], inverse=True)
        self.y_valid[f'{model_name}_rating'] = apply_ratings(self.y_valid[f'{model_name}_score'], self.train_rating_limits)
        
        self.y_test[f'{model_name}_prob'] = model.predict_proba(self.X_test[features])[:, 1]
        self.y_test[f'{model_name}_score'] = prob_to_score(self.y_test[f'{model_name}_prob'], inverse=True)
        self.y_test[f'{model_name}_rating'] = apply_ratings(self.y_test[f'{model_name}_score'], self.train_rating_limits)

        results = {
            'Train': get_classifier_metrics(self.y_train, model_name, self.target),
            'Valid': get_classifier_metrics(self.y_valid, model_name, self.target),
            'Test': get_classifier_metrics(self.y_test, model_name, self.target)
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
            
            params = get_params_objective(trial, random_state=self.random_state)

            model = HistGradientBoostingClassifier(**params)
            model.fit(self.X_train[self.best_features], self.y_train[self.target])
    
            preds_proba = model.predict_proba(self.X_valid[self.best_features])[:, 1]
    
            return self.func_metric(self.y_valid[self.target], preds_proba)
    
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
    
        return study.best_params

    def _train_best_params_model(self) -> dict[str, dict[str, float]]:
        
        best_params = self._get_best_params()
        self.best_params_model = HistGradientBoostingClassifier(**best_params)
        
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

class AutoMLHistGradientBoostingClassifierCV:

    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, best_features: list[str], 
        target: str, scoring: str, n_trials: int = 50, cv: int = 5, random_state: int = 42):

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

    def _cross_validate(self, model: HistGradientBoostingClassifier, features: list[str]) -> None:
        
        cv_results = cross_validate(
            estimator=model, 
            X=self.X_train[features], 
            y=self.y_train[self.target], 
            cv=self.cv,
            scoring={
                'balanced_accuracy': 'balanced_accuracy', 
                'precision': 'precision', 
                'recall': 'recall', 
                'f1': 'f1', 
                'roc_auc': 'roc_auc', 
                'ks': ks_scorer, 
                'brier': 'neg_brier_score'
            }
        )

        return {
            'Treshold': 0.5,
            'Balanced Accuracy': cv_results['test_balanced_accuracy'].mean(),
            'Precision': cv_results['test_precision'].mean(),
            'Recall': cv_results['test_recall'].mean(),
            'F1': cv_results['test_f1'].mean(),
            'AUC': cv_results['test_roc_auc'].mean(),
            'KS': cv_results['test_ks'].mean(),
            'Brier': np.abs(cv_results['test_brier'].mean())
        }

    def _train_model(self, model_name: str, features: list[str], model: HistGradientBoostingClassifier) -> tuple[HistGradientBoostingClassifier, dict[str, dict[str, float]]]:

        model.fit(self.X_train[features], self.y_train[self.target])
        
        self.y_train[f'{model_name}_prob'] = model.predict_proba(self.X_train[features])[:, 1]
        self.y_train[f'{model_name}_score'] = prob_to_score(self.y_train[f'{model_name}_prob'], inverse=True)
        self.train_rating_limits = calc_rating_limits(self.y_train[f'{model_name}_score'])
        self.y_train[f'{model_name}_rating'] = apply_ratings(self.y_train[f'{model_name}_score'], self.train_rating_limits)
        
        self.y_test[f'{model_name}_prob'] = model.predict_proba(self.X_test[features])[:, 1]
        self.y_test[f'{model_name}_score'] = prob_to_score(self.y_test[f'{model_name}_prob'], inverse=True)
        self.y_test[f'{model_name}_rating'] = apply_ratings(self.y_test[f'{model_name}_score'], self.train_rating_limits)
        
        results = {
            'Train CV': self._cross_validate(model, features),
            'Test': get_classifier_metrics(self.y_test, model_name, self.target)
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
            
            params = get_params_objective(trial, random_state=self.random_state)
    
            cv_results = cross_validate(
                estimator=HistGradientBoostingClassifier(**params), cv=self.cv, scoring=self.scorer,
                X=self.X_train[self.best_features], y=self.y_train[self.target])
    
            return cv_results['test_score'].mean()
    
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
    
        return study.best_params

    def _train_best_params_model(self) -> dict[str, dict[str, float]]:
        
        best_params = self._get_best_params()
        self.best_params_model = HistGradientBoostingClassifier(**best_params)
        
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
import optuna

import numpy   as np
import pandas  as pd

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_validate

from .analysis import (
    ks_scorer, 
    analyze_model,
    get_metrics,
    get_eval_scoring,
    analyze_model, 
    summarize_metric_results,
    prob_to_score,
    calc_rating_limits,
    apply_ratings
)


def get_default_model(random_state):
    
    return LGBMClassifier(random_state=random_state, verbose=-1, objective='binary')

def get_params_objective(trial, random_state=42):

    return {
        'objective': trial.suggest_categorical('objective', ['binary']),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
        'metric': trial.suggest_categorical('metric', ['auc', 'binary_logloss']),
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


class AutoMLLGBMClassifier:

    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, best_features, scoring, target, n_trials=50, random_state=42):

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

    def _train_model(self, model_name, features, model):
        
        model.fit(
            self.X_train[features], self.y_train[self.target],
            eval_set=[(self.X_valid[features], self.y_valid[self.target])]
        )

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
            'Train': get_metrics(self.y_train, model_name, self.target),
            'Valid': get_metrics(self.y_valid, model_name, self.target),
            'Test': get_metrics(self.y_test, model_name, self.target)
        }
        
        return model, results

    def _train_base_model(self):
        
        model = get_default_model(random_state=self.random_state)
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self):
        
        model = get_default_model(random_state=self.random_state)
        
        return self._train_model('best_feature_model', self.best_features, model)

    def _get_best_params(self):

        def objective(trial):
            
            params = get_params_objective(trial, random_state=self.random_state)
    
            model = LGBMClassifier(**params)
            model.fit(
                self.X_train[self.best_features], self.y_train[self.target],
                eval_set=[(self.X_valid[self.best_features], self.y_valid[self.target])]
            )
    
            preds_proba = model.predict_proba(self.X_valid[self.best_features])[:, 1]
    
            return self.func_metric(self.y_valid[self.target], preds_proba)
    
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
    
        return study.best_params

    def _train_best_params_model(self):
        
        best_params = self._get_best_params()
        self.best_params_model = LGBMClassifier(**best_params)
        
        return self._train_model('best_params_model', self.best_features, self.best_params_model)

    def train(self):
        
        self.base_model, self.base_model_results = self._train_base_model()
        self.best_feature_model, self.best_feature_model_results = self._train_best_feature_model()
        self.best_params_model, self.best_params_model_results = self._train_best_params_model()

    def get_metrics(self):

        model_results = {
            "Base Model": self.base_model_results,
            "Best Feature Model": self.best_feature_model_results,
            "Best Params Model": self.best_params_model_results,
        }

        summary_frames = [summarize_metric_results(results).assign(Model=name) for name, results in model_results.items()]

        return pd.concat(summary_frames, ignore_index=True)

    def get_result_analysis(self):
    
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

class AutoMLLGBMClassifierCV:

    def __init__(self, X_train, y_train, X_test, y_test, best_features, target, scoring, n_trials=50, cv=5, random_state=42):

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

    def _cross_validate(self, model, features):
        
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

    def _train_model(self, model_name, features, model):

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
            'Test': get_metrics(self.y_test, model_name, self.target)
        }
        
        return model, results

    def _train_base_model(self):
        
        model = get_default_model(random_state=self.random_state)
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self):
        
        model = get_default_model(random_state=self.random_state)
        
        return self._train_model('best_feature_model', self.best_features, model)

    def _get_best_params(self):

        def objective(trial):
            
            params = get_params_objective(trial, random_state=self.random_state)
    
            cv_results = cross_validate(
                estimator=LGBMClassifier(**params), cv=self.cv, scoring=self.scorer,
                X=self.X_train[self.best_features], y=self.y_train[self.target])
    
            return cv_results['test_score'].mean()
    
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
    
        return study.best_params

    def _train_best_params_model(self):
        
        best_params = self._get_best_params()
        self.best_params_model = LGBMClassifier(**best_params)
        
        return self._train_model('best_params_model', self.best_features, self.best_params_model)

    def train(self):
        
        self.base_model, self.base_model_results = self._train_base_model()
        self.best_feature_model, self.best_feature_model_results = self._train_best_feature_model()
        self.best_params_model, self.best_params_model_results = self._train_best_params_model()

    def get_metrics(self):

        model_results = {
            "Base Model": self.base_model_results,
            "Best Feature Model": self.best_feature_model_results,
            "Best Params Model": self.best_params_model_results,
        }

        summary_frames = [summarize_metric_results(results).assign(Model=name) for name, results in model_results.items()]

        return pd.concat(summary_frames, ignore_index=True)

    def get_result_analysis(self):
    
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
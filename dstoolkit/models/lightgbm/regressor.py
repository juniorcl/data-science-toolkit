import shap
import optuna

import pandas  as pd
import seaborn as sns

import matplotlib.pyplot as plt

from typing   import List, Dict, Callable
from sklearn  import metrics
from lightgbm import LGBMRegressor

from sklearn.inspection      import permutation_importance
from sklearn.model_selection import cross_validate


def plot_permutation_importance(model: LGBMRegressor, X: pd.DataFrame, y: pd.DataFrame):

    permu_results = permutation_importance(model, X[model.feature_name_], y, scoring='r2', n_repeats=5, n_jobs=-1, random_state=42)

    sorted_importances_idx = permu_results.importances_mean.argsort()
    
    df_results = pd.DataFrame(permu_results.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
    
    ax = df_results.plot.box(vert=False, whis=10)
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    
    plt.show()

def plot_feature_importance(model: LGBMRegressor) -> None:

    df_imp = pd.DataFrame(model.feature_importances_, model.feature_name_).reset_index()
    df_imp.columns = ["Variable", "Importance"]
    df_imp = df_imp.sort_values("Importance", ascending=False)
    
    sns.barplot(x="Importance", y="Variable", color="#006e9cff", data=df_imp[:20])
    
    plt.title(f"Importance of Variables")
    plt.show()

def get_shap_values(model: LGBMRegressor, X: pd.DataFrame):
    
    explainer = shap.TreeExplainer(model)
    
    return explainer.shap_values(X[model.feature_name_])

def plot_shap_summary(model: LGBMRegressor, X: pd.DataFrame) -> None:
    
    shap_values = get_shap_values(model, X)
    shap.summary_plot(shap_values, X[model.feature_name_])

def plot_shap_dependence(model: LGBMRegressor, X: pd.DataFrame) -> None:
   
    shap_values = get_shap_values(model, X)
    for feature in model.feature_name_[:3]:  # top 3 features
        shap.dependence_plot(feature, shap_values, X[model.feature_name_])

def plot_residuals(y: pd.DataFrame, pred_col: str, title: str, target: str) -> None:
        
    residuals = y[target] - y[pred_col]
    
    sns.histplot(residuals, kde=True)
    
    plt.title(f"Distribuição dos Resíduos - {title}")
    plt.xlabel("Erro (y_true - y_pred)")
    plt.show()

def plot_pred_vs_true(y: pd.DataFrame, pred_col: str, title: str, target: str) -> None:

    sns.scatterplot(x=y[target], y=y[pred_col])
    
    plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], '--r')
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(f"y_true vs y_pred - {title}")
    plt.show()

def plot_error_by_quantile(y: pd.DataFrame, pred_col: str, title: str, target: str) -> None:
    
    y_copy = y.copy()
    y_copy["quantile"] = pd.qcut(y_copy[target], q=5)
    y_copy["abs_error"] = abs(y_copy[target] - y_copy[pred_col])

    sns.boxplot(x="quantile", y="abs_error", data=y_copy)

    plt.title(f"Erro absoluto por quantil do target - {title}")
    plt.xticks(rotation=45)
    plt.show()

def get_metrics(y: pd.DataFrame, pred_col: str, target: str) -> dict[str, float]:
        
    y_true = y[target]
    y_pred = y[pred_col]
    
    return {
        'R2': metrics.r2_score(y_true, y_pred),
        'MAE': metrics.mean_absolute_error(y_true, y_pred),
        'MAPE': metrics.mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': metrics.root_mean_squared_error(y_true, y_pred),
        'Explained Variance': metrics.explained_variance_score(y_true, y_pred)
    }

def summarize_metric_results(results: dict[str, dict[str, float]]) -> pd.DataFrame:
        
    rows = []
    
    for dataset, metrics_dict in results.items():
        row = {"Dataset": dataset}
        row.update(metrics_dict)
        rows.append(row)
    
    return pd.DataFrame(rows)

def analyze_model(model_name: str, model: LGBMRegressor, results: dict, X_train: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, target: str) -> None:
    
    print(f"{model_name} Results")

    display(summarize_metric_results(results))

    pred_col = f"{model_name}_pred"
    plot_residuals(y_test, pred_col, f"{model_name} (Test Dataset)", target)
    plot_pred_vs_true(y_test, pred_col, f"{model_name} (Test Dataset)", target)
    plot_error_by_quantile(y_test, pred_col, f"{model_name} (Test Dataset)", target)
    plot_feature_importance(model)
    plot_permutation_importance(model, X_train, y_train[target])
    plot_shap_summary(model, X_train)
    plot_shap_dependence(model, X_train)

def get_default_model(**kwargs) -> LGBMRegressor:
    
    return LGBMRegressor(random_state=42, verbose=-1, **kwargs)


class AutoMLLightGBMRegressor:
    
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame, 
        X_test: pd.DataFrame, y_test: pd.DataFrame, best_features: list[str], target: str, n_trials: int = 50):

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.best_features = best_features
        self.target = target
        self.n_trials = n_trials

    def _train_model(self, model_name: str, features: list[str], model: LGBMRegressor) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        model.fit(
            self.X_train[features], self.y_train[self.target],
            eval_set=[(self.X_valid[features], self.y_valid[self.target])]
        )

        self.y_train[f'{model_name}_pred'] = model.predict(self.X_train[features])
        self.y_valid[f'{model_name}_pred'] = model.predict(self.X_valid[features])
        self.y_test[f'{model_name}_pred'] = model.predict(self.X_test[features])

        results = {
            'Train': get_metrics(self.y_train, f'{model_name}_pred', self.target),
            'Valid': get_metrics(self.y_valid, f'{model_name}_pred', self.target),
            'Test': get_metrics(self.y_test, f'{model_name}_pred', self.target)
        }
        
        return model, results

    def _train_base_model(self) -> dict[str, dict[str, float]]:
        
        model = get_default_model()
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self) -> dict[str, dict[str, float]]:
        
        model = get_default_model()
        
        return self._train_model('best_feature_model', self.best_features, model)

    def _get_best_params(self) -> dict:
        
        def objective(trial):
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbose': -1,
            }
            
            model = LGBMRegressor(**params)
            model.fit(
                self.X_train[self.best_features], self.y_train[self.target],
                eval_set=[(self.X_valid[self.best_features], self.y_valid[self.target])],
            )
            
            preds = model.predict(self.X_valid[self.best_features])
            
            return metrics.r2_score(self.y_valid[self.target], preds)

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

    def summarize_metrics(self) -> pd.DataFrame:

        model_results = {
            "Base Model": self.base_model_results,
            "Best Feature Model": self.best_feature_model_results,
            "Best Params Model": self.best_params_model_results,
        }

        summary_frames = [summarize_metric_results(results).assign(Model=name) for name, results in model_results.items()]

        return pd.concat(summary_frames, ignore_index=True)
    
    def get_model_analysis(self) -> None:
    
        analyze_model("base_model", self.base_model, self.base_model_results, self.X_train, self.y_train, self.y_test, self.target)
        analyze_model("best_feature_model", self.best_feature_model, self.best_feature_model_results, self.X_train, self.y_train, self.y_test, self.target)
        analyze_model("best_params_model", self.best_params_model, self.best_params_model_results, self.X_train, self.y_train, self.y_test, self.target)


class AutoMLLightGBMRegressorCV:
    
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, 
        best_features: list[str], target: str, n_trials: int = 50, cv: Callable or int = 5):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_features = best_features
        self.target = target
        self.n_trials = n_trials
        self.cv = cv

    def _cross_validate(self, model: LGBMRegressor, features: list[str]) -> None:

        cv_results = cross_validate(
            estimator=model, X=self.X_train[features], y=self.y_train[self.target], cv=self.cv, n_jobs=-1,
            scoring=('r2', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_root_mean_squared_error', 'explained_variance')
        )

        return {
            'R2': cv_results['test_r2'].mean(),
            'MAE': np.abs(cv_results['test_neg_mean_absolute_error'].mean()),
            'MAPE': np.abs(cv_results['test_neg_mean_absolute_percentage_error'].mean()),
            'RMSE': np.abs(cv_results['test_neg_root_mean_squared_error'].mean()),
            'Explained Variance': cv_results['test_explained_variance'].mean()
        }

    def _train_model(self, model_name: str, features: list[str], model: LGBMRegressor) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        model.fit(self.X_train[features], self.y_train[self.target])
        
        self.y_train[f'{model_name}_pred'] = model.predict(self.X_train[features])
        self.y_test[f'{model_name}_pred'] = model.predict(self.X_test[features])

        results = {
            'CV': self._cross_validate(model, features),
            'Test': get_metrics(self.y_test, f'{model_name}_pred', self.target)
        }
        
        return model, results

    def _get_best_params(self) -> dict:
        
        def objective(trial):
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbose': -1,
            }
            
            return self._cross_validate(LGBMRegressor(**params), self.best_features)['R2']

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study.best_params

    def _train_base_model(self) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        model = get_default_model()
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:

        model = get_default_model()

        return self._train_model('best_feature_model', self.best_features, model)

    def _train_best_params_model(self) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        best_params = self._get_best_params()
        model = LGBMRegressor(**best_params)
        
        return self._train_model('best_params_model', self.best_features, model)

    def train(self) -> None:
        
        self.base_model, self.base_model_results = self._train_base_model()
        self.best_feature_model, self.best_feature_model_results = self._train_best_feature_model()
        self.best_params_model, self.best_params_model_results = self._train_best_params_model()
    
    def summarize_metrics(self) -> pd.DataFrame:

        model_results = {
            "Base Model": self.base_model_results,
            "Best Feature Model": self.best_feature_model_results,
            "Best Params Model": self.best_params_model_results,
        }

        summary_frames = [summarize_metric_results(results).assign(Model=name) for name, results in model_results.items()]

        return pd.concat(summary_frames, ignore_index=True)

    def get_model_analysis(self) -> None:
    
        analyze_model("base_model", self.base_model, self.base_model_results, self.X_train, self.y_test, self.target)
        analyze_model("best_feature_model", self.best_feature_model, self.best_feature_model_results, self.X_train, self.y_test, self.target)
        analyze_model("best_params_model", self.best_params_model, self.best_params_model_results, self.X_train, self.y_test, self.target)

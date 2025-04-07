import shap
import optuna

import pandas  as pd
import seaborn as sns

import matplotlib.pyplot as plt

from typing   import List, Dict
from sklearn  import metrics
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split


class AutoMLLightGBMRegressor:
    
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame, 
        X_test: pd.DataFrame, y_test: pd.DataFrame, best_features: list[str], target: str, n_trials: int = 50, random_state: int = 42):

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.best_features = best_features
        self.target = target
        self.n_trials = n_trials
        self.random_state = random_state

    def _get_metrics(self, y: pd.DataFrame, pred_col: str) -> dict[str, float]:
        
        y_true = y[self.target]
        y_pred = y[pred_col]
        
        return {
            'R2': metrics.r2_score(y_true, y_pred),
            'MAE': metrics.mean_absolute_error(y_true, y_pred),
            'MAPE': metrics.mean_absolute_percentage_error(y_true, y_pred),
            'RMSE': metrics.root_mean_squared_error(y_true, y_pred),
            'Explained Variance': metrics.explained_variance_score(y_true, y_pred)
        }

    def _train_model(self, model_name: str, features: list[str], model: LGBMRegressor) -> tuple[LGBMRegressor, dict[str, dict[str, float]]]:
        
        model.fit(
            self.X_train[features], self.y_train[self.target],
            eval_set=[(self.X_valid[features], self.y_valid[self.target])]
        )

        self.y_train[f'{model_name}_pred'] = model.predict(self.X_train[features])
        self.y_valid[f'{model_name}_pred'] = model.predict(self.X_valid[features])
        self.y_test[f'{model_name}_pred'] = model.predict(self.X_test[features])

        results = {
            'Train': self._get_metrics(self.y_train, f'{model_name}_pred'),
            'Valid': self._get_metrics(self.y_valid, f'{model_name}_pred'),
            'Test': self._get_metrics(self.y_test, f'{model_name}_pred'),
        }
        
        return model, results
    
    def _plot_feature_impotance(self, model: LGBMRegressor) -> None:

        df_imp = pd.DataFrame(model.feature_importances_, model.feature_name_).reset_index()
        df_imp.columns = ["Variable", "Importance"]
        df_imp = df_imp.sort_values("Importance", ascending=False)
    
        sns.barplot(x="Importance", y="Variable", color="#006e9cff", data=df_imp[:20])
    
        plt.title(f"Importance of Variables")
        plt.show()

    def _plot_shap_summary(self, model: LGBMRegressor) -> None:

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_train[model.feature_name_])
        shap.summary_plot(shap_values, self.X_train[model.feature_name_])

    def _plot_residuals(self, y: pd.DataFrame, pred_col: str, title: str) -> None:
        
        residuals = y[self.target] - y[pred_col]
        
        sns.histplot(residuals, kde=True)
        
        plt.title(f"Distribuição dos Resíduos - {title}")
        plt.xlabel("Erro (y_true - y_pred)")
        plt.show()

    def _plot_pred_vs_true(self, y: pd.DataFrame, pred_col: str, title: str) -> None:
    
        sns.scatterplot(x=y[self.target], y=y[pred_col])
    
        plt.plot([y[self.target].min(), y[self.target].max()], [y[self.target].min(), y[self.target].max()], '--r')
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title(f"y_true vs y_pred - {title}")
        plt.show()

    def _error_by_quantile(self, y: pd.DataFrame, pred_col: str, title: str) -> None:
        
        y_copy = y.copy()
        y_copy["quantile"] = pd.qcut(y_copy[self.target], q=5)
        y_copy["abs_error"] = abs(y_copy[self.target] - y_copy[pred_col])
    
        sns.boxplot(x="quantile", y="abs_error", data=y_copy)
    
        plt.title(f"Erro absoluto por quantil do target - {title}")
        plt.xticks(rotation=45)
        plt.show()

    def _plot_shap_dependence(self, model: LGBMRegressor) -> None:
    
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_train[model.feature_name_])
        
        for feature in model.feature_name_[:3]:  # top 3 features
            shap.dependence_plot(feature, shap_values, self.X_train[model.feature_name_])

    def _train_base_model(self) -> dict[str, dict[str, float]]:
        
        model = LGBMRegressor(random_state=self.random_state)
        
        return self._train_model('base_model', self.X_train.columns.tolist(), model)

    def _train_best_feature_model(self) -> dict[str, dict[str, float]]:
        
        model = LGBMRegressor(random_state=self.random_state)
        
        return self._train_model('best_feature_model', self.best_features, model)

    def _summarize_metric_results(self, results: dict[str, dict[str, float]]) -> pd.DataFrame:
        
        rows = []
    
        for dataset, metrics_dict in results.items():
            row = {"Dataset": dataset}
            row.update(metrics_dict)
            rows.append(row)
    
        return pd.DataFrame(rows)

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
                'random_state': self.random_state,
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

    def _analyze_model(self, model_name: str, model: LGBMRegressor, results: dict[str, dict[str, float]]) -> None:
        
        print(f"{model_name} Results")
    
        display(self._summarize_metric_results(results))

        pred_col = f"{model_name}_pred"
        self._plot_residuals(self.y_test, pred_col, f"{model_name} (Test Dataset)")
        self._plot_pred_vs_true(self.y_test, pred_col, f"{model_name} (Test Dataset)")
        self._error_by_quantile(self.y_test, pred_col, f"{model_name} (Test Dataset)")
        self._plot_feature_impotance(model)
        self._plot_shap_summary(model)
        self._plot_shap_dependence(model)

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

        summary_frames = [self._summarize_metric_results(results).assign(Model=name) for name, results in model_results.items()]

        return pd.concat(summary_frames, ignore_index=True)
    
    def get_model_analysis(self) -> None:
    
        self._analyze_model("base_model", self.base_model, self.base_model_results)
        self._analyze_model("best_feature_model", self.best_feature_model, self.best_feature_model_results)
        self._analyze_model("best_params_model", self.best_params_model, self.best_params_model_results)
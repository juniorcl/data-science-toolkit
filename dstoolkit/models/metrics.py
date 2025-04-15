import numpy  as np
import pandas as pd

from typing   import List, Dict
from sklearn  import metrics
from lightgbm import LGBMClassifier, LGBMRegressor

from scipy.stats             import ks_2samp
from sklearn.base            import BaseEstimator, ClassifierMixin, RegressorMixin 
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    root_mean_squared_error, 
    explained_variance_score, 
    f1_score, 
    make_scorer,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    median_absolute_error
)


def ks_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    
    pos_prob = y_prob[y_true == 1]
    neg_prob = y_prob[y_true == 0]
    
    if len(pos_prob) == 0 or len(neg_prob) == 0:
        return 0.0
    
    result = ks_2samp(pos_prob, neg_prob)
    
    return result.statistic

ks_scorer = make_scorer(ks_score, greater_is_better=True, response_method="predict_proba")

def get_eval_scoring(scoring, return_func=True):
    
    scorers = {
        'r2': 'r2',
        'ks': ks_scorer,
        'brier': 'neg_brier_score',
        'roc_auc': 'roc_auc',
        'explained_variance': 'explained_variance',
        'mean_absolute_error': 'neg_mean_absolute_error',
        'median_absolute_error': 'neg_median_absolute_error',
        'root_mean_squared_error': 'neg_root_mean_squared_error',
        'mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error'
    }

    functions = {
        'r2': r2_score,
        'ks': ks_score,
        'brier': lambda y_true, y_pred: -1 * brier_score_loss(y_true, y_pred),
        'roc_auc': roc_auc_score,
        'explained_variance': explained_variance_score,
        'mean_absolute_error': lambda y_true, y_pred: -1 * mean_absolute_error(y_true, y_pred),
        'median_absolute_error': lambda y_true, y_pred: -1 * median_absolute_error(y_true, y_pred),
        'root_mean_squared_error': lambda y_true, y_pred: -1 * root_mean_squared_error(y_true, y_pred),
        'mean_absolute_percentage_error': lambda y_true, y_pred: -1 * mean_absolute_percentage_error(y_true, y_pred)
    }
    
    if scoring not in scorers:
        raise ValueError(f"Metric '{scoring}' is not supported.")

    if scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")

    return functions[scoring] if return_func else scorers[scoring]

def summarize_metric_results(results: dict[str, dict[str, float]]) -> pd.DataFrame:
        
    rows = []
    
    for dataset, metrics_dict in results.items():
        row = {"Dataset": dataset}
        row.update(metrics_dict)
        rows.append(row)
    
    return pd.DataFrame(rows)

def analyze_model(model_name: str, model: BaseEstimator, results: dict, X_train: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, target: str) -> None:
    
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

def get_regressor_metrics(y: pd.DataFrame, pred_col: str, target: str) -> dict[str, float]:
        
    y_true = y[target]
    y_pred = y[pred_col]
    
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MadAE': median_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred)
    }

def get_best_threshold_for_f1(y_true: pd.Series, y_probs: pd.Series) -> float:
    
    thresholds = np.linspace(0, 1, 200)
    best_thresh = 0
    best_f1 = 0
    
    for thresh in thresholds:
        
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

def get_classifier_metrics(y: pd.DataFrame, model_name: str, target: str) -> dict[str, float]:
        
    best_threshold = get_best_threshold_for_f1(y[target], y[f'{model_name}_prob'])

    y[f'{model_name}_pred'] = 0
    y.loc[y[f'{model_name}_prob'] >= best_threshold, f'{model_name}_pred'] = 1
    
    return {
        'Treshold': best_threshold,
        'Balanced Accuracy': balanced_accuracy_score(y[target], y[f'{model_name}_pred']),
        'Precision': precision_score(y[target], y[f'{model_name}_pred']),
        'Recall': recall_score(y[target], y[f'{model_name}_pred']),
        'F1': f1_score(y[target], y[f'{model_name}_pred']),
        'AUC': roc_auc_score(y[target], y[f'{model_name}_prob']),
        'KS': ks_score(y[target], y[f'{model_name}_prob']),
        'Brier': brier_score_loss(y[target], y[f'{model_name}_prob'])
    }

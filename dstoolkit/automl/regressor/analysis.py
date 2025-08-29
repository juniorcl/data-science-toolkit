import shap

import pandas  as pd
import numpy   as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.base            import BaseEstimator
from sklearn.inspection      import permutation_importance
from sklearn.model_selection import learning_curve, KFold

from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    root_mean_squared_error, 
    explained_variance_score, 
    median_absolute_error
)


def get_metrics(y, pred_col, target):
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

def plot_permutation_importance(model, features, X, y, scoring):
    permu_results = permutation_importance(model, X[features], y, scoring=scoring, n_repeats=5, random_state=42)
    sorted_importances_idx = permu_results.importances_mean.argsort()
    df_results = pd.DataFrame(permu_results.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
    ax = df_results.plot.box(vert=False, whis=10, patch_artist=True, boxprops={'facecolor':'skyblue', 'color':'blue'})
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel(f"Decrease in {scoring}")
    plt.show()

def plot_feature_importance(model):
    feature_names = model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_names_
    df_imp = pd.DataFrame(model.feature_importances_, feature_names).reset_index()
    df_imp.columns = ["Variable", "Importance"]
    df_imp = df_imp.sort_values("Importance", ascending=False)
    sns.barplot(x="Importance", y="Variable", color="#006e9cff", data=df_imp[:20])
    plt.title(f"Importance of Variables")
    plt.show()

def plot_shap_summary(model, features, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[features])
    shap.summary_plot(shap_values, X[features])

def plot_residuals(y, pred_col, target):
    residuals = y[target] - y[pred_col]
    sns.histplot(residuals, kde=True)
    plt.title("Waste Distribution")
    plt.xlabel("Erro (y_true - y_pred)")
    plt.show()

def plot_pred_vs_true(y, pred_col, target):
    sns.scatterplot(x=y[target], y=y[pred_col])
    plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], '--r')
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("y_true vs y_pred")
    plt.show()

def plot_error_by_quantile(y, pred_col, target):
    y_copy = y.copy()
    y_copy["quantile"] = pd.qcut(y_copy[target], q=5)
    y_copy["abs_error"] = abs(y_copy[target] - y_copy[pred_col])
    sns.boxplot(x="quantile", y="abs_error", data=y_copy)
    plt.title("Absolute Error by Target Quantile")
    plt.xticks(rotation=45)
    plt.show()

def plot_learning_curve(model, X, y, target, scoring, cv=5):
    splitter = KFold(cv, shuffle=True, random_state=42)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y[target],
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=splitter,
        scoring=scoring
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Train')
    plt.fill_between(
        train_sizes_abs, train_scores_mean - train_scores_std, 
        train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.plot(train_sizes_abs, val_scores_mean, 'o-', color='b', label='Validation')
    plt.fill_between(
        train_sizes_abs, val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std, alpha=0.1, color='b')
    plt.title("Learning Curve")
    plt.xlabel("Train Size")
    plt.ylabel(f"{scoring}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def summarize_metric_results(results):
    rows = []
    for dataset, metrics_dict in results.items():
        row = {"Dataset": dataset}
        row.update(metrics_dict)
        rows.append(row)
    return pd.DataFrame(rows)

def analyze_model(model_name, model, results, features, X_train, y_train, y_test, target, scoring):
    print(f"{model_name} Results")
    display(summarize_metric_results(results))
    pred_col = f"{model_name}_pred"
    plot_residuals(y_test, pred_col, target)
    plot_pred_vs_true(y_test, pred_col, target)
    plot_error_by_quantile(y_test, pred_col, target)
    plot_learning_curve(model, X_train, y_train, target, scoring=scoring)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model)
    plot_permutation_importance(model, features, X_train, y_train[target], scoring=scoring)
    plot_shap_summary(model, features, X_train)

def get_eval_scoring(scoring, return_func=True):
    scorers = {
        'r2': 'r2',
        'explained_variance': 'explained_variance',
        'mean_absolute_error': 'neg_mean_absolute_error',
        'median_absolute_error': 'neg_median_absolute_error',
        'root_mean_squared_error': 'neg_root_mean_squared_error',
        'mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error'
    }
    functions = {
        'r2': r2_score,
        'explained_variance': explained_variance_score,
        'mean_absolute_error': lambda y_true, y_pred: -1 * mean_absolute_error(y_true, y_pred),
        'median_absolute_error': lambda y_true, y_pred: -1 * median_absolute_error(y_true, y_pred),
        'root_mean_squared_error': lambda y_true, y_pred: -1 * root_mean_squared_error(y_true, y_pred),
        'mean_absolute_percentage_error': lambda y_true, y_pred: -1 * mean_absolute_percentage_error(y_true, y_pred)
    }
    if scoring not in scorers or scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    return functions[scoring] if return_func else scorers[scoring]
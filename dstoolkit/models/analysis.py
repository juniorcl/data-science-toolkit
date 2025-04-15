import shap

import pandas  as pd
import numpy   as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.base            import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics         import roc_curve, precision_recall_curve
from sklearn.inspection      import permutation_importance
from sklearn.calibration     import calibration_curve
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold


def plot_permutation_importance(model: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame, scoring: str) -> None:

    permu_results = permutation_importance(model, X[model.feature_name_], y, scoring=scoring, n_repeats=5, random_state=42)

    sorted_importances_idx = permu_results.importances_mean.argsort()
    
    df_results = pd.DataFrame(permu_results.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
    
    ax = df_results.plot.box(vert=False, whis=10, patch_artist=True, boxprops={'facecolor':'skyblue', 'color':'blue'})
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel(f"Decrease in {scoring}")

    plt.show()

def plot_feature_importance(model: BaseEstimator) -> None:

    df_imp = pd.DataFrame(model.feature_importances_, model.feature_name_).reset_index()
    df_imp.columns = ["Variable", "Importance"]
    df_imp = df_imp.sort_values("Importance", ascending=False)
    
    sns.barplot(x="Importance", y="Variable", color="#006e9cff", data=df_imp[:20])
    
    plt.title(f"Importance of Variables")
    plt.show() 

def plot_shap_summary(model: BaseEstimator, X: pd.DataFrame) -> None:

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[model.feature_name_])
    
    shap.summary_plot(shap_values, X[model.feature_name_])

def plot_residuals(y: pd.DataFrame, pred_col: str, target: str) -> None:
        
    residuals = y[target] - y[pred_col]
    
    sns.histplot(residuals, kde=True)
    
    plt.title("Waste Distribution")
    plt.xlabel("Erro (y_true - y_pred)")
    plt.show()

def plot_pred_vs_true(y: pd.DataFrame, pred_col: str, target: str) -> None:

    sns.scatterplot(x=y[target], y=y[pred_col])
    
    plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], '--r')
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("y_true vs y_pred")
    plt.show()

def plot_error_by_quantile(y: pd.DataFrame, pred_col: str, target: str) -> None:
    
    y_copy = y.copy()
    y_copy["quantile"] = pd.qcut(y_copy[target], q=5)
    y_copy["abs_error"] = abs(y_copy[target] - y_copy[pred_col])

    sns.boxplot(x="quantile", y="abs_error", data=y_copy)

    plt.title("Absolute Error by Target Quantile")
    plt.xticks(rotation=45)
    plt.show()

def plot_roc_curve(y: pd.DataFrame, prob_col: str, target: str) -> None:
    
    fpr, tpr, _ = roc_curve(y[target], y[prob_col])

    plt.plot(fpr, tpr)
    plt.title('ROC AUC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()

def plot_precision_recall_curve(y: pd.DataFrame, prob_col: str, target: str):

    precision, recall, _ = precision_recall_curve(y[target], y[prob_col])

    plt.plot(precision, recall)
    plt.title('Precision Recall Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.show()

def plot_learning_curve(model: BaseEstimator, X: pd.DataFrame, y: pd.DataFrame, target: str, scoring: str, cv: int = 5):

    if isinstance(model, ClassifierMixin):
        splitter = StratifiedKFold(cv, shuffle=True, random_state=42)
    else:
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

    # plt.figure(figsize=(10, 6))
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

def plot_calibration_curve(y: pd.DataFrame, model_name: str, target: str, n_bins=11, strategy='uniform') -> None:

    prob_true, prob_pred = calibration_curve(y[target], y[f"{model_name}_prob"], n_bins=n_bins, strategy=strategy)

    # plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def summarize_metric_results(results: dict[str, dict[str, float]]) -> pd.DataFrame:
        
    rows = []
    
    for dataset, metrics_dict in results.items():
        row = {"Dataset": dataset}
        row.update(metrics_dict)
        rows.append(row)
    
    return pd.DataFrame(rows)

def analyze_model(model_name: str, model: BaseEstimator, results: dict, X_train: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, target: str, scoring: str) -> None:
    
    print(f"{model_name} Results")

    display(summarize_metric_results(results))

    if isinstance(model, RegressorMixin):
        pred_col = f"{model_name}_pred"
        
        plot_residuals(y_test, pred_col, target)
        plot_pred_vs_true(y_test, pred_col, target)
        plot_error_by_quantile(y_test, pred_col, target)

    elif isinstance(model, ClassifierMixin):
        pred_col = f"{model_name}_pred"
        prob_col = f"{model_name}_prob"
        
        plot_roc_curve(y_test, prob_col, target)
        plot_precision_recall_curve(y_test, prob_col, target)
        plot_calibration_curve(y_test, model_name, target, strategy='uniform')

    else:
        raise ValueError("Model needs to be a classifier or regressor.")

    plot_learning_curve(model, X_train, y_train, target, scoring=scoring)
    plot_feature_importance(model)
    plot_permutation_importance(model, X_train, y_train[target], scoring=scoring)
    plot_shap_summary(model, X_train)
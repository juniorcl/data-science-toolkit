import shap
import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats             import ks_2samp
from sklearn.base            import BaseEstimator
from sklearn.inspection      import permutation_importance
from sklearn.calibration     import calibration_curve
from sklearn.model_selection import learning_curve, KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target


def ks_score(y_true, y_prob):
    pos_prob = y_prob[y_true == 1]
    neg_prob = y_prob[y_true == 0]
    if len(pos_prob) == 0 or len(neg_prob) == 0:
        return 0.0
    result = ks_2samp(pos_prob, neg_prob)
    return result.statistic

ks_scorer = metrics.make_scorer(ks_score, greater_is_better=True, response_method="predict_proba")

def get_class_metrics(y, pred_col, prob_col, target):
    y_true = y[target]
    y_pred = y[pred_col]
    y_prob = y[prob_col]
    return {
        'Balanced Accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'Precision': metrics.precision_score(y_true, y_pred),
        'Recall': metrics.recall_score(y_true, y_pred),
        'F1': metrics.f1_score(y_true, y_pred),
        'AUC': metrics.roc_auc_score(y_true, y_prob),
        'KS': ks_score(y_true, y_prob),
        'Brier': metrics.brier_score_loss(y_true, y_prob),
        'LogLoss': metrics.log_loss(y_true, y_prob)
    }

def get_reg_metrics(y, pred_col, target):
    y_true = y[target]
    y_pred = y[pred_col]
    return {
        'R2': metrics.r2_score(y_true, y_pred),
        'MAE': metrics.mean_absolute_error(y_true, y_pred),
        'MadAE': metrics.median_absolute_error(y_true, y_pred),
        'MAPE': metrics.mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': metrics.root_mean_squared_error(y_true, y_pred),
        'Explained Variance': metrics.explained_variance_score(y_true, y_pred)
    }

def plot_permutation_importance(model, features, X, y, scoring):
    permu_results = permutation_importance(model, X[features], y, scoring=scoring, n_repeats=5, random_state=42)
    sorted_importances_idx = permu_results.importances_mean.argsort()
    df_results = pd.DataFrame(permu_results.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
    ax = df_results.plot.box(vert=False, whis=10, patch_artist=True, boxprops={'facecolor':'skyblue', 'color':'blue'})
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel(f"Decrease in {scoring}")
    plt.show()

def plot_feature_importance(model, top_n=20):
    if hasattr(model, 'feature_name_'):  # LightGBM
        feature_names = model.feature_name_
    elif hasattr(model, 'feature_names_'):  # CatBoost
        feature_names = model.feature_names_
    elif hasattr(model, 'feature_names_in_'):  # XGBoost or sklearn-like models
        feature_names = model.feature_names_in_
    else:
        raise AttributeError("It was not possible to identify the feature names in the model.")

    if hasattr(model, 'feature_importances_'):  # LightGBM, XGBoost, CatBoost (modo sklearn)
        importances = model.feature_importances_
    elif hasattr(model, 'get_score'):  # XGBoost nativo (booster)
        importances_dict = model.get_score(importance_type='weight')
        feature_names = list(importances_dict.keys())
        importances = list(importances_dict.values())
    else:
        raise AttributeError("Model doesn't support featue importance attribute.")

    df_imp = pd.DataFrame({"Variable": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, min(top_n, 20) * 0.4 + 2))
    sns.barplot(x="Importance", y="Variable", data=df_imp.head(top_n), color="#006e9cff")
    plt.title("Importance of Variables")
    plt.tight_layout()
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
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X,
        y[target],
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=KFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Train')
    plt.fill_between(
        train_sizes_abs, train_scores_mean - train_scores_std, 
        train_scores_mean + train_scores_std, alpha=0.1, color='r'
    )

    plt.plot(train_sizes_abs, val_scores_mean, 'o-', color='b', label='Validation')
    plt.fill_between(
        train_sizes_abs, val_scores_mean - val_scores_std, 
        val_scores_mean + val_scores_std, alpha=0.1, color='b'
    )

    plt.title("Learning Curve")
    plt.xlabel("Train Size")
    plt.ylabel(f"{scoring}")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y, prob_col, target):
    fpr, tpr, _ = metrics.roc_curve(y[target], y[prob_col])
    plt.plot(fpr, tpr)
    plt.title('ROC AUC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()

def plot_precision_recall_curve(y, prob_col, target):
    precision, recall, _ = metrics.precision_recall_curve(y[target], y[prob_col])
    plt.plot(precision, recall)
    plt.title('Precision Recall Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.show()

def plot_calibration_curve(y, target, n_bins=10, strategy='uniform'):
    prob_true, prob_pred = calibration_curve(y[target], y['prob'], n_bins=n_bins, strategy=strategy)
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_class_eval_scoring(scoring, return_func=True):
    scorers = {
        'ks': ks_scorer,
        'roc_auc': 'roc_auc',
        'brier': 'neg_brier_score',
        'log_loss': 'neg_log_loss',
    }
    functions = {
        'ks': ks_score,
        'brier': lambda y_true, y_pred: -1 * metrics.brier_score_loss(y_true, y_pred),
        'roc_auc': metrics.roc_auc_score,
        'log_loss': lambda y_true, y_pred: -1 * metrics.log_loss(y_true, y_pred)
    }
    if scoring not in scorers or scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    return functions[scoring] if return_func else scorers[scoring]

def get_reg_eval_scoring(scoring, return_func=True):
    scorers = {
        'r2': 'r2',
        'explained_variance': 'explained_variance',
        'mean_absolute_error': 'neg_mean_absolute_error',
        'median_absolute_error': 'neg_median_absolute_error',
        'root_mean_squared_error': 'neg_root_mean_squared_error',
        'mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error'
    }
    functions = {
        'r2': metrics.r2_score,
        'explained_variance': metrics.explained_variance_score,
        'mean_absolute_error': lambda y_true, y_pred: -1*metrics.mean_absolute_error(y_true, y_pred),
        'median_absolute_error': lambda y_true, y_pred: -1*metrics.median_absolute_error(y_true, y_pred),
        'root_mean_squared_error': lambda y_true, y_pred: -1*metrics.root_mean_squared_error(y_true, y_pred),
        'mean_absolute_percentage_error': lambda y_true, y_pred: -1*metrics.mean_absolute_percentage_error(y_true, y_pred)
    }
    if scoring not in scorers or scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")
    return functions[scoring] if return_func else scorers[scoring]

def plot_ks_curve(y, target, bins=100):
    df = pd.DataFrame({"y_true": y[target], "y_proba": y['prob']})
    
    df = df.sort_values("y_proba")
    
    total_1 = (df["y_true"] == 1).sum()
    total_0 = (df["y_true"] == 0).sum()
    
    df["cum_1"] = (df["y_true"] == 1).cumsum() / total_1
    df["cum_0"] = (df["y_true"] == 0).cumsum() / total_0
    
    # Calc KS
    df["diff"] = np.abs(df["cum_1"] - df["cum_0"])
    ks_value = df["diff"].max()
    ks_idx = df["diff"].idxmax()
    ks_threshold = df.loc[ks_idx, "y_proba"]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["y_proba"], df["cum_1"], label="Classe 1 (Acumulada)", color="tab:blue")
    plt.plot(df["y_proba"], df["cum_0"], label="Classe 0 (Acumulada)", color="tab:orange")
    plt.vlines(ks_threshold, ymin=df.loc[ks_idx, "cum_0"], ymax=df.loc[ks_idx, "cum_1"],
               colors="red", linestyles="--", label=f"KS = {ks_value:.3f}")
    
    plt.title("Curvas de Distribuição Acumulada (KS Test)")
    plt.xlabel("Probabilidade prevista da classe 1")
    plt.ylabel("Proporção acumulada")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def analyze_reg_model(model, features, X_train, y_train, y_test, target, scoring):
    pred_col = 'pred'
    plot_residuals(y_test, pred_col, target)
    plot_pred_vs_true(y_test, pred_col, target)
    plot_error_by_quantile(y_test, pred_col, target)
    plot_learning_curve(model, X_train, y_train, target, scoring=scoring)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model)
    plot_permutation_importance(model, features, X_train, y_train[target], scoring=scoring)
    plot_shap_summary(model, features, X_train)

def analyze_class_model(model, features, X_train, y_train, y_test, target, scoring):
    prob_col = 'prob'
    plot_roc_curve(y_test, prob_col, target)
    plot_ks_curve(y_test, target)
    plot_precision_recall_curve(y_test, prob_col, target)
    plot_calibration_curve(y_test, target, strategy='uniform')
    plot_learning_curve(model, X_train, y_train, target, scoring=scoring)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model)
    plot_permutation_importance(model, features, X_train, y_train[target], scoring=scoring)
    plot_shap_summary(model, features, X_train)